#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:09:36 2019

@author: mulderg
"""

from logging import basicConfig, getLogger
#from logging import DEBUG as log_level
from logging import INFO as log_level
basicConfig(level = log_level,
            format  = '%(asctime)s %(levelname)-8s %(module)-20s: %(message)s',
            datefmt ='%Y-%m-%d %H:%M:%S')
logger = getLogger(__name__)

import numpy as np
import pandas as pd
from datetime import date
from hyperopt import fmin, tpe, hp, space_eval, STATUS_FAIL, STATUS_OK
from hyperopt.mongoexp import MongoTrials
from functools import partial
from os import environ
import sys
from math import log
from statistics import mean

import mxnet as mx
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer

rand_seed = 42
mx.random.seed(rand_seed, ctx='all')
np.random.seed(rand_seed)
    
########################################################################################################

if "VERSION" in environ:    
    version = environ.get("VERSION")
    logger.info("Using version : %s" % version)
    
    use_cluster = True
else:
    version = "test"
    logger.warning("VERSION not set, using: %s" % version)
    
    use_cluster = False

#if dataset_name == "m4_daily":
#    time_features = [DayOfWeek(), DayOfMonth(), DayOfYear(), MonthOfYear()]
#if dataset_name == "m4_hourly":
##    time_features = [HourOfDay(), DayOfWeek(), DayOfMonth(), DayOfYear(), MonthOfYear()]
#    time_features = [HourOfDay(), DayOfWeek()]

num_eval_samples = 1
freq="6D" # 6 days
prediction_length = 12 # Days
sample_data = [pd.read_csv("sample%d.csv" % idx, index_col=0) for idx in range(1,5)]
    
########################################################################################################
    
def gluon_fcast(cfg):        
    def convert(row):
        return({"start": "1970-01-01 00:00", "target": row.values, 'id': row.name})  

    def evaluate(sample_name, gluon_data_train, gluon_data_test, estimator):                
        estimator = estimator(prediction_length=prediction_length, freq=freq)
        estimator.ctx = mx.Context("gpu")
        
        logger.info(f"Evaluating {estimator} on {sample_name}")
        predictor = estimator.train(data_train)
        predictor.ctx = mx.Context("gpu")
        
        forecast_it, ts_it = make_evaluation_predictions(gluon_data_test, predictor=predictor, num_eval_samples=num_eval_samples)
        
        agg_metrics, item_metrics = Evaluator()(ts_it, forecast_it, num_series=len(gluon_data_test))
#        sMSE = agg_metrics["MSE"] / agg_metrics["target_sum_squared"]
        logger.info("Name: %s, MSE: %.3f" % (sample_name, agg_metrics["MSE"]))
        return agg_metrics["MSE"]

    ########################################################################## 

    if not use_cluster:
        cfg['num_cells'] = 10
        cfg['num_layers'] = 1
        cfg['max_epochs'] = 10
        
    logger.info("Params: %s" % cfg)
    results = []
    for idx in range(1, 5):
        sample_name = "sample%d.csv" % idx
        data_train = sample_data[0].iloc[:, :-2 * prediction_length].apply(convert, axis=1).tolist()
        data_test  = sample_data[0].iloc[:, :-1 * prediction_length].apply(convert, axis=1).tolist()
        gluon_data_train = ListDataset(data_train, freq=freq)
        gluon_data_test = ListDataset(data_test, freq=freq)
    
        try:    
            estimator = partial(
                DeepAREstimator,
                num_cells=cfg['num_cells'],
                num_layers=cfg['num_layers'],
                dropout_rate=cfg['dropout_rate'],
                trainer=Trainer(
                    mx.Context("gpu"),
                    epochs=cfg['max_epochs'],
                    num_batches_per_epoch=cfg['num_batches_per_epoch'],
                    batch_size=cfg['batch_size'],
                    patience=cfg['patience'],
                    
                    learning_rate=cfg['learning_rate'],
                    learning_rate_decay_factor=cfg['learning_rate_decay_factor'],
                    minimum_learning_rate=cfg['minimum_learning_rate'],
                    weight_decay=cfg['weight_decay']
                ))
            results.append(evaluate(sample_name, gluon_data_train, gluon_data_test, estimator))
        except Exception as e:
            logger.warning('Warning on line %d, exception: %s' % (sys.exc_info()[-1].tb_lineno, str(e)))
            return {'loss': None, 'status': STATUS_FAIL, 'cfg' : cfg, 'build_url' : environ.get("BUILD_URL"), 'dataset': sample_name}

    logger.info(results)
    return {'loss': mean(results), 'status': STATUS_OK, 'cfg' : cfg, 'build_url' : environ.get("BUILD_URL"), 'dataset': sample_name}

def call_hyperopt():
    space = {
        'num_cells'                  : hp.choice('num_cells', [200, 400]),
        'num_layers'                 : hp.choice('num_layers', [3, 5, 7]),
        'dropout_rate'               : hp.uniform('dropout_rate', 0.05, 0.15),

        'max_epochs'                 : hp.choice('max_epochs', [1]),
        'num_batches_per_epoch'      : hp.choice('num_batches_per_epoch', [50, 100]),
        'batch_size'                 : hp.choice('batch_size', [32, 64, 128, 256]),
        'patience'                   : hp.choice('patience', [16, 32, 64, 128]),
        
        'learning_rate'              : hp.uniform('learning_rate', 1e-04, 1e-01),
        'learning_rate_decay_factor' : hp.uniform('learning_rate_decay_factor', 0.1, 0.9),
        'minimum_learning_rate'      : hp.loguniform('minimum_learning_rate', log(1e-06), log(1e-04)),
        'weight_decay'               : hp.uniform('weight_decay', 0.5e-08, 5.0e-08),
    }
    
    # Search MongoDB for best trial for exp_key:
    # echo 'db.jobs.find({"exp_key" : "XXX", "result.status" : "ok"}).sort( { "result.loss": 1} ).limit(1).pretty()' | mongo --host heika m4_daily
    # echo 'db.jobs.remove({"exp_key" : "XXX", "result.status" : "new"})' | mongo --host heika

    if use_cluster:
        exp_key = "%s" % str(date.today())
        logger.info("exp_key for this job is: %s" % exp_key)
        trials = MongoTrials('mongo://heika:27017/sku-%s/jobs' % version, exp_key=exp_key)
        best = fmin(gluon_fcast, space, rstate=np.random.RandomState(rand_seed), algo=tpe.suggest, show_progressbar=False, trials=trials, max_evals=200)
    else:
        best = fmin(gluon_fcast, space, rstate=np.random.RandomState(rand_seed), algo=tpe.suggest, show_progressbar=False, max_evals=5)
        
    params = space_eval(space, best)   
    return(params)
    
if __name__ == "__main__":
    params = call_hyperopt()
    logger.info("Best params: %s" % params)