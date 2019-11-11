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
from datetime import date, timedelta
from hyperopt import fmin, tpe, hp, space_eval, STATUS_FAIL, STATUS_OK
from hyperopt.mongoexp import MongoTrials
from os import environ
import sys
from math import log
#from statistics import mean

import mxnet as mx
from gluonts.dataset.common import ListDataset
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.transformer import TransformerEstimator
from gluonts.model.deep_factor import DeepFactorEstimator
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
freq="1D"
prediction_length = 12
max_sMSE = 4.0

sample_data = [pd.read_csv("sample%d.csv" % idx, index_col=0) for idx in range(1, 6)]
    
########################################################################################################
    
def gluon_fcast(cfg):        
    #   errors_mse <- c(errors_mse, mean((as.numeric(rep(frc[i,j], fh))-as.numeric(outsample))^2)/(mean(insample)^2))
    def sMSE(insample, outsample, y_hat):
        err = np.mean((y_hat - outsample) * (y_hat - outsample)) / (np.mean(insample) * np.mean(insample))
        if np.isnan(err):
            raise ValueError("sMSE returned NaN")
        return err
         
    def compute_errors(train, test, y_hats):
        errs = []
        for idx in range(len(y_hats)):
            errs.append(sMSE(train.iloc[idx].dropna().values, test.iloc[idx].values, y_hats.iloc[idx].values))
        return np.mean(errs)
            
    def convert_7day(ts_6day):
#        print(ts_6day)
        ts_7day = pd.Series()
        num_weeks = int(len(ts_6day)/6) + 1
        for idx in range(num_weeks):
            week = ts_6day[(idx*6):(idx+1)*6]
            ts_7day = pd.concat([ts_7day, pd.Series([np.NaN]), week])
            
        ts_7day.index = pd.date_range(start="2017-01-01 00:00", periods=len(ts_7day), freq=freq)
        first_valid = ts_7day[ts_7day.notnull()].index[0]
        
#        if (ts_6day.name == 'id331225'):
#            print(ts_6day.tail(18))
#            print(ts_7day.tail(21))
#            print(first_valid)
            
        return {
            "id"     : ts_6day.name,
            "start"  : str(first_valid),
            "end"    : str(ts_7day.index[-1]),
            "target" : ts_7day[first_valid:].fillna(0).values, 
        }    
    
    def add_static_idx(ts, idx):
        ts['feat_static_cat'] = [idx]
        return ts
        
    def forecast(data, cfg):
        logger.info("Params: %s " % cfg)
#        print(data.describe())

        train_6day = data.iloc[ : , :-prediction_length]
        train_7day = train_6day.apply(convert_7day, axis=1).tolist()
        
        # Add a static index categorical
        train_7day_cat_idx = [add_static_idx(train_7day[idx], idx) for idx in range(len(train_7day))]
        gluon_train = ListDataset(train_7day_cat_idx, freq=freq)

#        trainer=Trainer(
#            mx.Context("gpu"),
#            epochs=10,
#        )
        trainer=Trainer(
            mx.Context("gpu"),
            epochs=cfg['trainer']['max_epochs'],
            num_batches_per_epoch=cfg['trainer']['num_batches_per_epoch'],
            batch_size=cfg['trainer']['batch_size'],
            patience=cfg['trainer']['patience'],
            
            learning_rate=cfg['trainer']['learning_rate'],
            learning_rate_decay_factor=cfg['trainer']['learning_rate_decay_factor'],
            minimum_learning_rate=cfg['trainer']['minimum_learning_rate'],
            weight_decay=cfg['trainer']['weight_decay'],
        )
              
        if cfg['model']['type'] == 'SimpleFeedForwardEstimator':
            estimator = SimpleFeedForwardEstimator(
                freq=freq,
                prediction_length=prediction_length+2, # 7 day week
                
                num_hidden_dimensions = cfg['model']['num_hidden_dimensions'],

                trainer=trainer)

        if cfg['model']['type'] == 'DeepAREstimator':            
            estimator = DeepAREstimator(
                freq=freq,
                prediction_length=prediction_length+2, # 7 day week        
                num_cells=cfg['model']['num_cells'],
                num_layers=cfg['model']['num_layers'],        
                dropout_rate=cfg['model']['dar_dropout_rate'],
                use_feat_static_cat=True,
                cardinality=[len(train_7day_cat_idx)],
                trainer=trainer)
            
        if cfg['model']['type'] == 'TransformerEstimator': 
             estimator = TransformerEstimator(
                freq=freq,
                prediction_length=prediction_length+2, # 7 day week
                model_dim=cfg['model']['model_dim'], 
                inner_ff_dim_scale=cfg['model']['inner_ff_dim_scale'],
                pre_seq=cfg['model']['pre_seq'], 
                post_seq=cfg['model']['post_seq'], 
                act_type=cfg['model']['act_type'], 
                num_heads=cfg['model']['num_heads'], 
                dropout_rate=cfg['model']['trans_dropout_rate'],
                use_feat_static_cat=True,
                cardinality=[len(train_7day_cat_idx)],
                trainer=trainer)

        if cfg['model']['type'] == 'DeepFactorEstimator': 
             estimator = DeepFactorEstimator(
                freq=freq,
                prediction_length=prediction_length+2, # 7 day week
                num_hidden_global=cfg['model']['num_hidden_global'], 
                num_layers_global=cfg['model']['num_layers_global'], 
                num_factors=cfg['model']['num_factors'], 
                num_hidden_local=cfg['model']['num_hidden_local'], 
                num_layers_local=cfg['model']['num_layers_local'], 
                trainer=trainer)
             
        logger.info("Fitting: %s" % cfg['model']['type'])
        model = estimator.train(gluon_train)
        
        # Get predections
        preds_iter = model.predict(gluon_train, num_eval_samples=1)
        y_hats = [ts.samples.reshape(prediction_length+2) for ts in preds_iter]
        
        # Add DateTime index to y_hats so we can drop Sundays
        y_hats_start_date = pd.to_datetime(train_7day[0]['end']) + timedelta(days=1)
        y_hats_srs = [pd.Series(y_hat, pd.date_range(start=y_hats_start_date, periods=prediction_length+2, freq=freq)) for y_hat in y_hats]
        y_hats_6day = pd.DataFrame([y_hat_srs[y_hat_srs.index.day_name() != "Sunday"].values for y_hat_srs in y_hats_srs])
        
        # Compute errors
        test_6day = data.iloc[ : , -prediction_length:]
        sMSE_n = compute_errors(train_6day, test_6day, y_hats_6day)
        logger.info("sMSE : %.3f" % sMSE_n)
        return sMSE_n

    ########################################################################################################
    
    results = []
    for idx in range(1, 6):
        logger.info("Sample # %s" % idx)
        try:
            sMSE_idx = forecast(sample_data[idx-1].iloc[ : , : -prediction_length], cfg) # Drop last prediction_length of sample data for final evaluation
            results.append(sMSE_idx)
            if idx > 1 and np.mean(results) > max_sMSE:
                logger.warning("Aborting run due to high mean(sMSE) = %.3f > %.3f" % (np.mean(results), max_sMSE))
                return {'loss': None, 'status': STATUS_FAIL, 'cfg' : cfg, 'results': results, 'build_url' : environ.get("BUILD_URL")}
            
        except Exception as e:
            exc_str = 'Warning on line %d, exception: %s' % (sys.exc_info()[-1].tb_lineno, str(e))
            logger.error(exc_str)
            return {'loss': None, 'status': STATUS_FAIL, 'cfg' : cfg, 'results': results, 'exception': exc_str, 'build_url' : environ.get("BUILD_URL")}
        
    logger.info("sMSEs per sample: %s" % [round(result, 3) for result in results])
    sMSE_final = np.mean(results)
    logger.info("Final sMSE: %.3f" % sMSE_final)
    return {'loss': sMSE_final, 'status': STATUS_OK, 'cfg' : cfg, 'results': results, 'build_url' : environ.get("BUILD_URL")}

def call_hyperopt():
    dropout_rate = [0.05, 0.15]
    space = {
        'trainer' : {
            'max_epochs'                 : hp.choice('max_epochs', [125, 250, 500, 1000, 2000, 4000]),
            'num_batches_per_epoch'      : hp.choice('num_batches_per_epoch', [25, 50, 100]),
            'batch_size'                 : hp.choice('batch_size', [25, 50, 100, 200]),
            'patience'                   : hp.choice('patience', [5, 10, 20, 40]),
            
            'learning_rate'              : hp.uniform('learning_rate', 1e-04, 1e-02),
            'learning_rate_decay_factor' : hp.uniform('learning_rate_decay_factor', 0.3, 0.8),
            'minimum_learning_rate'      : hp.loguniform('minimum_learning_rate', log(1e-06), log(0.5e-04)),
            'weight_decay'               : hp.uniform('weight_decay', 0.5e-08, 5.0e-08),
        },
        'model' : hp.choice('model', [
                    {
                        'type'                       : 'SimpleFeedForwardEstimator',
                        'num_hidden_dimensions'      : hp.choice('num_hidden_dimensions', [[25], [50], [100],
                                                                                           [25, 25], [50, 25], [50, 50], [100, 50], [100, 100],
                                                                                           [100, 50, 50]])
                    },
                    {
                        'type'                       : 'DeepAREstimator',
                        'num_cells'                  : hp.choice('num_cells', [25, 50, 100, 200, 400]),
                        'num_layers'                 : hp.choice('num_layers', [1, 3, 5, 7]),
                        
                        'dar_dropout_rate'           : hp.uniform('dar_dropout_rate', dropout_rate[0], dropout_rate[1]),
                    },
                    {
                        'type'                       : 'TransformerEstimator',
                        'model_dim'                  : hp.choice('model_dim', [16, 32, 64]),
                        'inner_ff_dim_scale'         : hp.choice('inner_ff_dim_scale', [3, 4, 5]),
                        'pre_seq'                    : hp.choice('pre_seq', ['dn']),
                        'post_seq'                   : hp.choice('post_seq', ['drn']),
                        'act_type'                   : hp.choice('act_type', ['softrelu']),
                        'num_heads'                  : hp.choice('num_heads', [4, 8, 16]),
                       
                        'trans_dropout_rate'         : hp.uniform('trans_dropout_rate', dropout_rate[0], dropout_rate[1]),
                    },
                    {
                        'type'                       : 'DeepFactorEstimator',
                        'num_hidden_global'          : hp.choice('num_hidden_global', [25, 50, 100, 200]),
                        'num_layers_global'          : hp.choice('num_layers_global', [1, 2, 3]),
                        'num_factors'                : hp.choice('num_factors', [5, 10, 20]),
                        'num_hidden_local'           : hp.choice('num_hidden_local', [2, 5, 10]),
                        'num_layers_local'           : hp.choice('num_layers_local', [1, 2, 3]),
                    },
                ])
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
        best = fmin(gluon_fcast, space, algo=tpe.suggest, show_progressbar=False, max_evals=20)
         
    return space_eval(space, best) 
    
if __name__ == "__main__":
    params = call_hyperopt()
    logger.info("Best params: %s" % params)