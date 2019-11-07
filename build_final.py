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
from functools import partial
#from os import environ

import mxnet as mx

from gluonts.dataset.repository.datasets import get_dataset
#from gluonts.time_feature import HourOfDay, DayOfWeek, DayOfMonth, DayOfYear, MonthOfYear
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer

rand_seed = 42
mx.random.seed(rand_seed, ctx='all')
np.random.seed(rand_seed)

num_eval_samples = 100

########################################################################################################

def evaluate(dataset_name, estimator):
    def convert(row):
        return({"start": "1970-01-01 00:00", "target": row.values, 'id': row.name})  
    
    dataset = get_dataset(dataset_name)
    estimator = estimator(prediction_length=dataset.metadata.prediction_length, freq=dataset.metadata.freq)
    estimator.ctx = mx.Context("gpu")

    logger.info(f"evaluating {estimator} on {dataset}")

    predictor = estimator.train(dataset.train)
    predictor.ctx = mx.Context("gpu")

    forecast_it, ts_it = make_evaluation_predictions(dataset.test, predictor=predictor, num_eval_samples=num_eval_samples)
    agg_metrics, item_metrics = Evaluator()(ts_it, forecast_it, num_series=len(dataset.test))
    eval_dict = agg_metrics
    eval_dict["dataset"] = dataset_name
    eval_dict["estimator"] = type(estimator).__name__
    return eval_dict

if __name__ == "__main__":
#		"loss" : 0.9078119258939598,
#		"status" : "ok",
#		"dataset" : "m4_hourly"
    cfg = {
            "batch_size" : 128,
            "dropout_rate" : 0.11487556921918471,
            "learning_rate" : 0.0011531060046036214,
            "learning_rate_decay_factor" : 0.45070831883744666,
            "max_epochs" : 5000,
            "minimum_learning_rate" : 0.000053239690820732165,
            "num_batches_per_epoch" : 70,
            "num_cells" : 400,
            "num_layers" : 3,
            "weight_decay" : 4.3229937548659974e-8,
            "patience" : 10,
    }

#		"loss" : 0.8356107397498784,
#		"status" : "ok",
#		"cfg" : {
#			"batch_size" : 128,
#			"dropout_rate" : 0.06648341605432148,
#			"learning_rate" : 0.0010602372763356665,
#			"learning_rate_decay_factor" : 0.7104804575647294,
#			"max_epochs" : 5000,
#			"minimum_learning_rate" : 0.00005562470061533188,
#			"num_batches_per_epoch" : 70,
#			"num_cells" : 400,
#			"num_layers" : 5,
#			"weight_decay" : 4.818670454712425e-8
#		},

    estimator = partial(
        DeepAREstimator,
        num_cells=cfg['num_cells'],
        num_layers=cfg['num_layers'],
        dropout_rate=cfg['dropout_rate'],
        use_feat_static_cat=True,
        cardinality=[4227, 6], # dataset.metadata.feat_static_cat[0].cardinality, dataset.metadata.feat_static_cat[1].cardinality
#            time_features=time_features, 
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

    results = evaluate(dataset_name, estimator)
    logger.info(results)