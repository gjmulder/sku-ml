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

#import numpy as np
import pandas as pd
import sys
#from statistics import mean

import mxnet as mx
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer

num_eval_samples = 1
freq="1D"
prediction_length = 12 # Days
sample_data = [pd.read_csv("sample%d.csv" % idx, index_col=0) for idx in range(1,6)]

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
        
    logger.info("Params: %s" % cfg)
    predictions = []
    for idx in range(0, 5):
        sample_name = "sample%d.csv" % (idx + 1)
        logger.info("Building final model for %s" % sample_name)
        data_train = sample_data[idx].iloc[:, :-2 * prediction_length].apply(convert, axis=1).tolist()
        data_test  = sample_data[idx].iloc[:, :-1 * prediction_length].apply(convert, axis=1).tolist()
        gluon_data_train = ListDataset(data_train, freq=freq)
        gluon_data_test = ListDataset(data_test, freq=freq)
    
        try:    
            estimator = DeepAREstimator(
                freq=freq,
                prediction_length=prediction_length,
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
            
            predictor = estimator.train(gluon_data_train)
    
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=gluon_data_test,
                predictor=predictor,
                num_eval_samples=1,
            )
            
#            logger.info("sMSE = %.3f, sME = %.3f" % (sMSE, sME))
#            logger.info("MSE = %.3f" % agg_metrics["MSE"])
            predictions.append(list(forecast_it))
        except Exception as e:
            logger.warning('Warning on line %d, exception: %s' % (sys.exc_info()[-1].tb_lineno, str(e)))
            return None

    logger.info(predictions)
    return predictions

if __name__ == "__main__":
    cfg =  {
        "batch_size" : 32,
        "dropout_rate" : 0.07883374392453434,
        "learning_rate" : 0.02851313877411612,
        "learning_rate_decay_factor" : 0.7634826582117775,
        "max_epochs" : 5,
        "minimum_learning_rate" : 0.000003534074768081984,
        "num_batches_per_epoch" : 50,
        "num_cells" : 200,
        "num_layers" : 3,
        "patience" : 32,
        "weight_decay" : 3.165855238029401e-8
    }

    predictions = gluon_fcast(cfg)