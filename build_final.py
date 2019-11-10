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
from datetime import timedelta
from os import environ
import sys

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
    
       
#   errors_mse <- c(errors_mse, mean((as.numeric(rep(frc[i,j], fh))-as.numeric(outsample))^2)/(mean(insample)^2))
def sMSE(insample, outsample, y_hat):
    err = np.mean((y_hat - outsample) * (y_hat - outsample)) / (np.mean(insample) * np.mean(insample))
    return(err)
     
def compute_errors(train, test, y_hats):
    errs = []
    for idx in range(len(y_hats)):
        errs.append(sMSE(train.iloc[idx], test.iloc[idx], y_hats[idx]))
    return(np.mean(errs))
        
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
        
    return({
        "id"     : ts_6day.name,
        "start"  : str(first_valid),
        "end"    : str(ts_7day.index[-1]),
        "target" : ts_7day[first_valid:].fillna(0).values, 
    })    

def add_static_idx(ts, idx):
    ts['feat_static_cat'] = [idx]
    return(ts)
    
def forecast(data, cfg):
    logger.info("Params: %s " % cfg)
#        print(data.describe())

    train_6day =  train_6day = data.iloc[ : , : -prediction_length]
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
    y_hats_6day = [y_hat_srs[y_hat_srs.index.day_name() != "Sunday"].values for y_hat_srs in y_hats_srs]
    
    # Compute errors
    test_6day = data.iloc[ : , -prediction_length:]
    sMSE_n = compute_errors(train_6day, test_6day, y_hats_6day)
    logger.info("sMSE : %.3f" % sMSE_n)
    return(sMSE_n)

########################################################################################################

    results = []
    for idx in range(1, 6):
        logger.info("Sample # %s" % idx)
        sMSE_idx = forecast(sample_data[idx-1].iloc[ : , : -prediction_length], cfg) # Drop last prediction_length of sample data for final evaluation
        results.append(sMSE_idx)
                   
    logger.info("sMSEs per sample: %s" % [round(result, 3) for result in results])
    sMSE_final = np.mean(results)
    logger.info("Final sMSE: %.3f" % sMSE_final)
    
if __name__ == "__main__":
    cfg = 
      "result" : {
                "loss" : 2.6898342036076097,
                "status" : "ok",
                "cfg" : {
                        "model" : {
                                "act_type" : "softrelu",
                                "inner_ff_dim_scale" : 5,
                                "model_dim" : 16,
                                "num_heads" : 8,
                                "post_seq" : "drn",
                                "pre_seq" : "dn",
                                "trans_dropout_rate" : 0.1072951908699155,
                                "type" : "TransformerEstimator"
                        },
                        "trainer" : {
                                "batch_size" : 200,
                                "learning_rate" : 0.0014557112535426254,
                                "learning_rate_decay_factor" : 0.7622377071816162,
                                "max_epochs" : 500,
                                "minimum_learning_rate" : 0.0000018515726634949832,
                                "num_batches_per_epoch" : 25,
                                "patience" : 40,
                                "weight_decay" : 2.6747692257944095e-8
                        }
                },
    forecast(sample_data, cfg)
