#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:11:38 2019

@author: mulderg
"""
import mxnet as mx
from gluonts.dataset.common import ListDataset
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer
#from gluonts.evaluation.backtest import make_evaluation_predictions
#from gluonts.evaluation import Evaluator
#import json
from statistics import mean
from datetime import timedelta

import pandas as pd
import numpy as np

rand_seed = 42
mx.random.seed(rand_seed, ctx='all')
np.random.seed(rand_seed)

freq="1D"
prediction_length = 12

#   errors_mse <- c(errors_mse, mean((as.numeric(rep(frc[i,j], fh))-as.numeric(outsample))^2)/(mean(insample)^2))
def sMSE(insample, outsample, y_hat):
    error = np.mean((y_hat - outsample) * (y_hat - outsample)) / (np.mean(insample) * np.mean(insample))
    return(error)
     
def compute_errors(train, test, y_hats):
    errors = []
    for idx in range(len(y_hats)):
        errors.append(sMSE(train.iloc[idx], test.iloc[idx], y_hats[idx]))
    return(mean(errors))
        
def convert_7day(row):
#    print(row)
    full_ts = pd.Series()
    num_weeks = int(len(row)/6) + 1
    for idx in range(num_weeks):
        week6 = row[(idx*6):(idx+1)*6]
        full_ts = pd.concat([full_ts, pd.Series([np.NaN]), week6])  
    full_ts.index = pd.DatetimeIndex(start="2017-01-01 00:00", periods=len(full_ts), freq=freq)
    first_valid = full_ts[full_ts.notnull()].index[0]
    
    if (row.name == 'id331225'):
        print(row.tail(18))
        print(full_ts.tail(21))
        print(first_valid)
        
    return({
        "id"     : row.name,
        "start"  : str(first_valid),
        "end"    : str(full_ts.index[-1]),
        "target" : full_ts[first_valid:].fillna(0).values, 
    })    
    
def forecast(data):
    print(data.describe())
    train_6day = data.iloc[:, : -prediction_length]
    train_7day = train_6day.apply(convert_7day, axis=1).tolist()
    gluon_train = ListDataset(train_7day, freq=freq)

    estimator = SimpleFeedForwardEstimator(
        freq=freq,
        prediction_length=prediction_length+2, # 7 day week
        num_hidden_dimensions = [28],
        trainer=Trainer(epochs=200, num_batches_per_epoch=500),
        
    )
    model = estimator.train(gluon_train)
    
    # Get predections
    preds_iter = model.predict(gluon_train, num_eval_samples=1)
    y_hats = [ts.samples.reshape(prediction_length+2) for ts in preds_iter]
    
    # Add DateTime index to y_hats so we can drop Sundays
    y_hats_start_date = pd.to_datetime(train_7day[0]['end']) + timedelta(days=1)
    y_hats_srs = [pd.Series(y_hat, pd.DatetimeIndex(start=y_hats_start_date, periods=prediction_length+2, freq=freq)) for y_hat in y_hats]
    y_hats_6day = [y_hat_srs[y_hat_srs.index.day_name() != "Sunday"].values for y_hat_srs in y_hats_srs]
    
    # Compute errors
    test_6day = data.iloc[:, -prediction_length:]
    sMSE = compute_errors(train_6day, test_6day, y_hats_6day)
    print("sMSE : %.3f" % sMSE)
    return(sMSE)
                                 
sample_data = [pd.read_csv("sample%d.csv" % idx, index_col=0) for idx in range(1,5)]
sMSEs = forecast([forecast(sample) for sample in sample_data])
print(sMSEs)
print(mean(sMSEs))