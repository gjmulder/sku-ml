#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:11:38 2019

@author: mulderg
"""
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator

#import json
import pandas as pd

freq="6D" # 6 days
prediction_length = 12 # Days

def convert(row):
#    print(row)
    return({"start": "1970-01-01 00:00", "target": row.values, 'id': row.name})    
    
def forecast_sample(data):
    data_train = sample_data[0].iloc[:, :-prediction_length].apply(convert, axis=1).tolist()
    data_test = sample_data[0].apply(convert, axis=1).tolist()
    
    gluon_data_train = ListDataset(data_train, freq=freq)
    gluon_data_test = ListDataset(data_test, freq=freq)
    
    #train_entry = next(iter(gluon_data_train))
    estimator = DeepAREstimator(
        freq=freq,
        prediction_length=prediction_length,
        trainer=Trainer(epochs=10, num_batches_per_epoch=10)
    )
    predictor = estimator.train(gluon_data_train)
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=gluon_data_test,
        predictor=predictor,
        num_eval_samples=1,
    )
    evaluator = Evaluator()
    agg_metrics, item_metrics = evaluator(ts_it, forecast_it, num_series=len(gluon_data_test))
#    print(json.dumps(agg_metrics, indent=4))
    return(agg_metrics["MSE"])

sample_data = [pd.read_csv("sample%d.csv" % idx, index_col=0) for idx in range(1,5)]

forecasts = [forecast_sample(data) for data in sample_data]
print(forecasts)