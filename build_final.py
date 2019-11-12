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
from itertools import repeat

import mxnet as mx
from gluonts.dataset.common import ListDataset
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.transformer import TransformerEstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions

rand_seed = 42
mx.random.seed(rand_seed, ctx='all')
np.random.seed(rand_seed)
    
########################################################################################################

num_eval_samples = 1
freq="1B"
prediction_length = 12
dl_prediction_length=1
max_sMSE = 4.0

sample_data = [pd.read_csv("sample%d.csv" % idx, index_col=0) for idx in range(1, 6)]
    
########################################################################################################
    
       
#   errors_mse <- c(errors_mse, mean((as.numeric(rep(frc[i,j], fh))-as.numeric(outsample))^2)/(mean(insample)^2))
def sMSE(insample, outsample, y_hat):
    if np.isnan(insample).any():
        raise ValueError("np.isnan(insample).any()")
    if np.isnan(outsample).any():
        raise ValueError("np.isnan(outsample).any()")
    if np.isnan(y_hat).any():
        raise ValueError("np.isnan(y_hat).any()")
        
    err = np.mean((y_hat - outsample) * (y_hat - outsample)) / (np.mean(insample) * np.mean(insample))

    if np.isnan(err):
        raise ValueError("sMSE returned NaN")
    return err
     
def compute_errors(train, test, y_hats):
    errs = []
    for idx in range(len(y_hats)):
        errs.append(sMSE(train.iloc[idx].dropna().values, test.iloc[idx].values, y_hats.iloc[idx].values))
    return np.mean(errs)

def load_greek_holidays():
    holidays = pd.read_csv("greek_holidays.csv")
    holidays['date'] = pd.to_datetime(holidays['date'])
    holidays.set_index('date', inplace=True)
    
    dates = pd.date_range("1/1/2017", "31/12/2017", freq="1D")
    all_days = pd.DataFrame({'idx' : [str(idx) for idx in range(1, len(dates)+1)]}, index = dates).join(holidays)
    hols_1_hot = pd.get_dummies(all_days['type'])
    hols_1_hot.set_index(all_days['idx'], inplace=True)
    return(hols_1_hot.transpose())
    
#def convert_7day(ts_6day):
##        print(ts_6day)
#    ts_7day = pd.Series()
#    num_weeks = int(len(ts_6day)/6) + 1
#    for idx in range(num_weeks):
#        week = ts_6day[(idx*6):(idx+1)*6]
#        ts_7day = pd.concat([ts_7day, pd.Series([np.NaN]), week])
#        
#    ts_7day.index = pd.date_range(start="2017-01-01 00:00", periods=len(ts_7day), freq=freq)
#    first_valid = ts_7day[ts_7day.notnull()].index[0]
#    
##        if (ts_6day.name == 'id331225'):
##            print(ts_6day.tail(18))
##            print(ts_7day.tail(21))
##            print(first_valid)
#        
#    return {
#        "id"     : ts_6day.name,
#        "start"  : str(first_valid),
#        "end"    : str(ts_7day.index[-1]),
#        "target" : ts_7day[first_valid:].fillna(0).values, 
#    } 

def ts_to_dict(idx, ts):
    ts_srs = pd.Series(ts, name='ts')
    
    # Find the first non NaN
    first_valid = ts_srs[ts_srs.notnull()].index[0]
    rec = {
        "start"             : "2017-01-01 00:00",
        "target"            : ts_srs[first_valid:].values, 
        "feat_static_cat"   : [idx],
    } 
#    print("Target len: %d, feat_dynamic_real shape: %s" % (len(rec['target']), rec['feat_dynamic_real'].shape))
    return rec

def ts_to_dict_1hot(idx, ts, one_hot):
    one_hot_ts = one_hot.append(pd.Series(ts, index=one_hot.columns, name='ts')).transpose()
    
    # Find the first non NaN
    first_valid = one_hot_ts['ts'][one_hot_ts['ts'].notnull()].index[0]
    rec = {
        "start"             : "2017-01-01 00:00",
        "target"            : one_hot_ts['ts'][first_valid:].values, 
        "feat_static_cat"   : [idx],
        "feat_dynamic_real" : one_hot_ts[first_valid:].drop(['ts'], axis=1).transpose().values,
    } 
#    print("Target len: %d, feat_dynamic_real shape: %s" % (len(rec['target']), rec['feat_dynamic_real'].shape))
    return rec
    
def forecast(data, cfg):
    logger.info("Params: %s " % cfg)
#        print(data.describe())
    dow_1_hot = pd.get_dummies(pd.Series(np.arange(1, data.shape[1] + 6) % 6)).iloc[:data.shape[1],1:].transpose()
    dow_1_hot.columns = data.columns
    hols_1_hot = load_greek_holidays()[data.columns.tolist()]
    all_1_hot = dow_1_hot.append(hols_1_hot)
    
    train_data = data.iloc[ : , :-prediction_length]
    train_1hot = all_1_hot.iloc[ : , :-prediction_length]
    
    train_data_list = train_data.values.tolist()
    if cfg['model']['type'] in ['SimpleFeedForwardEstimator', 'DeepFactorEstimator']:
        train_dict = [ts_to_dict(idx, train_data_list[idx]) for idx in range(len(train_data_list))]
    else:
        train_dict = [ts_to_dict_1hot(idx, train_data_list[idx], train_1hot) for idx in range(len(train_data_list))]
    gluon_train = ListDataset(train_dict, freq=freq)

#    trainer=Trainer(
#        mx.Context("gpu"),
#        epochs=5,
#    )
    
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

    lags_seq=[1,2,3,4,5,6,7,8,9,10,11,12,13, 17,18,19, 23,24,25, 26,27,28, 29,30,31, 35,36,37, 41,42,43, 47,48,49]
    
    if cfg['model']['type'] == 'SimpleFeedForwardEstimator':
        estimator = SimpleFeedForwardEstimator(
            freq=freq,
            prediction_length=dl_prediction_length, 
            num_hidden_dimensions = cfg['model']['num_hidden_dimensions'],
            num_parallel_samples=1,
            trainer=trainer)

    if cfg['model']['type'] == 'DeepFactorEstimator': 
         estimator = DeepFactorEstimator(
            freq=freq,
            prediction_length=dl_prediction_length,
            num_hidden_global=cfg['model']['num_hidden_global'], 
            num_layers_global=cfg['model']['num_layers_global'], 
            num_factors=cfg['model']['num_factors'], 
            num_hidden_local=cfg['model']['num_hidden_local'], 
            num_layers_local=cfg['model']['num_layers_local'],
            trainer=trainer)
         
    if cfg['model']['type'] == 'DeepAREstimator':            
        estimator = DeepAREstimator(
            freq=freq,
            prediction_length=dl_prediction_length,        
            num_cells=cfg['model']['num_cells'],
            num_layers=cfg['model']['num_layers'],        
            dropout_rate=cfg['model']['dar_dropout_rate'],
            use_feat_dynamic_real=True,
            use_feat_static_cat=True,
            cardinality=[len(train_dict)],
            lags_seq=lags_seq,
            num_parallel_samples=1,
            trainer=trainer)
        
    if cfg['model']['type'] == 'TransformerEstimator': 
         estimator = TransformerEstimator(
            freq=freq,
            prediction_length=dl_prediction_length,
            model_dim=cfg['model']['model_dim'], 
            inner_ff_dim_scale=cfg['model']['inner_ff_dim_scale'],
            pre_seq=cfg['model']['pre_seq'], 
            post_seq=cfg['model']['post_seq'], 
            act_type=cfg['model']['act_type'], 
            num_heads=cfg['model']['num_heads'], 
            dropout_rate=cfg['model']['trans_dropout_rate'],
            use_feat_dynamic_real=True,
            use_feat_static_cat=True,
            cardinality=[len(train_dict)],
            lags_seq=lags_seq,
            num_parallel_samples=1,
            trainer=trainer)

    logger.info("Fitting: %s" % cfg['model']['type'])
    model = estimator.train(gluon_train)
    
    test_data = data.iloc[ : , -prediction_length:]
    test_1hot = all_1_hot.iloc[ : , -prediction_length:]
    
    test_data_list = test_data.values.tolist()
    if cfg['model']['type'] in ['SimpleFeedForwardEstimator', 'DeepFactorEstimator']:
        test_dict = [ts_to_dict(idx, test_data_list[idx]) for idx in range(len(test_data_list))]
    else:
        test_dict = [ts_to_dict_1hot(idx, test_data_list[idx], test_1hot) for idx in range(len(test_data_list))]
    gluon_test = ListDataset(test_dict, freq=freq)
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=gluon_test,
        predictor=model,
        num_eval_samples=1,
    )
    
#    tss = list(ts_it)
    y_hats = [yhat.samples.reshape(dl_prediction_length) for yhat in list(forecast_it)]
    y_hats = [repeat(float(yhat), prediction_length) for yhat in y_hats]
    
    # Compute errors        
    test = data.iloc[ : , -prediction_length:]
    sMSE_n = compute_errors(train_data, test, pd.DataFrame(data=y_hats, columns=test.columns))
    logger.info("sMSE : %.3f" % sMSE_n)
    return sMSE_n

#######################################################################################################
    
if __name__ == "__main__":
#		"loss" : 2.3583231655481782,
#		"status" : "ok",
    cfg = {
            "model" : {
                    "dar_dropout_rate" : 0.08317089468998375,
                    "num_cells" : 400,
                    "num_layers" : 5,
                    "type" : "DeepAREstimator"
                    },
            "trainer" : {
                    "batch_size" : 200,
                    "learning_rate" : 0.0038250901642141026,
                    "learning_rate_decay_factor" : 0.7576763674695886,
                    "max_epochs" : 1000,
                    "minimum_learning_rate" : 0.0000068293835830944845,
                    "num_batches_per_epoch" : 25,
                    "patience" : 40,
                    "weight_decay" : 4.466652879348235e-8
            }
    }
#		"results" : [
#			2.3359812756073106,
#			2.0285312519930505,
#			2.3751604163891216,
#			2.4734001675060435,
#			2.5785427162453667
#		],


    results = []
    for idx in range(1, 6):
        logger.info("Sample # %s" % idx)
#        sMSE_idx = forecast(sample_data[idx-1].iloc[500:700,], cfg)
        sMSE_idx = forecast(sample_data[idx-1], cfg)
        results.append(sMSE_idx)
                   
    logger.info("sMSEs per sample: %s" % [round(result, 3) for result in results])
    sMSE_final = np.mean(results)
    logger.info("Final sMSE: %.3f" % sMSE_final)
