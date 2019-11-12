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
from os import environ
import traceback
from math import log
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

def load_greek_holidays():
    holidays = pd.read_csv("greek_holidays.csv")
    holidays['date'] = pd.to_datetime(holidays['date'])
    holidays.set_index('date', inplace=True)
    
    dates = pd.date_range("1/1/2017", "31/12/2017", freq="1D")
    all_days = pd.DataFrame({'idx' : [str(idx) for idx in range(1, len(dates)+1)]}, index = dates).join(holidays)
    hols_1_hot = pd.get_dummies(all_days['type'])
    hols_1_hot.set_index(all_days['idx'], inplace=True)
    return(hols_1_hot.transpose())
    
def forecast(data, cfg):
    logger.info("Params: %s " % cfg)

    dow_1_hot = pd.get_dummies(pd.Series(np.arange(1, data.shape[1] + 6) % 6)).iloc[:data.shape[1],1:].transpose()
    dow_1_hot.columns = data.columns
#    hols_1_hot = load_greek_holidays()[data.columns.tolist()]
#    all_1_hot = dow_1_hot.append(hols_1_hot)

    train_data = data.iloc[ : , :-prediction_length]
    train_1hot = dow_1_hot.iloc[ : , :-prediction_length]
    
    train_data_list = train_data.values.tolist()
    if cfg['model']['type'] in ['SimpleFeedForwardEstimator', 'DeepFactorEstimator']:
        train_dict = [ts_to_dict(idx, train_data_list[idx]) for idx in range(len(train_data_list))]
    else:
        train_dict = [ts_to_dict_1hot(idx, train_data_list[idx], train_1hot) for idx in range(len(train_data_list))]
    gluon_train = ListDataset(train_dict, freq=freq)

    trainer=Trainer(
        mx.Context("gpu"),
        epochs=5,
    )
    
#    trainer=Trainer(
#        mx.Context("gpu"),
#        epochs=cfg['trainer']['max_epochs'],
#        num_batches_per_epoch=cfg['trainer']['num_batches_per_epoch'],
#        batch_size=cfg['trainer']['batch_size'],
#        patience=cfg['trainer']['patience'],
#        
#        learning_rate=cfg['trainer']['learning_rate'],
#        learning_rate_decay_factor=cfg['trainer']['learning_rate_decay_factor'],
#        minimum_learning_rate=cfg['trainer']['minimum_learning_rate'],
#        weight_decay=cfg['trainer']['weight_decay'],
#    )

    # lags with a period of 6 +/-1, and month end (maybe)
    lags_seq=[1,2,3,4,5,6,7,8,9,10,11,12,13, 17,18,19, 23,24,25, 26,27,28, 29,30,31, 35,36,37, 41,42,43, 47,48,49, 53,54,55, 59,60,61]
    
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
    test_1hot = dow_1_hot.iloc[ : , -prediction_length:]
    
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

def gluon_fcast(cfg):        
    results = []
    for idx in range(1, 6):
        logger.info("Sample # %s" % idx)
        try:
            # Drop last prediction_length of sample data for final evaluation
#            sMSE_idx = forecast(sample_data[idx-1].iloc[300:500, : -prediction_length], cfg) 
#            sMSE_idx = forecast(sample_data[idx-1].iloc[ : , : -prediction_length], cfg)
            sMSE_idx = forecast(sample_data[idx-1], cfg)
            results.append(sMSE_idx)
            if idx > 1 and np.mean(results) > max_sMSE:
                logger.warning("Aborting run due to high mean(sMSE) = %.3f > %.3f" % (np.mean(results), max_sMSE))
                return {'loss': None, 'status': STATUS_FAIL, 'cfg' : cfg, 'results': results, 'build_url' : environ.get("BUILD_URL")}
            
        except Exception as e:
            exc_str = '\n%s' % traceback.format_exc()
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
            'num_batches_per_epoch'      : hp.choice('num_batches_per_epoch', [5, 10, 20, 40, 80]),
            'batch_size'                 : hp.choice('batch_size', [100, 200, 400]),
            'patience'                   : hp.choice('patience', [20, 40, 80]),
            
            'learning_rate'              : hp.uniform('learning_rate', 1e-04, 1e-02),
            'learning_rate_decay_factor' : hp.uniform('learning_rate_decay_factor', 0.4, 0.9),
            'minimum_learning_rate'      : hp.loguniform('minimum_learning_rate', log(1e-06), log(0.5e-04)),
            'weight_decay'               : hp.uniform('weight_decay', 00.5e-08, 10.0e-08),
        },
        'model' : hp.choice('model', [
                    {
                        'type'                       : 'SimpleFeedForwardEstimator',
                        'num_hidden_dimensions'      : hp.choice('num_hidden_dimensions', [[25], [50], [100],
                                                                                           [25, 25], [50, 25], [50, 50], [100, 50], [100, 100],
                                                                                           [100, 50, 50]])
                    },
                    {
                        'type'                       : 'DeepFactorEstimator',
                        'num_hidden_global'          : hp.choice('num_hidden_global', [25, 50, 100, 200]),
                        'num_layers_global'          : hp.choice('num_layers_global', [1, 2, 3]),
                        'num_factors'                : hp.choice('num_factors', [5, 10, 20]),
                        'num_hidden_local'           : hp.choice('num_hidden_local', [2, 5, 10]),
                        'num_layers_local'           : hp.choice('num_layers_local', [1, 2, 3]),
                    },
                    {
                        'type'                       : 'DeepAREstimator',
                        'num_cells'                  : hp.choice('num_cells', [200, 400, 600]),
                        'num_layers'                 : hp.choice('num_layers', [1, 3, 5, 7]),
                        
                        'dar_dropout_rate'           : hp.uniform('dar_dropout_rate', dropout_rate[0], dropout_rate[1]),
                    },
                    {
                        'type'                       : 'TransformerEstimator',
                        'model_dim'                  : hp.choice('model_dim', [8, 16, 32, 64]),
                        'inner_ff_dim_scale'         : hp.choice('inner_ff_dim_scale', [3, 4, 5]),
                        'pre_seq'                    : hp.choice('pre_seq', ['dn']),
                        'post_seq'                   : hp.choice('post_seq', ['drn']),
                        'act_type'                   : hp.choice('act_type', ['softrelu']),
                        'num_heads'                  : hp.choice('num_heads', [4, 8, 16]),
                       
                        'trans_dropout_rate'         : hp.uniform('trans_dropout_rate', dropout_rate[0], dropout_rate[1]),
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
        best = fmin(gluon_fcast, space, rstate=np.random.RandomState(rand_seed), algo=tpe.suggest, show_progressbar=False, trials=trials, max_evals=500)
    else:
        best = fmin(gluon_fcast, space, algo=tpe.suggest, show_progressbar=False, max_evals=20)
         
    return space_eval(space, best) 
    
if __name__ == "__main__":
    params = call_hyperopt()
    logger.info("Best params: %s" % params)