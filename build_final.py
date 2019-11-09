#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:09:36 2019

@author: mulderg
"""

from logging import basicConfig, getLogger
from logging import DEBUG as log_level
#from logging import INFO as log_level
basicConfig(level = log_level,
            format  = '%(asctime)s %(levelname)-8s %(module)-20s: %(message)s',
            datefmt ='%Y-%m-%d %H:%M:%S')
logger = getLogger(__name__)

import numpy as np
import pandas as pd
#import sys
from statistics import mean

import mxnet as mx
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
#from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer

num_eval_samples = 1
freq="1D"
prediction_length = 12 # Days
sample_data = [pd.read_csv("sample%d.csv" % idx, index_col=0) for idx in range(1,6)]

########################################################################################################
    
def gluon_fcast(cfg):

        
    logger.info("Params: %s" % cfg)
    MSE_cv = []
    sMSE_cv = []
    sME_cv = []
    
    for idx in range(1, 6):
        sample_name = "sample%d.csv" % idx
        logger.info("Sample: %s" % sample_name)

        
    return {"MSE" : MSE_cv, "sMSE" : sMSE_cv, "sME" : sME_cv}

if __name__ == "__main__":
    cfg =  {
#			"batch_size" : 32,
#			"dropout_rate" : 0.13666771922095983,
#			"learning_rate" : 0.016025873489796245,
#			"learning_rate_decay_factor" : 0.46333617412748473,
#			"max_epochs" : 200,
#			"minimum_learning_rate" : 0.00008899219665860409,
#			"num_batches_per_epoch" : 50,
#			"num_cells" : 200,
#			"num_layers" : 2,
#			"patience" : 32,
#			"weight_decay" : 2.3454678148399117e-8
    
            "batch_size" : 16,
            "dropout_rate" : 0.10462617503637472,
            "learning_rate" : 0.014908288955675229,
            "learning_rate_decay_factor" : 0.4373211414963845,
            "max_epochs" : 2000,
            "minimum_learning_rate" : 0.000009077720917882001,
            "num_batches_per_epoch" : 25,
            "num_cells" : 200,
            "num_layers" : 2,
            "patience" : 16,
            "weight_decay" : 6.280441431382861e-9
    }

    errors_cv = gluon_fcast(cfg)
    logger.info("Cross validated MSE  : %d" % mean(errors_cv["MSE"]))
    logger.info("Cross validated sMSE : %10.3f" % mean(errors_cv["sMSE"]))
    logger.info("Cross validated sME  : %10.3f" % mean(errors_cv["sME"]))