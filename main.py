import argparse

from pipeline import _parse_args
from pipeline_config import set_config, CONFIGS
from pipeline_utils import set_intput_papams

from train.train_op import train
from recode.recode_op import recode


import torch
# make workflow
if __name__=="__main__":     
    args, input_args = _parse_args()  
    cfg = set_config(args)
        
    params = set_intput_papams()
    
    for key, item in params.items():
        if key == 'cfg_recode' and isinstance(item, dict):
            recode(params[key])
        elif key == 'cfg_train' and isinstance(item, dict):
            cfg = params[key]
            train(cfg)
  