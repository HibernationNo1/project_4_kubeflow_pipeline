import argparse

from pipeline import _parse_args
from pipeline_config import set_config, CONFIGS
from pipeline_utils import set_intput_papams






import torch
# make workflow
if __name__=="__main__":     
    args, input_args = _parse_args()  
    cfg = set_config(args)
    
    params = set_intput_papams(pipeline = False)        
    
    for key, item in params.items():
        if key == 'cfg_recode' and isinstance(item, dict):   
            from recode.recode_op import recode   
            recode(params[key])
        elif key == 'cfg_train' and isinstance(item, dict):
            from train.train_op import train
            train(params[key])
        
        elif key == 'cfg_test' and isinstance(item, dict):
            from inference import test
            test(params[key])

        elif key == 'cfg_validation' and isinstance(item, dict):
            from inference import validation
            validation(params[key])