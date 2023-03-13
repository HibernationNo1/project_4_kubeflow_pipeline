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
    
    
    for param in params:
        _, name, cfg = param
        if name == 'record':   
            from component.record.record_op import record   
            record(cfg, None, None)
        
        elif name == 'train':
            from component.train.train_op import train
            train(cfg, None, None)
        
        elif name == 'test':
            from component.test.test_op import test
            test(cfg, None, None)

        elif name == 'evaluate':
            from component.evaluate.evaluate_op import evaluate
            evaluate(cfg, None, None)