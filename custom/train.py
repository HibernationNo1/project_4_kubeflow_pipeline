import argparse
import os, os.path as osp

import time
from config import Config
from log import get_logger, collect_env
from builder import build_model, build_dataset
import __init__ # 모든 module 및 function import 

# python train.py --cfg configs/swin_maskrcnn.py

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required = True, help="name of config file")
    parser.add_argument('--epo', type= int, help= "max epoch")
    
    args = parser.parse_args()
    
    return args

def set_config(cfg_path):
    
    config_file_path = osp.join(os.getcwd(), cfg_path)
    cfg = Config.fromfile(config_file_path)
    
    return cfg


    
        
    
if __name__ == "__main__":
 

    args = parse_args()
    cfg = set_config(args.cfg)
    
    result_dir = osp.join(os.getcwd(), cfg.result) 
    os.makedirs(result_dir, exist_ok= True)
    
    
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(result_dir, f'{timestamp}.log')
    log_name = 'project_pipeline'
    get_logger(name=log_name, log_file=log_file, log_level=cfg.log_level)        
    from log import log_recorder
    logger = log_recorder[log_name]
    
    # log env info      
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)           
    # logger.info(f'Config:\n{cfg.pretty_text}')
    
    
    
    datasets = build_dataset(cfg.data.train)
    num_classes = datasets.CLASSES

    assert cfg.get('train_cfg') is None , 'train_cfg must be specified in both outer field and model field'
    model = build_model(cfg.model)


