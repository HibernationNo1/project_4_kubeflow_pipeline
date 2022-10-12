import argparse
import os, os.path as osp
import numpy as np

from utils.utils import get_device, set_meta
from utils.config import Config
from utils.log import set_logger_info, create_logger, collect_env
from builder import (build_model, 
                     build_dataset, 
                     build_dataloader, 
                     build_dp, 
                     build_optimizer, 
                     build_runner)
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
    
    cfg.seed = np.random.randint(2**31)
    cfg.device = get_device()    
    return cfg


    
    
if __name__ == "__main__":
 

    args = parse_args()
    cfg = set_config(args.cfg)
    
    
    set_logger_info(osp.join(os.getcwd(), cfg.result), cfg.log_level)
    logger = create_logger('enviroments')
    
    # log env info      
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)           
    # logger.info(f'Config:\n{cfg.pretty_text}')
    
    meta = set_meta(cfg, args, env_info)
    
    
    dataset = build_dataset(cfg.data.train)
    cfg.model.roi_head.bbox_head.num_classes = len(dataset.CLASSES) # TODO: MaskRCNN이 아닌 경우도 상정하기
    cfg.model.roi_head.mask_head.num_classes = len(dataset.CLASSES)    
    

    assert cfg.get('train_cfg') is None , 'train_cfg must be specified in both outer field and model field'
    model = build_model(cfg.model)
    
    if cfg.pretrained is not None:
        model.init_weights()
    
    if cfg.checkpoint_config is not None:   pass # TODO 
    
    logger = create_logger('train')

    train_loader_cfg = dict(
        batch_size=cfg.data.samples_per_gpu,
        num_workers=cfg.data.workers_per_gpu,
        seed = cfg.seed,
        shuffle = True)

    data_loaders = build_dataloader(dataset, **train_loader_cfg)     # 이게 run안에서 어떻게 동작하는지 보자

    model = build_dp(model, cfg.device)
    
    # build optimizer
    optimizer = build_optimizer(model, cfg, logger)     # TODO 어떤 layer에 optimizer를 적용시켰는지 확인
    
    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta))

