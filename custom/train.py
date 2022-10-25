import argparse
import os, os.path as osp
import numpy as np

from utils.utils import get_device, set_meta
from utils.config import Config
from utils.log import set_logger_info, create_logger, collect_env, log_info
from builder import (build_model, 
                     build_dataset, 
                     build_dataloader, 
                     build_dp, 
                     build_optimizer, 
                     build_runner)
import __init__ # 모든 module 및 function import 

# TODO : import torch.distributed as dist 사용해보기

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
    
    
    logger_timestamp = set_logger_info(osp.join(os.getcwd(), cfg.result), cfg.log_level)
    logger = create_logger('enviroments')
    
    # log env info      
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)           
    # logger.info(f'Config:\n{cfg.pretty_text}')
    
    
    
    # TODO: validation dataset을 build하고 validation dataloader도 build하기
    # val_dataset = build_dataset(cfg.data.val)
    data_loaders = []
    train_dataset = build_dataset(cfg.data.train)
    if cfg.model.type == 'MaskRCNN':
        cfg.model.roi_head.bbox_head.num_classes = len(train_dataset.CLASSES) 
        cfg.model.roi_head.mask_head.num_classes = len(train_dataset.CLASSES)    
        # cfg.model.rpn_head.num_classes = len(train_dataset.CLASSES)   # TODO: 이거 추가 안되어있는데, 추가하고 학습에 차이있는지 확인
    
    train_loader_cfg = dict(
        batch_size=cfg.data.samples_per_gpu,
        num_workers=cfg.data.workers_per_gpu,
        seed = cfg.seed,
        shuffle = True)
    data_loaders.append(build_dataloader(train_dataset, **train_loader_cfg))   
    
    

    assert cfg.get('train_cfg') is None , 'train_cfg must be specified in both outer field and model field'
    model = build_model(cfg.model)
    
    if cfg.pretrained is not None:
        model.init_weights()
    

    
        
    
    # train_detector ---
    logger = create_logger('train')


    model = build_dp(model, cfg.device)

    
    # build optimizer
    optimizer = build_optimizer(model, cfg, logger)     # TODO 어떤 layer에 optimizer를 적용시켰는지 확인
    
    runner_meta = set_meta(cfg, args, env_info)
    
    # flow를 통해 train, evluate를 명령어 하나헤 한 번에 실행할 수 있게. (validate은 training도중 실행) 
    assert len(cfg.workflow) == len(data_loaders)
    if cfg.get('epoch_or_iter', None) is None: cfg.epoch_or_iter = "epoch"
    for flow, data_loader in zip(cfg.workflow, data_loaders):   # TODO : epoch단위로할지 iter단위로할지 구성    해당config확인
        mode, epoch = flow
        if mode == "train":
            runner_build_cfg = dict(
                    model=model,
                    optimizer=optimizer,
                    work_dir=log_info['result_dir'],
                    logger=logger,
                    meta=runner_meta,
                    batch_size = cfg.data.samples_per_gpu)
            
            if cfg.epoch_or_iter == "iter" : 
                runner_build_cfg['max_iters'] = epoch
            else:
                runner_build_cfg['max_epochs'] = epoch
   
            
            train_runner = build_runner(runner_build_cfg)

            train_runner.timestamp = logger_timestamp

            if cfg.checkpoint_config is not None: 
                # set model name to be saved(.pth format)
                filename_tmpl = cfg.checkpoint_config.filename_tmpl
                assert len(filename_tmpl.split('.')) < 3, "wrong model name. \
                                                            \ncheck : cfg.checkpoint_config.filename_tmpl \
                                                            \ncurrent: {cfg.checkpoint_config.filename_tmpl}"
                if len(filename_tmpl.split('.')) == 1: 
                    filename_tmpl += "_{}.pth"
                elif len(filename_tmpl.split('.')) == 2:                    
                    if filename_tmpl.split('.')[-1] != "pth": 
                        filename_tmpl = filename_tmpl.split('.')[0] + ".pth"
                        
                cfg.checkpoint_config.filename_tmpl = filename_tmpl
                
                # set CLASSES
                cfg.checkpoint_config.meta = dict(
                    CLASSES=train_dataset.CLASSES)
                
            # register hooks
            train_runner.register_training_hooks(
                cfg.lr_config,
                cfg.optimizer_config,
                cfg.checkpoint_config,
                cfg.log_config,
                custom_hooks_config=cfg.get('custom_hooks', None))

            resume_from = cfg.get('resume_from', None)
            load_from = cfg.get('load_from', None)
            
            # TODO
            if resume_from is not None:
                train_runner.resume(cfg.resume_from)
            elif cfg.load_from:
                train_runner.load_checkpoint(cfg.load_from)
            
            train_runner.run(data_loader, flow)
        
        
        
        
        elif mode == "train":
            pass
                # TODO : validate 수행
                # if validate:      
                #     val_dataloader_default_args = dict(
                #         samples_per_gpu=1,
                #         workers_per_gpu=cfg.data.train_dataloader.workers_per_gpu,
                #         dist=distributed,
                #         shuffle=False,
                #         persistent_workers=False)

                #     val_dataloader_args = {
                #         **val_dataloader_default_args,
                #         **cfg.data.get('val_dataloader', {})
                #     }
                #     # Support batch_size > 1 in validation

                #     if val_dataloader_args['samples_per_gpu'] > 1:
                #         # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                #         cfg.data.val.pipeline = replace_ImageToTensor(
                #             cfg.data.val.pipeline)
                    
                #     val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))

                #     val_dataloader = build_dataloader(val_dataset, **val_dataloader_args)
                #     eval_cfg = cfg.get('evaluation', {})
                #     eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
                #     eval_hook = DistEvalHook if distributed else EvalHook
                
                #     # In this PR (https://github.com/open-mmlab/mmcv/pull/1193), the
                #     # priority of IterTimerHook has been modified from 'NORMAL' to 'LOW'.
                #     runner.register_hook(
                #         eval_hook(val_dataloader, **eval_cfg), priority='LOW')
    
    