import argparse
import os, os.path as osp
import numpy as np
import torch
import glob
import cv2
from tqdm import tqdm

from utils.utils import get_device, set_meta, confirm_model_path
from utils.config import Config
from utils.log import set_logger_info, create_logger, collect_env, log_info
from utils.visualization import show_result, mask_to_polygon
from builder import (build_model, 
                     build_dataset, 
                     build_dataloader, 
                     build_dp, 
                     build_optimizer, 
                     build_runner,
                     build_detector)
from eval.eval import Evaluate
from eval.inference import inference_detector


import __init__ # to import all module and function 


# python train.py --cfg configs/swin_maskrcnn.py --epo 50 --val_iter 50
# python train.py --cfg configs/swin_maskrcnn.py --model_path model/model_21.pth  --test
# python train.py --cfg configs/swin_maskrcnn.py --model_path model/model_21.pth  --val

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required = True, help="name of config file")
    parser.add_argument('--epo', type= int, help= "max epoch in train mode")
    parser.add_argument('--val_iter', type= int, help= "iters that run validation. if -1: run val after every epoch")
    
    parser.add_argument('--model_path', type = str, help= "path of model(.pth format)") 
    parser.add_argument('--test', action='store_true', default=False, help= "if True: run only test mode") 
    parser.add_argument('--val', action='store_true', default=False, help= "if True: run only val mode") 
    
    swin_parser = parser.add_argument_group('SwinTransformer')
    swin_parser.add_argument('--pm_dilation', type = int, help= "dilation of SwinTransformer.PatchMerging") 
    swin_parser.add_argument('--drop_rate', type = float, help= "drop_rate of SwinTransformer") 
    swin_parser.add_argument('--drop_path_rate', type = float, help= "drop_path_rate of SwinTransformer") 
    swin_parser.add_argument('--attn_drop_rate', type = float, help= "attn_drop_rate of SwinTransformer.SwinBlockSequence.ShiftWindowMSA.WindowMSA") 
     
  
    args = parser.parse_args()
    
    return args



def set_config(args):
    cfg_path = args.cfg
    config_file_path = osp.join(os.getcwd(), cfg_path)
    cfg = Config.fromfile(config_file_path)
    print(cfg)
    exit()
    cfg.seed = np.random.randint(2**31)
    cfg.device = get_device()    
    
    if args.test and args.val:
        cfg.workflow = [("test", None), ("val", None)]
    elif not args.test and not args.val:  pass
    else:
        if cfg.workflow[0][0] == 'test' or args.test:
            cfg.workflow = [("test", None)]
            confirm_model_path(cfg, args)
            assert isinstance(cfg.data.test.batch_size, int)
            assert osp.isdir(cfg.data.test.data_root)
        
        if cfg.workflow[0][0] == 'val' or args.val:
            cfg.workflow = [("val", None)]
            confirm_model_path(cfg, args)
            assert isinstance(cfg.data.val.batch_size, int)
            assert osp.isdir(cfg.data.val.data_root)           
    
    if args.epo is not None:
        new_flow = []
        for flow in cfg.workflow:
            mode, epoch = flow
            if mode == 'train':
                epoch = args.epo
            new_flow.append((mode, epoch))
        cfg.workflow = new_flow
        
    if args.val_iter is not None: 
        for i, custom_hook_config in enumerate(cfg.custom_hook_config):
            if custom_hook_config.get("type", None) == "Custom_Hook":
                cfg.custom_hook_config[i].val_iter = args.val_iter
    
    
    if args.pm_kernel_size is not None:
        cfg.model.backbone.pm_kernel_size = args.pm_kernel_size
    if args.pm_dilation is not None:
        cfg.model.backbone.pm_dilation = args.pm_dilation
    if args.window_size is not None:
        cfg.model.backbone.window_size = args.window_size
    if args.drop_rate is not None:
        cfg.model.backbone.drop_rate = args.drop_rate
    if args.attn_drop_rate is not None:
        cfg.model.backbone.attn_drop_rate = args.attn_drop_rate
    if args.drop_path_rate is not None:
        cfg.model.backbone.drop_path_rate = args.drop_path_rate
    
      
    return cfg


    
    
if __name__ == "__main__":
    print(f'pytorch version: {torch.__version__}')
    assert torch.cuda.is_available(), f'torch.cuda.is_available() is {torch.cuda.is_available()}!'
    print(f"torch.version.cuda: {torch.version.cuda}")
    
    args = parse_args()
    cfg = set_config(args)
    
    logger_timestamp = set_logger_info(cfg)
    logger = create_logger('enviroments')
    
    # log env info      
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)           
    # logger.info(f'Config:\n{cfg.pretty_text}')
    
    runner_meta = set_meta(cfg, args, env_info)    
    
    for flow in cfg.workflow:   
        mode, epoch = flow
        logger = create_logger(f'{mode}')

        if mode == "train":
            # build dataloader
            val_data_cfg = cfg.data.val.copy()
            _ = val_data_cfg.pop("batch_size", None)
            train_dataset, val_dataset = build_dataset(train_cfg = cfg.data.train, val_cfg = val_data_cfg)
            
            if cfg.model.type == 'MaskRCNN':
                cfg.model.roi_head.bbox_head.num_classes = len(train_dataset.CLASSES) 
                cfg.model.roi_head.mask_head.num_classes = len(train_dataset.CLASSES)    
                # cfg.model.rpn_head.num_classes = len(train_dataset.CLASSES)   # TODO: add this and confirm training
            
            train_loader_cfg = dict(train_dataset = train_dataset,
                                    val_dataset = val_dataset,
                                    train_batch_size=cfg.data.samples_per_gpu,
                                    val_batch_size = cfg.data.val.get("batch_size", None),
                                    num_workers=cfg.data.workers_per_gpu,
                                    seed = cfg.seed,
                                    shuffle = True)
            train_dataloader, val_dataloader = build_dataloader(**train_loader_cfg)
            # build model
            assert cfg.get('train_cfg') is None , 'train_cfg must be specified in both outer field and model field'
            model = build_model(cfg.model)
            
            if cfg.pretrained is not None: model.init_weights()
            dp_cfg = dict(model = model, device = cfg.device,
                          cfg = cfg,
                          classes = train_dataset.CLASSES)
            model = build_dp(**dp_cfg)
          
            # build optimizer
            
            optimizer = build_optimizer(model, cfg, logger)    
                           
            # from mmcv.runner import build_optimizer as _build_optimizer
            # optimizer_cfg = {'type': 'AdamW', 'lr': 0.0001, 'betas': (0.9, 0.999), 'weight_decay': 0.05, 'paramwise_cfg': {'custom_keys': {'absolute_pos_embed': {'decay_mult': 0.0}, 'relative_position_bias_table': {'decay_mult': 0.0}, 'norm': {'decay_mult': 0.0}}}}
            # optimizer = _build_optimizer(model, cfg.optimizer)
                
          
            
            # build runner
            runner_build_cfg = dict(
                    model=model,
                    optimizer=optimizer,
                    work_dir=log_info['result_dir'],
                    logger=logger,
                    meta=runner_meta,
                    batch_size = cfg.data.samples_per_gpu,
                    max_epochs = epoch)
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
                
                cfg.checkpoint_config.model_cfg = cfg.model

            # register hooks
            if cfg.get('custom_hook_config', None) is not None:
                for i, c_cfg in enumerate(cfg.custom_hook_config):
                    if c_cfg.get("type", None) =="Custom_Hook":
                        cfg.custom_hook_config[i].ev_iter = cfg.log_config.ev_iter= len(train_dataloader)
                        cfg.custom_hook_config[i].max_epochs = cfg.log_config.max_epochs= epoch
            
            train_runner.register_training_hooks(
                cfg.lr_config,
                cfg.optimizer_config,
                cfg.checkpoint_config,
                cfg.log_config,
                custom_hooks_config=cfg.get('custom_hook_config', None))

            resume_from = cfg.get('resume_from', None)
            load_from = cfg.get('load_from', None)
            
            # TODO
            if resume_from is not None:
                train_runner.resume(cfg.resume_from)
            elif cfg.load_from:
                train_runner.load_checkpoint(cfg.load_from)
            
            run_cfg = dict(train_dataloader = train_dataloader,
                           val_dataloader = val_dataloader,
                           flow = flow,
                           val_cfg = cfg,
                           mask_to_polygon = mask_to_polygon)
            train_runner.run(**run_cfg)
            
        
        elif mode == "test":            
            batch_size = cfg.data.test.batch_size
            all_imgs_path = glob.glob(os.path.join(cfg.data.test.data_root, "*.jpg"))
            batch_imgs_list = [all_imgs_path[x:x + batch_size] for x in range(0, len(all_imgs_path), batch_size)]
            
            model = build_detector(cfg, cfg.model_path, device = cfg.device, logger = logger)
            classes = model.CLASSES

            dp_cfg = dict(model = model, 
                          device = cfg.device,
                          cfg = model.cfg,
                          classes = model.CLASSES)
            model = build_dp(model, cfg.device, cfg = model.cfg)
            outputs = []  
            
            for batch_imgs in tqdm(batch_imgs_list):
                with torch.no_grad():
                    # len: batch_size
                    results = inference_detector(model, batch_imgs, batch_size)   
            
                # set path of result images
                out_files = []
                for img_path in batch_imgs:
                    file_name = os.path.basename(img_path)
                    out_file = os.path.join(log_info['result_dir'], file_name)
                    out_files.append(out_file)
                    
                for img_path, out_file, result in zip(batch_imgs, out_files, results):
                    img = cv2.imread(img_path)      

                    # draw bbox, seg, label and save drawn_img
                    show_result(img, result, classes,   
                                out_file=out_file,
                                score_thr=cfg.show_score_thr)
            
         
        
        elif mode == "val":     
            val_data_cfg = cfg.data.val.copy()
            _ = val_data_cfg.pop("batch_size", None)
            _, val_dataset = build_dataset(val_cfg = val_data_cfg)
            
            tmp_list = glob.glob(val_dataset.img_prefix +"*.jpg")
           
            val_loader_cfg = dict(val_dataset = val_dataset,
                                    val_batch_size = len(tmp_list), #cfg.data.val.get("batch_size", None),
                                    num_workers=cfg.data.workers_per_gpu,
                                    seed = cfg.seed,
                                    shuffle = True)
            _, val_dataloader = build_dataloader(**val_loader_cfg)
            
            model_for_val = build_detector(cfg, cfg.model_path, device = cfg.device, logger = logger)    
                
            eval_cfg = dict(model= model_for_val, 
                       cfg= cfg,
                       dataloader= val_dataloader,
                       mask_to_polygon= mask_to_polygon)    
            eval = Evaluate(**eval_cfg)
            mAP = eval.compute_mAP()
            # F1_score = eval.compute_F1_score()
            
    
    