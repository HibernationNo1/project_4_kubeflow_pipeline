import argparse
import os, os.path as osp
import numpy as np
import torch
import glob
import cv2
from tqdm import tqdm

from utils.utils import get_device, set_meta
from utils.config import Config
from utils.log import set_logger_info, create_logger, collect_env, log_info
from builder import (build_model, 
                     build_dataset, 
                     build_dataloader, 
                     build_dp, 
                     build_optimizer, 
                     build_runner,
                     build_detector)
from eval import inference_detector
from visualization import show_result

import __init__ # to import all module and function 


# python train.py --cfg configs/swin_maskrcnn.py --epo 50 --val_iter 50
# python train.py --cfg configs/swin_maskrcnn.py --model_path model/model_3.pth  --test

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required = True, help="name of config file")
    parser.add_argument('--epo', type= int, help= "max epoch in train mode")
    parser.add_argument('--val_iter', type= int, help= "iters that run validation. if -1: run val after every epoch")
    
    parser.add_argument('--model_path', type = str, help= "path of model(.pth format)") 
    parser.add_argument('--test', action='store_true', default=False, help= "if True: run only test mode") 
    
    args = parser.parse_args()
    
    return args

def set_config(args):
    cfg_path = args.cfg
    config_file_path = osp.join(os.getcwd(), cfg_path)
    cfg = Config.fromfile(config_file_path)
    
    cfg.seed = np.random.randint(2**31)
    cfg.device = get_device()    
    
    if cfg.workflow[0][0] == 'test' or args.test:
        cfg.workflow = [("test", None)]
        assert cfg.model_path is not None and args.model_path is not None, f"model path is not set!"
        if args.model_path is not None: cfg.model_path = args.model_path
        assert osp.isfile(cfg.model_path), f"model path: '{cfg.model_path}' is not exist!"
        cfg.model_path = osp.join(os.getcwd(), cfg.model_path)
        assert isinstance(cfg.data.test.batch_size, int)
        assert osp.isdir(cfg.data.test.data_root)
        
                
    
    if args.epo is not None:
        new_flow = []
        for flow in cfg.workflow:
            mode, epoch = flow
            if mode == 'train':
                epoch = args.epo
            new_flow.append((mode, epoch))
        cfg.workflow = new_flow
        
    if args.val_iter is not None: cfg.log_config.interval = args.val_iter
    
 
    return cfg


    
    
if __name__ == "__main__":
    print(f'pytorch version: {torch.__version__}')
    assert torch.cuda.is_available(), f'torch.cuda.is_available() is {torch.cuda.is_available()}!'
 
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
    
    # flow를 통해 train, evluate를 명령어 하나헤 한 번에 실행할 수 있게. (validate은 training도중 실행) 
    for flow in cfg.workflow:   
        mode, epoch = flow
        logger = create_logger(f'{mode}')

        if mode == "train":
            # build dataloader
            train_dataset, val_dataset = build_dataset(cfg.data.train, cfg.data.val)
            
            if cfg.model.type == 'MaskRCNN':
                cfg.model.roi_head.bbox_head.num_classes = len(train_dataset.CLASSES) 
                cfg.model.roi_head.mask_head.num_classes = len(train_dataset.CLASSES)    
                # cfg.model.rpn_head.num_classes = len(train_dataset.CLASSES)   # TODO: 이거 추가 안되어있는데, 추가하고 학습에 차이있는지 확인
            
             

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
            model = build_dp(model, cfg.device)
            model.cfg = cfg
            
            # build optimizer
            optimizer = build_optimizer(model, cfg, logger)     # TODO 어떤 layer에 optimizer를 적용시켰는지 확인
            
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
            cfg.log_config.iter_per_epoch = len(train_dataloader)
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
                           val_batch_size =  cfg.data.val.batch_size,
                           val_score_thr = cfg.show_score_thr)
            
            train_runner.run(**run_cfg)
        
        elif mode == "test":            
            batch_size = cfg.data.test.batch_size
            all_imgs_path = glob.glob(os.path.join(cfg.data.test.data_root, "*.jpg"))
            batch_imgs_list = [all_imgs_path[x:x + batch_size] for x in range(0, len(all_imgs_path), batch_size)]
            
            
            model = build_detector(cfg, cfg.model_path, device = cfg.device, logger = logger)

            classes = model.CLASSES
            model_config = model.cfg  
            model = build_dp(model, cfg.device)
            
            model.cfg = model_config  
            outputs = []  
            
            for batch_imgs in tqdm(batch_imgs_list):
                with torch.no_grad():
                    # len: batch_size
                    results = inference_detector(model, batch_imgs, batch_size)   # TODO: 여기서 시간 너무많이 잡아먹힘
            
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
            
         
        
        elif mode == "val":     # is different from the validation as run during training.
                                # need validation dataloader 
            # TODO: validation dataset을 build하고 validation dataloader도 build하기
            # val_dataset = build_dataset(cfg.data.val)
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
    
    