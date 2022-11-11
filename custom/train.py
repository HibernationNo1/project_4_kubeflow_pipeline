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
from builder import (build_model, 
                     build_dataset, 
                     build_dataloader, 
                     build_dp, 
                     build_optimizer, 
                     build_runner,
                     build_detector)
from eval import get_precision_recall_value
from inference import inference_detector
from visualization import show_result, mask_to_polygon

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
    
    args = parser.parse_args()
    
    return args

def set_config(args):
    cfg_path = args.cfg
    config_file_path = osp.join(os.getcwd(), cfg_path)
    cfg = Config.fromfile(config_file_path)
    
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
            val_data_cfg = cfg.data.val.copy()
            _ = val_data_cfg.pop("batch_size", None)
            train_dataset, val_dataset = build_dataset(train_cfg = cfg.data.train, val_cfg = val_data_cfg)
            
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
            
         
        
        elif mode == "val":     
            val_data_cfg = cfg.data.val.copy()
            _ = val_data_cfg.pop("batch_size", None)
            _, val_dataset = build_dataset(val_cfg = val_data_cfg)
            
            val_loader_cfg = dict(val_dataset = val_dataset,
                                    val_batch_size = cfg.data.val.get("batch_size", None),
                                    num_workers=cfg.data.workers_per_gpu,
                                    seed = cfg.seed,
                                    shuffle = True)
            _, val_dataloader = build_dataloader(**val_loader_cfg)
            
            model_for_val = build_detector(cfg, cfg.model_path, device = cfg.device, logger = logger)    
            
            precision_recall_dict = get_precision_recall_value(model_for_val, cfg, val_dataloader, mask_to_polygon)
            
            
            # def compute_PR_area()         
                                
            
        
            import matplotlib.pyplot as plt
            
            for class_name, threshold_list in precision_recall_dict.items():
                print(f"\n---class_name : {class_name}")
                 
                sorted_pr_value = []
                
                prcs_rcl_list = []
                for idx, threshold in enumerate(threshold_list):
                    prcs_rcl_list.append([threshold['precision'], threshold['recall']])
                prcs_rcl_list.reverse()
                
                # for idx, prcs_rcl in enumerate(prcs_rcl_list):
                #     precision, recall = prcs_rcl
                #     print(f"recall, precision ; {recall, precision}")
                ap_area = 0
                before_recall =0
                continue_count = -1
                for idx, prcs_rcl in enumerate(prcs_rcl_list):
                    if continue_count > 0: 
                        continue_count -=1 
                        if continue_count == 0 : continue_count = -1
                        continue
                    precision, recall = prcs_rcl
                    
                    if idx+1 < len(prcs_rcl_list) and prcs_rcl_list[idx+1][1] == recall:
                        tmp_precision_list = []
                        for i in range(idx, len(prcs_rcl_list)):
                            if recall == prcs_rcl_list[i][1]:       # same recall
                                continue_count +=1
                                tmp_precision_list.append(prcs_rcl_list[i][0])
                            else: break
                        precision = max([precis for precis in tmp_precision_list])
                    
                    print(f"before_recall : {before_recall:.2f}, recall : {recall:.2f}      ::: {abs(before_recall - recall):.2f}, {precision:.2f}     area = {abs(before_recall - recall)*precision:.2f}")
                    area = abs(before_recall - recall)*precision
                    ap_area += area
                    print(f"ap_area : {ap_area}")
                    
                    before_recall = recall
                
                if ap_area > 1: exit()
                print(f"sum_ap_area : {ap_area}")  
                # TODO : ap_area가 1보다 큰 경우는 우쨰?
            exit()
                    
                # before_recall = None
                # curve_flag = False
                # max_recall = -1
                # prec_recall_list = []
                # tmp_precision_list, tmp_recall_list = [], []
                # for idx, threshold in enumerate(threshold_list):
                #     recall, precision = threshold['recall'], threshold['precision']
                #     prec_recall_list.append([precision, recall])
                #     tmp_precision_list.append(precision)
                #     tmp_recall_list.append(recall)
                        
                    
                #     if before_recall is None:
                #         before_recall = recall
                #         continue
                    
                #     if before_recall - recall < 0:
                #         curve_flag = True
                        
                #     if max_recall < recall : max_recall = recall
                    
                    
                #     before_recall = recall
                
            
                        
                # ap_area = 0
                # if curve_flag:
                #     assert max_recall != -1
                    
                # else:
                    # prec_recall_list.sort(key= lambda x:x[1])
                    
                    # before_recall =0
                    # continue_count = -1
                    # for idx, prec_recall in enumerate(prec_recall_list):
                    #     if continue_count > 0: 
                    #         continue_count -=1 
                    #         continue
                    #     precision, recall = prec_recall
                        
                    #     if idx+1 < len(prec_recall_list) and prec_recall_list[idx+1][1] == recall:
                    #         tmp_precision_list = []
                    #         for i in range(idx, len(prec_recall_list)):
                    #             if recall == prec_recall_list[i][1]:
                    #                 continue_count +=1
                    #                 tmp_precision_list.append(prec_recall_list[i][0])
                    #             else: break
                    #         precision = max([precis for precis in tmp_precision_list])
                            
                    #     area = abs(before_recall - recall)*precision
                    #     ap_area += area
                        
                    #     before_recall = recall
                    
                    
                        
                        
          
                        
                        
                    
                    # if len(stac_pre_rcal) > 0:
                        
                        
                    
                    # print(f"\nthreshold: {threshold['iou_threshold']}")
                    # print(f"num_gt : {threshold['num_gt']},       num_pred : {threshold['num_pred']},       num_true : {threshold['num_true']}")
                    # print(f"recall: {threshold['recall']:.3f},    precision: {threshold['precision']:.3f}")
                    # print(f"F1_score: {threshold['F1_score']:.3f}")
                   
                    
                    
                
                    
            exit()
                # sorted_pr_value.sort(key = lambda x: x[1])
                # precision_list, recall_list = [], []
                # for precision, recall in sorted_pr_value:
                #     precision_list.append(precision)
                #     recall_list.append(recall)
                
                
                
                # pr_area = 0
                # before_precision, before_recall = None, None
                # for precision, recall in sorted_pr_value:
                #     print(f"precision, recall : {precision, recall}")
                #     if before_precision is None:
                #         before_precision, before_recall = precision, recall
                #         continue
                    
                #     if recall == before_recall: continue
                    
                #     difference_recall = before_recall-recall
                #     pr_area += (difference_recall*before_precision)
                #     before_precision, before_recall = precision, recall                    

                # plt.plot(recall_list, precision_list)
                # # plt.scatter(recall_list, precision_list, color='darkmagenta')
                # plt.show()
                
                
            exit()
            
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
    
    