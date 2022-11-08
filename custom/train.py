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
from eval import inference_detector, parse_inferece_result, comfute_iou
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
            
            for i, val_data_batch in enumerate(val_dataloader):
                gt_bboxes_list = val_data_batch['gt_bboxes'].data
                gt_labels_list = val_data_batch['gt_labels'].data
                img_list = val_data_batch['img'].data
                gt_masks_list = val_data_batch['gt_masks'].data
                assert len(gt_bboxes_list) == 1 and (len(gt_bboxes_list) ==
                                                        len(gt_labels_list) ==
                                                        len(img_list) == 
                                                        len(gt_masks_list))
                # len: batch_size
                batch_gt_bboxes = gt_bboxes_list[0]           
                batch_gt_labels = gt_labels_list[0]  
                batch_gt_masks = gt_masks_list[0]    
                
                img_metas = val_data_batch['img_metas'].data[0]
                batch_images_path = []    
                for img_meta in img_metas:
                    batch_images_path.append(img_meta['filename'])
                    
                model_for_val = build_detector(cfg, cfg.model_path, device = cfg.device, logger = logger)
                
                batch_results = inference_detector(model_for_val, batch_images_path, cfg.data.val.batch_size)
                
                assert (len(batch_gt_bboxes) == 
                            len(batch_gt_labels) ==
                            len(batch_images_path) ==
                            len(batch_gt_masks) ==
                            len(batch_results))
                            
                   
                for gt_mask, gt_bboxes, gt_labels, result, img_path in zip(
                    batch_gt_masks, batch_gt_bboxes, batch_gt_labels, batch_results, batch_images_path
                    ):
                    i_bboxes, i_labels, i_mask = parse_inferece_result(result)
                    img = cv2.imread(img_path)
                    if cfg.show_score_thr > 0:
                        assert i_bboxes is not None and i_bboxes.shape[1] == 5
                        scores = i_bboxes[:, -1]
                        inds = scores > cfg.show_score_thr
                        i_bboxes = i_bboxes[inds, :]
                        i_labels = i_labels[inds]
                        if i_mask is not None:
                            i_mask = i_mask[inds, ...]
                    
                    i_cores = i_bboxes[:, -1]      # [num_instance]
                    i_bboxes = i_bboxes[:, :4]      # [num_instance, [x_min, y_min, x_max, y_max]]

                    
                    i_polygons = mask_to_polygon(i_mask)
                    gt_polygons = mask_to_polygon(gt_mask.masks)
                    
                    infer_dict = dict(bboxes = i_bboxes,
                                      polygons = i_polygons,
                                      labels = i_labels,
                                      score = i_cores)
                    gt_dict = dict(bboxes = gt_bboxes,
                                   polygons = gt_polygons,
                                   labels = gt_labels)
                    
                    for i_label in i_labels:
                        print(f"i_label ; {i_label}")
                    for gt_label in gt_labels:
                        print(f"gt_label ; {gt_label}")
                    exit()
                        
                    # for i in range(len(infer_dict['bboxes'])):
                    #     i_bboxes = infer_dict['bboxes'][i]
                    #     i_xmin, i_ymin, i_xmax, i_ymax  = i_bboxes
                    #     i_lt, i_rb = (int(i_xmin), int(i_ymin)), (int(i_xmax), int(i_ymax))
                    #     cv2.rectangle(img, i_lt, i_rb, color = (255, 255, 0), thickness = 2)
                    # cv2.imshow("img", img)
                    # while True:
                    #     if cv2.waitKey() == 27: break   
                        
                    # for j in range(len(gt_dict['bboxes'])):
                    #     gt_bboxes = gt_dict['bboxes'][j]
                    #     gt_xmin, gt_ymin, gt_xmax, gt_ymax  = gt_bboxes
                    #     gt_lt, gt_rb = (int(gt_xmin), int(gt_ymin)), (int(gt_xmax), int(gt_ymax))
                    #     cv2.rectangle(img, gt_lt, gt_rb, color = (0, 255, 255), thickness = 2)
                        
                    instance_dict = {}
                    for i in range(len(infer_dict['bboxes'])):
                        for j in range(len(gt_dict['bboxes'])):
                            i_bboxes, gt_bboxes = infer_dict['bboxes'][i], gt_dict['bboxes'][j]
                            iou = comfute_iou(i_bboxes, gt_bboxes)
                            if (iou > cfg.iou_threshold and 
                                infer_dict['score'][i] > cfg.confidence_threshold and
                                infer_dict['score'][i] == infer_dict['score'][i]):
                                
                                break
                            i_xmin, i_ymin, i_xmax, i_ymax  = i_bboxes
                            gt_xmin, gt_ymin, gt_xmax, gt_ymax  = gt_bboxes
                            i_lt, i_rb = (int(i_xmin), int(i_ymin)), (int(i_xmax), int(i_ymax))
                            gt_lt, gt_rb = (int(gt_xmin), int(gt_ymin)), (int(gt_xmax), int(gt_ymax))
                            
                            cv2.rectangle(img, gt_lt, gt_rb, color = (0, 255, 255), thickness = 2)
                            cv2.rectangle(img, i_lt, i_rb, color = (255, 255, 0), thickness = 2)
                            
                         
                    cv2.imshow("img", img)
                    while True:
                        if cv2.waitKey() == 27: break   
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
    
    