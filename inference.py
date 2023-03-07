import os, os.path as osp
import argparse
import glob
import torch
import cv2
from tqdm import tqdm
import sys

from pipeline_base_config import Path_cfg
LOCAL_PACKAGE_PATH = osp.join('/opt/local-path-provisioner', Path_cfg.volume)
sys.path.append(LOCAL_PACKAGE_PATH)    
    
from hibernation_no1.configs.config import Config
from hibernation_no1.mmdet.inference import build_detector, inference_detector, parse_inferece_result
from hibernation_no1.mmdet.modules.dataparallel import build_dp
from hibernation_no1.mmdet.visualization import draw_to_img

from hibernation_no1.mmdet.data.dataset import build_dataset
from hibernation_no1.mmdet.data.dataloader import build_dataloader
from hibernation_no1.mmdet.eval import Evaluate

from hibernation_no1.mmdet.get_info_algorithm import Get_info

import numpy as np


def test(cfg):
    cfg.pop('flag')
    cfg = Config(cfg)
    model_path = cfg.model_path
    if osp.isfile(model_path):
        model_path = osp.join(os.getcwd(), cfg.model_path)
    else:
        raise OSError(f"The path is not exist!!     path : {model_path}")
 
    os.makedirs(osp.join(os.getcwd(), cfg.test_result), exist_ok = True) 
    
    model = build_detector(cfg, model_path, device = cfg.device)
    
    dp_cfg = dict(model = model, 
                  cfg = cfg,
                  device = cfg.device,
                  classes = model.CLASSES)
    model = build_dp(**dp_cfg)
    
    batch_size = cfg.data.batch_size
    all_imgs_path = glob.glob(osp.join(cfg.data.data_root, "*.jpg"))
    batch_imgs_list = [all_imgs_path[x:x + batch_size] for x in range(0, len(all_imgs_path), batch_size)]
    

    for batch_imgs in tqdm(batch_imgs_list):
        if not isinstance(batch_imgs, list) : batch_imgs = list(batch_imgs)
        
        with torch.no_grad():
            # len: batch_size
            batch_results = inference_detector(model, batch_imgs)   
            
        # set path of result images
        out_files = []
        for img_path in batch_imgs:
            file_name = osp.basename(img_path)
            out_file = osp.join(os.getcwd(), cfg.test_result, file_name)
            out_files.append(out_file)

        for img_path, out_file, results in zip(batch_imgs, out_files, results):
            img = cv2.imread(img_path)  
            
            bboxes, labels, masks = parse_inferece_result(results)    

            # draw bbox, seg, label and save drawn_img
            draw_cfg = dict(img = img,
                            bboxes = bboxes,
                            labels = labels,
                            masks = masks,
                            class_names = model.CLASSES,
                            score_thr = cfg.show_score_thr)
            img = draw_to_img(**draw_cfg)
            
            
            cv2.imwrite(out_file, img)

        
        if cfg.get_board_info:      
            get_info_instance = Get_info(results, model.CLASSES, score_thr = cfg.show_score_thr)
            number_board_list = get_info_instance.get_board_info()









    