import os, os.path as osp
import argparse
import glob
import torch
import cv2
from tqdm import tqdm


# WORKSPACE = dict(pack = '/pvc',         # pvc volume path
#                      work = '/workspace',   # path if workspace in docker container
#                      local = '/opt/local-path-provisioner'\
#                              '/pvc-c16a15b9-962b-4636-a0b4-2fd18a1423ae_project-pipeline_hibernation-project-qffnp-pipeline-pvc-2'
#                     )    
# import sys
# sys.path.append(f"{WORKSPACE['local']}")    
    
from hibernation_no1.configs.config import Config
from hibernation_no1.mmdet.inference import build_detector, inference_detector, parse_inferece_result
from hibernation_no1.mmdet.modules.dataparallel import build_dp
from hibernation_no1.mmdet.visualization import draw_to_img


def inference(cfg):
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
    
    batch_size = cfg.data.test.batch_size
    all_imgs_path = glob.glob(osp.join(cfg.data.test.data_root, "*.jpg"))
    batch_imgs_list = [all_imgs_path[x:x + batch_size] for x in range(0, len(all_imgs_path), batch_size)]
    
    for batch_imgs in tqdm(batch_imgs_list):
        if not isinstance(batch_imgs, list) : batch_imgs = list(batch_imgs)
        
        with torch.no_grad():
            # len: batch_size
            results = inference_detector(model, batch_imgs, batch_size)   
            
        # set path of result images
        out_files = []
        for img_path in batch_imgs:
            file_name = osp.basename(img_path)
            out_file = osp.join(os.getcwd(), cfg.test_result, file_name)
            out_files.append(out_file)
            
        for img_path, out_file, result in zip(batch_imgs, out_files, results):
            img = cv2.imread(img_path)  
            
            bboxes, labels, masks = parse_inferece_result(result)    

            # draw bbox, seg, label and save drawn_img
            draw_cfg = dict(img = img,
                            bboxes = bboxes,
                            labels = labels,
                            masks = masks,
                            class_names = model.CLASSES,
                            score_thr = cfg.show_score_thr)
            img = draw_to_img(**draw_cfg)
            
            
            cv2.imwrite(out_file, img)
            