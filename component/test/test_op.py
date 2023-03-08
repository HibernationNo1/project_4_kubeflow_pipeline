from kfp.components import create_component_from_func, OutputPath

from pipeline_base_config import Base_Image_cfg
base_image = Base_Image_cfg()


def test(cfg):
    import os, os.path as osp
    import glob
    import torch
    import cv2
    import sys
    from git.repo import Repo

    WORKSPACE = dict(volume = cfg['path']['volume'],       # pvc volume path
                     work =  cfg['path']['work_space'],     # path if workspace in docker container
                     local_package = cfg['path']['local_volume'])    

    if __name__=="component.test.test_op": 
        assert osp.isdir(WORKSPACE['local_package']), f"The path '{WORKSPACE['local_package']}' is not exist!"
        sys.path.append(f"{WORKSPACE['local_package']}")    
              
    if __name__=="__main__":    
        assert osp.isdir(WORKSPACE['work']), f"The path '{WORKSPACE['work']}' is not exist!"
        assert osp.isdir(WORKSPACE['volume']), f"The path '{WORKSPACE['volume']}' is not exist!"
        # for import hibernation_no1
        package_path = osp.join(WORKSPACE['volume'], cfg['git_repo']['package'])
        if not osp.isdir(package_path):
            print(f" git clone 'hibernation_no1' to {package_path}")
            Repo.clone_from(f"git@github.com:HibernationNo1/{cfg['git_repo']['package']}.git", package_path)
        
        sys.path.append(f"{WORKSPACE['volume']}")  

    from hibernation_no1.configs.utils import change_to_tuple
    from hibernation_no1.configs.config import Config
    from hibernation_no1.mmdet.inference import build_detector, inference_detector, parse_inference_result
    from hibernation_no1.mmdet.modules.dataparallel import build_dp
    from hibernation_no1.mmdet.visualization import draw_to_img

    from hibernation_no1.mmdet.get_info_algorithm import Get_info

    import numpy as np



    def main(cfg, in_pipeline = False):
    
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
        

        for batch_imgs in batch_imgs_list:
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

            for img_path, out_file, results in zip(batch_imgs, out_files, batch_results):
                img = cv2.imread(img_path)  
                
                bboxes, labels, masks = parse_inference_result(results)    

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
                get_info_instance = Get_info(bboxes, labels, model.CLASSES, score_thr = cfg.get('show_score_thr', 0.5))
                number_board_list = get_info_instance.get_board_info()
                print(f"number_board_list : {number_board_list}")



    def dict2Config(cfg):
        cfg_flag = cfg.get('flag', None)
        if cfg_flag is not None:
            cfg = change_to_tuple(cfg, cfg_flag)
        cfg = Config(cfg)
        return cfg


    if __name__=="component.test.test_op":  
        cfg = dict2Config(cfg)
        main(cfg)

    if __name__=="__main__":  
        cfg = dict2Config(cfg)
        main(cfg, in_pipeline = True)

test_op = create_component_from_func(func = test,
                                     base_image = base_image.test,
                                     output_component_file= base_image.test_cp)