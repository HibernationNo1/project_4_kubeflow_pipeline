from kfp.components import create_component_from_func, InputPath, OutputPath

from pipeline_base_config import Base_Image_cfg
base_image = Base_Image_cfg()


def test(cfg : dict, input_run_flag: InputPath("dict"),
         run_flag_path: OutputPath('dict')):       # Error when using variable name 'run_flag_path'

    import json
    import os, os.path as osp
    import glob
    import torch
    import cv2
    import sys
    from git.repo import Repo

    WORKSPACE = dict(component_volume = cfg['path']['component_volume'],       # pvc volume path on component container
                     local_volume = cfg['path'].get('local_volume', None),     # pvc volume path on local
                     docker_volume = cfg['path']['docker_volume'],     # volume path on katib container
                     work = cfg['path']['work_space']
                     )    

    # set package path to 'import {custom_package}'
    if __name__=="component.test.test_op":
        docker_volume = f"/{WORKSPACE['docker_volume']}"
        
        if WORKSPACE['local_volume'] is not None:
            local_module_path = osp.join('/opt/local-path-provisioner', WORKSPACE['local_volume']) 
        else:
            local_module_path = osp.join(os.getcwd(), cfg['git']['package_repo'])       
        
        if osp.isdir(local_module_path):
            PACKAGE_PATH = os.getcwd()
            print(f"    Run `test` locally")
            
        elif osp.isdir(docker_volume):
            PACKAGE_PATH = docker_volume
            print(f"    Run `test` in docker container")
            
        else:
            raise OSError(f"Paths '{docker_volume}' and '{local_module_path}' do not exist!")

    if __name__=="__main__":    
        assert osp.isdir(WORKSPACE['work']), f"The path '{WORKSPACE['work']}' is not exist!"
        assert osp.isdir(WORKSPACE['component_volume']), f"The path '{WORKSPACE['component_volume']}' is not exist!"
        print(f"    Run `test` in component for pipeline")
        PACKAGE_PATH = WORKSPACE['component_volume']
        # for import sub_module
        package_repo_path = osp.join(WORKSPACE['component_volume'], cfg['git']['package_repo'])
        if not osp.isdir(package_repo_path):
            print(f" git clone 'sub_module' to {package_repo_path}")
            
            Repo.clone_from(f"git@github.com:HibernationNo1/{cfg['git']['package_repo']}.git", package_repo_path)
     
    sys.path.append(PACKAGE_PATH)   

    from sub_module.configs.pipeline import dict2Config
    from sub_module.configs.config import Config
    from sub_module.mmdet.inference import build_detector, inference_detector, parse_inference_result
    from sub_module.mmdet.modules.dataparallel import build_dp
    from sub_module.mmdet.visualization import draw_to_img

    from sub_module.mmdet.get_info_algorithm import Get_info

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
        
        batch_size = cfg.test_data.batch_size
        all_imgs_path = glob.glob(osp.join(cfg.test_data.data_root, "*.jpg"))
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




    if __name__=="component.test.test_op":  
        cfg = Config(cfg)
        main(cfg)

    if __name__=="__main__":    
        with open(input_run_flag, "r", encoding='utf-8') as f:
            input_run_flag = json.load(f) 

        if 'test' in input_run_flag['pipeline_run_flag']:
            cfg = dict2Config(cfg, key_name ='flag_list2tuple')    
            # git_clone_dataset(cfg)
            
            # main(cfg, in_pipeline = True)
        else:
            print(f"Pass component: test")
        
        return json.dump(input_run_flag, open(run_flag_path, "w"), indent=4)

test_op = create_component_from_func(func = test,
                                     base_image = base_image.test,
                                     output_component_file= base_image.test_cp)