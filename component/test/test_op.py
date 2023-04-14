from kfp.components import create_component_from_func, InputPath, OutputPath

from pipeline_base_config import Base_Image_cfg
base_image = Base_Image_cfg()


def test(cfg : dict, input_run_flag: InputPath("dict"),
         run_flag_path: OutputPath('dict')):       # Error when using variable name 'run_flag_path'

    import json
    import os, os.path as osp
    import numpy as np
    import glob
    import torch
    import cv2
    import sys
    from git.repo import Repo
    from google.cloud import storage

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
    from sub_module.cloud.google.storage import set_gs_credentials, get_client_secrets
    from sub_module.mmdet.inference import build_detector, inference_detector, parse_inference_result
    from sub_module.mmdet.modules.dataparallel import build_dp
    from sub_module.mmdet.visualization import draw_to_img

    from sub_module.mmdet.get_info_algorithm import Get_info

    def main(cfg, test_result, in_pipeline = False):   
        os.makedirs(test_result, exist_ok = True)
        
        batch_size = cfg.test_data.batch_size
        
        if in_pipeline:
            input_data_dir = osp.join(WORKSPACE['component_volume'], cfg.test_data.data_root)
        else:
            input_data_dir = osp.join(os.getcwd(), cfg.test_data.data_root)
        
        all_imgs_path = glob.glob(osp.join(input_data_dir, "*.jpg"))
        batch_imgs_list = [all_imgs_path[x:x + batch_size] for x in range(0, len(all_imgs_path), batch_size)]
        
        if in_pipeline:
            if cfg.gs.download:
                model_dir = download_model(cfg)
            else:
                model_dir = osp.join(WORKSPACE['component_volume'], cfg.eval_result)
        else:
            if cfg.get('model_path', None) is None:
                model_dir = osp.join(os.getcwd(), cfg.eval_result)
            else:
                model_dir = osp.join(os.getcwd(), osp.dirname(cfg.model_path))
            
        model_path_list = list()
        def get_model_list(dir_path):
            # Get path only models
            if osp.isdir(dir_path):
                for path_ in os.listdir(dir_path):
                    get_model_list(osp.join(dir_path, path_))       
            elif osp.isfile(dir_path):
                if osp.splitext(dir_path)[-1] == ".pth":
                    model_path_list.append(dir_path)
        get_model_list(model_dir)   
                  
        for model_path in model_path_list:
            print(f"\nRun inference - model name: {model_path}")
            model = build_detector(cfg, model_path, device = cfg.device)
        
            dp_cfg = dict(model = model, 
                        cfg = cfg,
                        device = cfg.device,
                        classes = model.CLASSES)
            model = build_dp(**dp_cfg)
        

            for batch_imgs in batch_imgs_list:
                if not isinstance(batch_imgs, list) : batch_imgs = list(batch_imgs)
                
                with torch.no_grad():
                    # len: batch_size
                    batch_results = inference_detector(model, batch_imgs)   

                for img_path, results in zip(batch_imgs, batch_results):
                    out_file = osp.join(cfg.test_result, osp.basename(img_path))
                    
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
                    if len(number_board_list) == 0: continue
                    print(f"img_path : {img_path}")
                    for number_board in number_board_list:
                    	print(f"number_board : {number_board}")


    def download_model(cfg):
        set_gs_credentials(get_client_secrets())
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(cfg.gs.result_bucket)
        
        if cfg.gs.test.get('target', None) is None:
            gs_path = cfg.gs.eval.dir
        else:
            gs_path = osp.join(cfg.gs.eval.dir, cfg.gs.test.target)
        
        blobs = bucket.list_blobs(prefix=gs_path)
       
        model_dir = osp.join(os.getcwd(), cfg.gs.test.get('download_dir', "download"))   
        for blob in blobs:
            bucket_path = blob.name[len(gs_path):]
            if osp.isabs(bucket_path):
                bucket_path = bucket_path.lstrip('/')
            
            if osp.splitext(bucket_path)[-1] != ".pth": continue
            local_file_path = os.path.join(model_dir, bucket_path)
            os.makedirs(osp.dirname(local_file_path), exist_ok = True)
            print(f"Download From `gs:{cfg.gs.result_bucket}/{bucket_path}` to `{local_file_path}`")
            blob.download_to_filename(local_file_path)

        return model_dir
    
    

    if __name__=="component.test.test_op":  
        cfg = Config(cfg)
        main(cfg)

    if __name__=="__main__":    
        with open(input_run_flag, "r", encoding='utf-8') as f:
            input_run_flag = json.load(f) 

        if 'test' in input_run_flag['pipeline_run_flag']:
            cfg = dict2Config(cfg, key_name ='flag_list2tuple')    
            main(cfg, osp.join(WORKSPACE['component_volume'], cfg.test_result), in_pipeline = True)
            
    
        else:
            print(f"Pass component: test")
        
        return json.dump(input_run_flag, open(run_flag_path, "w"), indent=4)

test_op = create_component_from_func(func = test,
                                     base_image = base_image.test,
                                     output_component_file= base_image.test_cp)