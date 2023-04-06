from kfp.components import create_component_from_func, InputPath, OutputPath

from pipeline_base_config import Base_Image_cfg
base_image = Base_Image_cfg()

def evaluate(cfg : dict, input_run_flag: InputPath("dict"),
             run_flag_path: OutputPath('dict')):       # Error when using variable name 'run_flag_path'
    import os, os.path as osp
    import sys
    import json
    from git.repo import Repo

    WORKSPACE = dict(component_volume = cfg['path']['component_volume'],       # pvc volume path on component container
                     local_volume = cfg['path']['local_volume'],     # pvc volume path on local
                     docker_volume = cfg['path']['docker_volume'],     # volume path on katib container
                     work = cfg['path']['work_space']
                     )    

    # set package path to 'import {custom_package}'
    if __name__=="component.evaluate.evaluate_op":
        local_volume = osp.join('/opt/local-path-provisioner', WORKSPACE['local_volume']) 
        docker_volume = f"/{WORKSPACE['docker_volume']}"
        if osp.isdir(local_volume):
            PACKAGE_PATH = local_volume
            print(f"    Run `evaluate` locally")
            
        elif osp.isdir(docker_volume):
            PACKAGE_PATH = docker_volume
            print(f"    Run `evaluate` in container for katib")
            
        else:
            raise OSError(f"Paths '{docker_volume}' and '{local_volume}' do not exist!")

    if __name__=="__main__":    
        assert osp.isdir(WORKSPACE['work']), f"The path '{WORKSPACE['work']}' is not exist!"
        assert osp.isdir(WORKSPACE['component_volume']), f"The path '{WORKSPACE['component_volume']}' is not exist!"
        print(f"    Run `evaluate` in component for pipeline")
        PACKAGE_PATH = WORKSPACE['component_volume']
        # for import sub_module
        package_repo_path = osp.join(WORKSPACE['component_volume'], cfg['git']['package_repo'])
        if not osp.isdir(package_repo_path):
            print(f" git clone 'sub_module' to {package_repo_path}")
            
            Repo.clone_from(f"git@github.com:HibernationNo1/{cfg['git']['package_repo']}.git", package_repo_path)
     
    sys.path.append(PACKAGE_PATH)   

    from sub_module.configs.pipeline import dict2Config
    from sub_module.configs.config import Config
    from sub_module.mmdet.inference import build_detector
    from sub_module.mmdet.modules.dataparallel import build_dp
    from sub_module.mmdet.visualization import draw_to_img

    from sub_module.mmdet.data.dataset import build_dataset
    from sub_module.mmdet.data.dataloader import build_dataloader
    from sub_module.mmdet.eval import Evaluate

    import numpy as np

    def main(cfg, in_pipeline = False):
        cfg.seed = np.random.randint(2**31)


        val_cfg = cfg.data.val.copy()
        val_cfg.pop("batch_size")
        _, val_dataset = build_dataset(val_cfg = val_cfg)
    
        dataloader_cfg = dict(val_dataset = val_dataset,
                            val_batch_size = cfg.data.val.batch_size,
                            num_workers=cfg.data.workers_per_gpu,
                            seed = cfg.seed,
                            shuffle = True)
        _, val_dataloader = build_dataloader(**dataloader_cfg)



        if in_pipeline:
            # model: bring from google storage
            # output_path: volume,      bucket
            pass
        else:
            if osp.isfile(cfg.model_path):
                model_path = osp.join(os.getcwd(), cfg.model_path)
            else:
                raise OSError(f"The path is not exist!!     path : {model_path}")

            output_path = osp.join(os.getcwd(), cfg.val_result) 
        os.makedirs(output_path, exist_ok = True) 


        model = build_detector(cfg, model_path, device = cfg.device)
        dp_cfg = dict(model = model, 
                    cfg = cfg,
                    device = cfg.device,
                    classes = model.CLASSES)
        model = build_dp(**dp_cfg)


        eval_cfg = dict(model= model, 
                        output_path = output_path,
                        cfg= cfg.eval_cfg,
                        dataloader= val_dataloader)
        eval_ = Evaluate(**eval_cfg)   
        summary = eval_.compute_mAP()
         
        correct_inference_rate = eval.run_inference()
        if correct_inference_rate is None:
            json.dump(summary, open(osp.join(output_path, "summary.json"), "w"), indent=4)
        else: 
            summary['correct_inference_rate'] = correct_inference_rate
            json.dump(summary, open(osp.join(output_path, "summary.json"), "w"), indent=4)
            

    if __name__=="component.evaluate.evaluate_op":  
        cfg = Config(cfg)
        main(cfg)
        
        
    if __name__=="__main__":    
        with open(input_run_flag, "r", encoding='utf-8') as f:
            input_run_flag = json.load(f) 

        if 'evaluate' in input_run_flag['pipeline_run_flag']:
            print("Run component: evaluate")
            cfg = dict2Config(cfg, key_name ='flag_list2tuple')    
            
            # main(cfg, in_pipeline = True)
        else:
            print(f"Pass component: evaluate")
        
        return json.dump(input_run_flag, open(run_flag_path, "w"), indent=4)

   
evaluate_op = create_component_from_func(func = evaluate,
                                         base_image = base_image.evaluate,
                                         output_component_file= base_image.evaluate_cp)