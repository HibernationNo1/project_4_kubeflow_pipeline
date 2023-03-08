from kfp.components import create_component_from_func, OutputPath

from pipeline_base_config import Base_Image_cfg
base_image = Base_Image_cfg()

def evaluate(cfg):
    import os, os.path as osp
    import sys
    import json
    from git.repo import Repo

    WORKSPACE = dict(volume = cfg['path']['volume'],       # pvc volume path
                     work =  cfg['path']['work_space'],     # path if workspace in docker container
                     local_package = cfg['path']['local_volume'])    


    if __name__=="component.evaluate.evaluate_op": 
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
    from hibernation_no1.mmdet.inference import build_detector
    from hibernation_no1.mmdet.modules.dataparallel import build_dp
    from hibernation_no1.mmdet.visualization import draw_to_img

    from hibernation_no1.mmdet.data.dataset import build_dataset
    from hibernation_no1.mmdet.data.dataloader import build_dataloader
    from hibernation_no1.mmdet.eval import Evaluate

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
            


    def dict2Config(cfg):
        cfg_flag = cfg.get('flag', None)
        if cfg_flag is not None:
            cfg = change_to_tuple(cfg, cfg_flag)
        cfg = Config(cfg)
        return cfg


    if __name__=="component.evaluate.evaluate_op":  
        cfg = dict2Config(cfg)
        main(cfg)
        
        
    if __name__=="__main__":       
        cfg = dict2Config(cfg)       
         
        # git_clone_dataset(cfg)              
        # main(cfg, in_pipeline = True)        
        # upload_models(cfg)

   
evaluate_op = create_component_from_func(func = evaluate,
                                         base_image = base_image.evaluate,
                                         output_component_file= base_image.evaluate_cp)