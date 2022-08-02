from genericpath import exists
from kfp.components import OutputPath, create_component_from_func
from config import SETUP_IMAGE      

def set_config(args: dict, config_path: OutputPath("dict")):

    import os 
    import json
  
    from configs.config import Config
    from utils.utils import filter_config
    
    pipeline_taeuk4958_config_path ='/usr/local/lib/python3.8/site-packages/pipeline_taeuk4958/configs'
    cfg_file_path = os.path.join(pipeline_taeuk4958_config_path, args['cfg'])
    cfg = Config.fromfile(cfg_file_path)
    class_cfg_dict = {}
    for key in list(cfg._cfg_dict.keys()):
        class_cfg_dict[key] = cfg._cfg_dict[key]
        
    cfg_dict = filter_config(class_cfg_dict, cfg_dict)
    json.dump(cfg_dict, open(config_path, "w"), indent = 4)
    
    
    os.listdir(os.getcwd())
    os.listdir(os.path.join(os.getcwd(), "project_4_kubeflow_pipeline")
    
    exit()
    import subprocess
    import glob
    access_key_id = f"dvc remote modify --local storage access_key_id {args['access_key_id']}"
    secret_access_key = f"dvc remote modify --local storage access_key_id {args['secret_access_key']}"
    
    subprocess.call([access_key_id], shell=True)
    subprocess.call([secret_access_key], shell=True)
    subprocess.call(["dvc pull"], shell=True)
    data_dir = os.path.join(os.getcwd(), cfg_dict['dataset_dir'])
    print(f"data_dir : {data_dir}")
    
    dataset_list = glob.glob(f"{data_dir}.*json")
    print(f"\n dataset_list : {dataset_list}")
    

set_config_op = create_component_from_func(func = set_config,
                                        base_image = SETUP_IMAGE,
                                        output_component_file="set_config.component.yaml")

