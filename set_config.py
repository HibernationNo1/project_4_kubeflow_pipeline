from kfp.components import OutputPath, create_component_from_func
from config import SETUP_IMAGE      

def set_config(args: dict, config_path: OutputPath("dict")):

    import os 
    import json
    import pipeline_taeuk4958
    print(f"pipeline_taeuk4958.__version__ : {pipeline_taeuk4958.__version__}")
    from pipeline_taeuk4958.configs.config import Config
    from pipeline_taeuk4958.utils.utils import filter_config
    
    pipeline_taeuk4958_config_path ='/usr/local/lib/python3.8/site-packages/pipeline_taeuk4958/configs'
    cfg_file_path = os.path.join(pipeline_taeuk4958_config_path, args['cfg'])
    cfg = Config.fromfile(cfg_file_path)
    class_cfg_dict = {}
    for key in list(cfg._cfg_dict.keys()):
        class_cfg_dict[key] = cfg._cfg_dict[key]
        
    cfg_dict = filter_config(class_cfg_dict, cfg_dict)
    json.dump(cfg_dict, open(config_path, "w"), indent = 4)
    

set_config_op = create_component_from_func(func = set_config,
                                        base_image = SETUP_IMAGE,
                                        output_component_file="set_config.component.yaml")

