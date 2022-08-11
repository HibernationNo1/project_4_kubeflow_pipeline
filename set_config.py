from genericpath import exists
from kfp.components import OutputPath, create_component_from_func
from config import SETUP_IMAGE, SETUP_COM_FILE  
        

def set_config(args: dict, config_path: OutputPath("dict"), ) :

    import os 
    import pipeline_taeuk4958
    print(f"\n pipeline_taeuk4958.__version__ : {pipeline_taeuk4958.__version__} \n")
    from pipeline_taeuk4958.configs.config import Config

    pipeline_taeuk4958_config_path ='/usr/local/lib/python3.8/site-packages/pipeline_taeuk4958/configs'
    cfg_file_path = os.path.join(pipeline_taeuk4958_config_path, args['cfg'])        
    cfg = Config.fromfile(cfg_file_path)                # cfg_file_path : must be .py format
    
    import subprocess
    talk_1 = "echo $AWS_ACCESS_KEY_ID"
    talk_2 = "echo $AWS_SECRET_ACCESS_KEY"
    subprocess.call([talk_1], shell=True)
    subprocess.call([talk_2], shell=True)
    exit()
    if args['p_name'] is not None : cfg.pipeline.pipeline_name = args['p_name']
    if args['p_version'] is not None : cfg.pipeline.pipeline_version = args['p_version']
    
    if args['mode'] == "record":
        if args['proportion_val'] is not None : cfg.options.proportion_val = args['proportion_val']
        if args['dataset'] is not None : cfg.dataset.dataset_dir = args['dataset']
        
        if args['client_secrets'] is not None : cfg.gs.client_secrets = args['client_secrets']
        if args['ann_bk_name'] is not None : cfg.gs.ann_bucket_name = args['ann_bk_name']
        if args['dataset_bk_name'] is not None : cfg.gs.recoded_dataset_bucket_name = args['dataset_bk_name']
        if args['d_version'] is not None : cfg.gs.recoded_dataset_version = args['d_version']
    
    elif args['mode'] == "train":
        if args['train_json'] is not None : cfg.dataset.train_file_name = args['train_json']
        if args['val_json'] is not None : cfg.dataset.val_file_name = args['val_json']
        if args['validate'] : cfg.train.validate = True
        if args['finetun'] : cfg.train.finetun = True
        if args['model_vers'] is not None : cfg.train.model_version = args['model_vers']
        if args['d_version_t'] is not None : cfg.dataset.dataset_version = args['d_version_t']
    
    
     
    cfg.dump(config_path)                               # save to config_path : {wiorkspace}/outputs/set_config/data
    
    
set_config_op = create_component_from_func(func = set_config,
                                        base_image = SETUP_IMAGE,
                                        output_component_file= SETUP_COM_FILE)

