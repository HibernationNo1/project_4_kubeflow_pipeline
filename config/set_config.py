from genericpath import exists
from kfp.components import OutputPath, create_component_from_func
from pipeline_config import Pipeline_Config
pl_cfg = Pipeline_Config    


def set_config(args: dict, config_path: OutputPath("dict")) :
    
    
    import os             
    import pipeline_taeuk4958
    from pipeline_taeuk4958.configs.config import Config as Config_pipeline
    print(f"\n pipeline_taeuk4958.__version__ : {pipeline_taeuk4958.__version__} \n")
        
    import mmcv
    from mmcv import Config
    
    
    pipeline_taeuk4958_config_path = "/opt/conda/lib/python3.7/site-packages/pipeline_taeuk4958/configs"   # '/usr/local/lib/python3.8/site-packages/pipeline_taeuk4958/configs'
    cfg_file_path = os.path.join(pipeline_taeuk4958_config_path, args['cfg'])    
    
    
    if args['mode'] == "record":
        cfg = Config_pipeline.fromfile(cfg_file_path)                # cfg_file_path : must be .py format    
    elif args['mode'] == "train":
    
        cfg = Config.fromfile(cfg_file_path)
        assert isinstance(cfg, mmcv.Config), \
        f'cfg got wrong type: {type(cfg)}, expected mmcv.Config'
    
    import torch
    print(f"torch.cuda.is_available() ; {torch.cuda.is_available()}")
    print(f"torch.__version__ : {torch.__version__}")

    exit()
    cfg.pipeline.pipeline_name = args['p_name']
    if args['p_version'] is not None : cfg.pipeline.pipeline_version = args['p_version']
    
    if args['mode'] == "record":
        if args['proportion_val'] is not None : cfg.options.proportion_val = args['proportion_val']
        if args['dataset'] is not None : cfg.dataset.dataset_dir = args['dataset']
        
        if args['client_secrets'] is not None : cfg.gs.client_secrets = args['client_secrets']
        if args['ann_bk_name'] is not None : cfg.gs.ann_bucket_name = args['ann_bk_name']
        if args['dataset_bk_name'] is not None : cfg.gs.recoded_dataset_bucket_name = args['dataset_bk_name']
        if args['d_version'] is not None : cfg.gs.recoded_dataset_version = args['d_version']
    
    elif args['mode'] == "train":
        if args['train_json'] is not None : cfg.train_dataset_json = args['train_json']
        if args['val_json'] is not None : cfg.val_dataset_json = args['val_json']
        if args['validate'] : cfg.train.validate = True
        if args['finetun'] : cfg.train.finetun = True
        if args['model_vers'] is not None : cfg.train.model_version = args['model_vers']
        
    
    
    cfg.dump(config_path)                               # save to config_path : {wiorkspace}/outputs/set_config/data
    
print(f"set_config base_image : {pl_cfg.SETCONFIG_IMAGE}")
set_config_op = create_component_from_func(func = set_config,
                                        base_image = pl_cfg.SETCONFIG_IMAGE,
                                        output_component_file= pl_cfg.SETCONFIG_COM_FILE)

