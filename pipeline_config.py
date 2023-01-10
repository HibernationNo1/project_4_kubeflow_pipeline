import os, os.path as osp

from docker.hibernation_no1.configs.config import Config

CONFIGS = dict()     # parameters for pipeline run  
MAP_CONFIG = "config/map.py"

def set_cfg_pipeline(args, cfg):
    if args.pipeline_n is not None: cfg.kbf.pipeline.name = args.pipeline_n
    cfg.kbf.pipeline.version =  args.pipeline_v
    if args.experiment_n is not None: cfg.kbf.experiment.name = args.experiment_n
    if args.run_n is not None: cfg.kbf.run.name = args.run_n    
    cfg.kbf.dashboard.pw =  args.pipeline_pw 
    
    if cfg.kbf.volume.get("pvc", None) is not None:
        import kfp.dsl as dsl
        mode = cfg.kbf.volume.pvc.mode
        if mode == 'VOLUME_MODE_RWO':
            cfg.kbf.volume.pvc.mode = dsl.VOLUME_MODE_RWO
        elif mode == 'VOLUME_MODE_RWM':
            cfg.kbf.volume.pvc.mode = dsl.VOLUME_MODE_RWM
        elif mode == 'VOLUME_MODE_ROM':
            cfg.kbf.volume.pvc.mode = dsl.VOLUME_MODE_ROM
            
       
            
def comman_set(cfg):
    if CONFIGS['pipeline'].kbf.volume.get("pvc", None) is not None:
        cfg.volume_path = CONFIGS['pipeline'].kbf.volume.pvc.mount_path
        
        
def set_cfg_recode(args, cfg):
    assert cfg.dvc.recode.train == cfg.recode.train_dataset
    assert cfg.dvc.recode.val == cfg.recode.val_dataset
    
    comman_set(cfg)
    
     
def set_cfg_train(args, cfg):
    # set config of model to training 
    assert args.model, f"Model to be trained must be specified!!\n"\
        f"add `--model` option when entering the command."  
    
    comman_set(cfg)
    
    map_cfg = Config.fromfile(MAP_CONFIG)
    models_cfg_path = osp.join(os.getcwd(), 
                                map_cfg.dir.config.name, 
                                map_cfg.dir.config.models)        
    
    model_cfg = Config.fromfile(osp.join(models_cfg_path, f"{args.model}.py")) 
    for key, item in cfg.model.get(args.model).items():
        sub_cfg = Config.fromfile(osp.join(models_cfg_path, key, f"{item}.py"))
        if key == 'backbone':
            model_cfg.model.backbone = sub_cfg.get(key)
            
        if key == 'neck':
            model_cfg.model.neck = sub_cfg.get(key)
    cfg.model = model_cfg.model
    
    if args.name_db is not None: cfg.db.db = args.name_db 
    if args.user_db is not None: cfg.db.user = args.user_db 
    
    
    # for katib
    if args.lr is not None: cfg.optimizer.lr = float(args.lr)
    if args.wd is not None: 
        cfg.optimizer.weight_decay = float(args.wd)
        assert 0.<cfg.optimizer.weight_decay and cfg.optimizer.weight_decay < 0.999

    if cfg.model.backbone.type == "SwinTransformer":
        if args.swin_drop_rate is not None : 
            cfg.model.backbone.drop_rate = float(args.swin_drop_rate)
            assert 0.<cfg.optimizer.weight_decay and cfg.optimizer.weight_decay < 0.999
        if args.swin_window_size is not None : 
            cfg.model.backbone.window_size = int(args.swin_window_size)
            assert cfg.optimizer.window_size in [1, 3, 5, 7, 9, 11, 13, 15]
        if args.swin_mlp_ratio is not None : 
            cfg.model.backbone.mlp_ratio = int(args.swin_mlp_ratio)
            assert cfg.optimizer.mlp_ratio in [i for i in range(10)] 
         
    
    # If get dataset with dvc, load the paths from the database.
    # And all paths were set by dvc config
    if cfg.get('dvc', None) is not None:
        if args.cfg_pipeline is not None:
            cfg.data.train.data_root = cfg.data.val.data_root = osp.join(cfg.dvc.category, 
                                                                        cfg.dvc.recode.name, 
                                                                        cfg.dvc.recode.version)
            cfg.data.train.ann_file = cfg.dvc.recode.train
            cfg.data.val.ann_file = cfg.dvc.recode.val    

    if args.pm_dilation is not None: cfg.model.backbone.pm_dilation = args.pm_dilation
    if args.drop_rate is not None: cfg.model.backbone.drop_rate = args.drop_rate
    if args.drop_path_rate is not None: cfg.model.backbone.drop_path_rate = args.drop_path_rate
    if args.attn_drop_rate is not None: cfg.model.backbone.attn_drop_rate = args.attn_drop_rate    
    
    
    if CONFIGS['pipeline'].kbf.volume.get('pvc', None) is not None:
        for i, hook_cfg in enumerate(cfg.hook_config):
            if hook_cfg.type == "TensorBoard_Hook":
                cfg.hook_config[i].out_dir = osp.join(CONFIGS['pipeline'].kbf.volume.pvc.mount_path,
                                                      cfg.hook_config[i].out_dir)
                break
            

def set_cfg_infer(args, cfg):
    if args.model_path is None:
        raise KeyError(f"Path of model file must be specific to run inference, but got None..  add option '--model_path'")
    
    cfg.model_path = args.model_path
    
    pass

CONFIG_SET_FUNCTION = dict(
    pipeline = set_cfg_pipeline,
    recode = set_cfg_recode,
    train = set_cfg_train,
    test = set_cfg_infer,
)


def set_config(args):
    """ 
        cfg arguments determines which component be run.
        Components that matching cfg arguments which got `None` are excluded from the pipeline.
        cfg arguments: is chooses in [args.cfg_train, args.cfg_recode]
    Args:
        args : argparse
    """
    
    if (args.cfg_pipeline is not None) and (args.pipeline_v is not None) and (args.pipeline_pw is not None):
        print("Run with kubeflow pipeline")
        CONFIGS['pipeline'] = args.cfg_pipeline
        
    elif (args.cfg_pipeline is None) and (args.pipeline_v is None) and (args.pipeline_pw is None):
        print(f"Run without kubeflow pipleine")
        CONFIGS['pipeline'] = None
    else:
        raise ValueError(f"To run in pipeline of kubeflow, config, version and password of pipeline must be set.\n"\
                         f"add options --cfg_pipeline, --pipeline_v, --pipeline_pw")
           
     
    CONFIGS['train'] = args.cfg_train
    CONFIGS['recode'] = args.cfg_recode
    CONFIGS['test'] = args.cfg_infer
   
    for key, func in CONFIG_SET_FUNCTION.items():        
        if CONFIGS[key] is not None:
            # Assign config only included in args 
            config =  Config.fromfile(CONFIGS[key])
            func(args, config)
        else: config = None
        # CONFIGS[key] = False or Config
        # if False, components matching the key will be excluded from the pipeline.
        # >>    example
        # >>    CONFIGS[recode] = False
        # >>    `recode_op` component will be excluded from the pipeline.
        CONFIGS[key] = config


