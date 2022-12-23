import os, os.path as osp

from hibernation_no1.configs.config import Config

CONFIGS = dict()     # parameters for pipeline run  

def set_cfg_recode(args, cfg):
    assert cfg.dvc.recode.train == cfg.recode.train_dataset
    assert cfg.dvc.recode.val == cfg.recode.val_dataset
   
def set_cfg_pipeline(args, cfg):
    if args.pipeline_n is not None: cfg.kbf.pipeline.name = args.pipeline_n
    cfg.kbf.pipeline.version =  args.pipeline_v
    if args.experiment_n is not None: cfg.kbf.experiment.name = args.experiment_n
    if args.run_n is not None: cfg.kbf.run.name = args.run_n    
    cfg.kbf.dashboard.pw =  args.pipeline_pw 
    
    
     
def set_cfg_train(args, cfg):
    # set config of model to training 
    assert args.model, f"Model to be trained must be specified!!\n"\
        f"add `--model` option when entering the command."    
    models_cfg_path = osp.join(os.getcwd(), 
                              CONFIGS['pipeline'].dir.config.name, 
                              CONFIGS['pipeline'].dir.config.models)
    
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
    
    if args.save_model_name is not None: cfg.checkpoint_config.filename_tmpl = f"{args.save_model_name}"+"_{}.path"
    if args.val_iter is not None: 
        for i, custom_hook in enumerate(cfg.custom_hook_config):
            if custom_hook.get('Validation_Hook', None) is not None:
                 cfg.custom_hook_config[i].val = args.val_iter
                 break
    
    # If get dataset with dvc, load the paths from the database.
    # And all paths were set by dvc config
    if cfg.get('dvc', None) is not None:
        cfg.data.train.data_root = cfg.data.val.data_root = osp.join(cfg.dvc.category, 
                                                                     cfg.dvc.recode.name, 
                                                                     cfg.dvc.recode.version)
        cfg.data.train.ann_file = cfg.dvc.recode.train
        cfg.data.val.ann_file = cfg.dvc.recode.val
    

    if args.pm_dilation is not None: cfg.model.backbone.pm_dilation = args.pm_dilation
    if args.drop_rate is not None: cfg.model.backbone.drop_rate = args.drop_rate
    if args.drop_path_rate is not None: cfg.model.backbone.drop_path_rate = args.drop_path_rate
    if args.attn_drop_rate is not None: cfg.model.backbone.attn_drop_rate = args.attn_drop_rate    


CONFIG_SET_FUNCTION = dict(
    pipeline = set_cfg_pipeline,
    recode = set_cfg_recode,
    train = set_cfg_train
)


def set_config(args):
    """ 
        cfg arguments determines which component be run.
        Components that matching cfg arguments which got `None` are excluded from the pipeline.
        cfg arguments: is chooses in [args.cfg_train, args.cfg_recode]
    Args:
        args : argparse
    """

    CONFIGS['pipeline'] = args.cfg_pipeline
    CONFIGS['train'] = args.cfg_train
    CONFIGS['recode'] = args.cfg_recode
    
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


