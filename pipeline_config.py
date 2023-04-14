import os, os.path as osp

from pipeline_base_config import Config

CONFIGS = dict()     # parameters for pipeline run  
MAP_CONFIG = "config/map.py"

def set_cfg_pipeline(args, cfg):
    if args.pipeline_n is not None: cfg.kbf.pipeline.name = args.pipeline_n
    cfg.kbf.pipeline.version =  args.pipeline_v
    if args.experiment_n is not None: cfg.kbf.experiment.name = args.experiment_n
    if args.run_n is not None: cfg.kbf.run.name = args.run_n    
    cfg.kbf.dashboard.pw =  args.dashboard_pw 
    
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
    if CONFIGS['pipeline'] is not None:     # when run only by kubeflow pipeline
        if CONFIGS['pipeline'].kbf.volume.get("pvc", None) is not None:
            cfg.path.volume = CONFIGS['pipeline'].kbf.volume.pvc.mount_path
        
        
def set_cfg_record(args, cfg):
    assert cfg.dvc.record.train == cfg.record.train_dataset
    assert cfg.dvc.record.val == cfg.record.val_dataset
    
    comman_set(cfg)
    
     
def set_cfg_train(args, cfg):
    # set config of model to training 
    assert args.model, f"Model to be trained must be specified!!\n"\
        f"add `--model` option when entering the command."  
    
    if args.katib:
        cfg.katib = True

        # set 0 when running for katib or in pod (not have enough shared memory) 
        cfg.data.workers_per_gpu = 0

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
    
    if args.epoch is not None: cfg.max_epochs = args.epoch

    if args.lr is not None: cfg.optimizer.lr = float(args.lr)
   
    if cfg.model.backbone.type == "SwinTransformer":
        if args.swin_drop_rate is not None : 
            cfg.model.backbone.drop_rate = float(args.swin_drop_rate)
            assert 0.<=cfg.model.backbone.drop_rate and cfg.model.backbone.drop_rate < 0.999
        if args.swin_window_size is not None : 
            cfg.model.backbone.window_size = int(args.swin_window_size)
            assert cfg.model.backbone.window_size in [3, 5, 7, 9, 11, 13, 15]
        if args.swin_mlp_ratio is not None : 
            cfg.model.backbone.mlp_ratio = int(args.swin_mlp_ratio)
            assert cfg.model.backbone.mlp_ratio in [i for i in range(10)] 
         

    # If get dataset with dvc, load the paths from the database.
    # And all paths were set by dvc config
    if cfg.get('dvc', None) is not None:
        if args.cfg_pipeline is not None:
            cfg.data.train.data_root = cfg.data.val.data_root = osp.join(cfg.git.dataset.repo,
                                                                         cfg.dvc.record.dir,
                                                                         cfg.dvc.category)
            
            
            cfg.data.train.ann_file = osp.join(cfg.data.train.data_root, cfg.dvc.record.train)
            cfg.data.val.ann_file = osp.join(cfg.data.val.data_root, cfg.dvc.record.val)  
        else:
            cfg.pop('dvc')
  
    if args.pm_dilation is not None: cfg.model.backbone.pm_dilation = args.pm_dilation
    if args.drop_rate is not None: cfg.model.backbone.drop_rate = args.drop_rate
    if args.drop_path_rate is not None: cfg.model.backbone.drop_path_rate = args.drop_path_rate
    if args.attn_drop_rate is not None: cfg.model.backbone.attn_drop_rate = args.attn_drop_rate    
    
    if CONFIGS['pipeline'] is not None:     # when run only by kubeflow pipeline
        if CONFIGS['pipeline'].kbf.volume.get('pvc', None) is not None:
            for i, hook_cfg in enumerate(cfg.hook_configs):
                if hook_cfg.type == "TensorBoard_Hook":
                    cfg.hook_configs[i].pvc_dir = osp.join(CONFIGS['pipeline'].kbf.volume.pvc.mount_path,
                                                          cfg.hook_configs[i].pvc_dir)
                    break


def set_cfg_test(args, cfg):
    if args.model_path is not None: cfg.model_path = args.model_path  
    
    print(f"test: {cfg.get('data_root', None)}")
    

def sef_cfg_evaluate(args, cfg):
    if args.model_path is not None: cfg.model_path = args.model_path

    # If get dataset with dvc, load the paths from the database.
    # And all paths were set by dvc config
    if cfg.get('dvc', None) is not None:
        if args.cfg_pipeline is not None:
            # why set `cfg.data.train.data_root` even unused in `evaluate_op.py`?
            # To prevent conflicts between configs in `combine_config`
            cfg.data.train.data_root = cfg.data.val.data_root = osp.join(cfg.git.dataset.repo,
                                                                         cfg.dvc.record.dir,
                                                                         cfg.dvc.category)
            
            
            cfg.data.train.ann_file = osp.join(cfg.data.train.data_root, cfg.dvc.record.train)
            cfg.data.val.ann_file = osp.join(cfg.data.val.data_root, cfg.dvc.record.val) 
        else:
            cfg.pop('dvc')
    
    


CONFIG_SET_FUNCTION = dict(
    pipeline = set_cfg_pipeline,
    record = set_cfg_record,
    train = set_cfg_train,
    test = set_cfg_test,
    evaluate = sef_cfg_evaluate
)


def set_config(args):
    """ 
        cfg arguments determines which component be run.
        Components that matching cfg arguments which got `None` are excluded from the pipeline.
        cfg arguments: is chooses in [args.cfg_train, args.cfg_record]
    Args:
        args : argparse
    """

    if args.katib:
        if args.model is None:
            args.model = 'MaskRCNN' 
        if args.cfg_train is None:
            args.cfg_train = 'config/train_cfg.py'

 
    if (args.cfg_pipeline is not None) and (args.pipeline_v is not None) and (args.dashboard_pw is not None):
        print("Run with kubeflow pipeline")
        CONFIGS['pipeline'] = args.cfg_pipeline
        
    elif (args.cfg_pipeline is None) and (args.pipeline_v is None) and (args.dashboard_pw is None):
        print(f"Run without kubeflow pipleine")
        CONFIGS['pipeline'] = None
    else:
        raise ValueError(f"To run in pipeline of kubeflow, config, version and password of pipeline must be set.\n"\
                         f"add options --cfg_pipeline, --pipeline_v, --dashboard_pw")
           
    CONFIGS['train'] = args.cfg_train
    CONFIGS['record'] = args.cfg_record
    CONFIGS['test'] = args.cfg_test
    CONFIGS['evaluate'] = args.cfg_eval 

    for key, func in CONFIG_SET_FUNCTION.items(): 
        if CONFIGS[key] is not None:
            # Assign config only included in args 
            config =  Config.fromfile(CONFIGS[key])
            func(args, config)
        else: config = None
        # CONFIGS[key] = False or Config
        # if False, components matching the key will be passed from the pipeline.
        # >>    example
        # >>    CONFIGS[record] = False
        # >>    `record_op` component will be passed from the pipeline.
        CONFIGS[key] = config


