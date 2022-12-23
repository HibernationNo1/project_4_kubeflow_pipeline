_base_ = [
    'base/model.py',
    'base/dataset_config.py',
    'base/schedule_1x.py',
    'base/validation.py',
    "utils/dvc.py",
    "utils/database.py",
    "utils/gs.py"
]

log = dict(
    env = "enviroments",
    train = "train"
)

train_result = "result/train"
max_epochs = 5




hook_config = [
    dict(
        type='Validation_Hook',        
        priority = 'VERY_HIGH',     # be higher than loghook to log validation information.
        val = ['iter', 50],         # val_iter = 50         ['epoch', 1]
        show_eta_iter = 10),          # Divisor number of iter printing the training state log.
    dict(
        type='Check_Hook',
        priority = 'VERY_HIGH',    
        val_iter = 50,
        show_eta_iter = 10),
    dict(   
        type = "StepLrUpdaterHook",
        priority = 'HIGH',
        warmup='linear',
        warmup_iters=1000,
        warmup_ratio=0.001,
        step=[8, 11]),
    dict(
        type = "OptimizerHook",
        priority='ABOVE_NORMAL',
        grad_clip=None),
    dict(
        type = "CheckpointHook",
        priority='NORMAL',
        interval=1,         # epoch(or iter) unit to save model
        filename_tmpl = 'model_{}.pth'),        # model name to be save :  {model_name}_{epoch}.pth
    dict(
        type = "LoggerHook",
        # priority='VERY_LOW',      # default priority: 'VERY_LOW'
        interval=50,
        out_dir = train_result,
        max_epochs = max_epochs,
        ev_iter = None,             # set in register_training_hooks
        out_suffix = '.log'
        ),         
    dict(
        type = "IterTimerHook"  
        # priority='VERY_LOW',
    )   
]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
# workflow = [('train', 5)]   # TODO : [('train', n_1), ('val', n_2)]     n_1: epoch



device = 'cuda:0'

       

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)