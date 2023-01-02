_base_ = [
    'base/model.py',
    'base/dataset_config.py',
    'base/schedule_1x.py',
    'base/validation.py',
    "utils/dvc.py",
    "utils/database.py",
    "utils/gs.py"
]


train_result = "result/train"

max_epochs = 30

dist_params = dict(backend='nccl')      # TODO: ?

load_from = None
resume_from = None

device = 'cuda:0'


hook_config = [
    dict(
        type='Validation_Hook',        
        priority = 'VERY_HIGH',     # be higher than loghook to log validation information.
        interval = ['iter', 50]     # epoch(or iter) unit to run validation ['epoch', 1]
        ),          
    dict(
        type='Check_Hook',
        priority = 'VERY_HIGH'),
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
        interval=['epoch', 1],                  # epoch(or iter) unit to save model    ['iter', 2000]
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
        type = "IterTimerHook",  
        show_eta_iter = 10,         # Divisor number of iter printing the training state log.
        priority='LOW'            # more important than LoggerHook for log_buffer.update
    )   
]



log = dict(
    level = 'INFO',
    env = "enviroments",
    train = "train"
)

       

# disable opencv multithreading to avoid system being overloaded        # TODO: using
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training     # TODO: using
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)      # TODO: using