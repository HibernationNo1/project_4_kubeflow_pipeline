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

dist_params = dict(backend='nccl')      # TODO: ?

device = 'cuda:0'


hook_config = [
    dict(
        type='Validation_Hook',        
        priority = 'VERY_HIGH',     # be higher than loghook to log validation information.
        interval = ['iter', 50]     # epoch(or iter) unit to run validation ['epoch', 1]
    ),          
    dict(
        type='Check_Hook',
        priority = 'VERY_HIGH'
    ),
    dict(   
        type = "StepLrUpdaterHook",
        priority = 'HIGH',
        warmup='linear',
        warmup_iters=1000,              # last iteration at the end of warmup
        warmup_ratio=0.001,
        step=[10, 15, 20, 25, 30],      # epoch step to apply gamma by the learning rate
        gamma = 0.1                     # value of the exponent to apply by the learning rate
    ),
    dict(
        type = "OptimizerHook",
        priority='ABOVE_NORMAL',
        grad_clip=None
    ),
    dict(
        type = "CheckpointHook",
        priority='NORMAL',
        interval=['epoch', 1],                  # epoch(or iter) unit to save model    ['iter', 2000]
        filename_tmpl = 'model_{}.pth'          # model name to be save :  {model_name}_{epoch}.pth
    ),
    dict(
        type = "LoggerHook", 
        interval=50,                # unit: iter
        out_dir = train_result,
        out_suffix = '.log',        #  if '.json', can write max 21889 lines
    ),         
    dict(
        type = "IterTimerHook",  
        show_eta_iter = 20,         # Divisor number of iter printing the training state log.
        priority='LOW'              # more important than LoggerHook for log_buffer.update
    ),
    dict(
        type = "TensorBoard_Hook",
        interval = ['iter', 10],
        out_dir = "tensorboard"              # tensorboard --logdir=
        # priority='VERY_LOW'       # default priority: 'VERY_LOW'
    )      
]



log = dict(
    level = 'INFO',
    env = "enviroments",
    train = "train"
    # val = "validation"
)
