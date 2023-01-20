_base_ = [
    'base/model.py',
    'base/dataset_config.py',
    'base/schedule_1x.py',
    "utils/dvc.py",
    "utils/database.py",
    "utils/utils.py"
]


train_result = "result/train"

dist_params = dict(backend='nccl')      # TODO: ?

device = 'cuda:0'


hook_config = [
    dict(
        type='Validation_Hook',        
        priority = 'VERY_HIGH',     # be higher than loghook to log validation information.
        interval = ['iter', 50],     # epoch(or iter) unit to run validation ['epoch', 1]
        val_cfg = dict(
            iou_thrs = [0.3, 0.9],          # range of iou threshold
            num_thrs_divi = 10,             # divide range of `iou_threshold` by number of `num_thrshd_divi` for compute mAP
            confidence_thrs = 0.6,                 # threshold of confidence score
            batch_size = 4)
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
        interval=20,                # unit: iter
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
        pvc_dir = "tensorboard",
        out_dir = train_result          
        # priority='VERY_LOW'       # default priority: 'VERY_LOW'
    )      
]



log = dict(
    level = 'INFO',
    env = "enviroments",
    train = "train"
    # val = "validation"
)
