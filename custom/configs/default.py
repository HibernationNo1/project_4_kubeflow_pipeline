


checkpoint_config = dict(
    interval=1,         # epoch(or iter) unit to save model
    filename_tmpl = 'model_{}.pth'        # model name to be save :  {model_name}_{epoch}.pth
    )
# yapf:disable
log_config = dict(
    interval=50,        # iter unit to write log
    hooks=[
        dict(type='LoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# yapf:enable
custom_hook_config = [dict(
    type='Custom_Hook',
    priority = 'VERY_HIGH',     # be higher than loghook to log validation information.
    val_iter = 200,
    show_eta_iter = 10
    )]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 5)]   # TODO : [('train', n_1), ('val', n_2)]     n_1: epoch

device = 'cuda:0'

# validation
iou_threshold = [0.3, 0.9]      # range of iou threshold
num_thrshd_divi = 10            # divide range of `iou_threshold` by number of `num_thrshd_divi`
confidence_threshold = 0.6             

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)