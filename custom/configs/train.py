


checkpoint_config = dict(
    interval=1,         # number of epoch(or iter) to save model
    filename_tmpl = 'model_{}.pth'        # model name to be save :  {model_name}_{epoch}.pth
    )
# yapf:disable
log_config = dict(
    interval=50,        # number of epoch(or iter) to save log
    hooks=[
        dict(type='LoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 5)]   # TODO : [('train', n_1), ('val', n_2)]     n_1: epoch

result = "result/train"
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