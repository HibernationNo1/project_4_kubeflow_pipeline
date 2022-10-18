


checkpoint_config = dict(
    interval=1,
    filename_tmpl = 'model_{}.pth'        # model name to be save :  {model_name}_{epoch}.pth
    )
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 5)]   # TODO : [('train', n_1), ('val', n_2)]    
epoch_or_iter = "epoch"      # training을 epoch단위로할지 iter단위로할지 구성.  defaut: 'epoch'
                             # iter시 그 횟수에 따라 몇 epoch가 진행될지 자동으로 계산됨
                             # iter시 장점: 학습 시간을 좀 더 세밀하게 조정 가능
                             

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)