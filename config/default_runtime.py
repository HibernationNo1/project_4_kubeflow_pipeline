pipeline = dict(
    pipeline_name = 'train',
    pipeline_version = "0.1"
)

dataset_name = "dataset"

gs = dict(
    client_secrets = "client_secrets.json",
    ann_bucket_name = "dataset_tesuk4958",
    recoded_dataset_bucket_name = "pipeline_taeuk4958",
    model_bucket_name = 'pipeline_models_taeuk4958',
    recoded_dataset_version = "0.1",
    train_file_name = 'train_dataset.json',
    val_file_name = 'val_dataset.json',
    imgs_dir = f"{dataset_name}/images",
    anns_dir = f"{dataset_name}/anns",
    anns_config_path = f"{dataset_name}/config.json",
    
    )


checkpoint_config = dict(interval=1)
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
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)