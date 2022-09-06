# dataset settings
mode = "train"


img_scale = (1333, 800)     # (720, 1280)  height, width

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [          # 순서 중요
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),           
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'CustomDataset'        


test_images_dir_path = "test_images"    # TODO

get_result_ann = False

data = dict(
    samples_per_gpu=1,  # batch_size
    workers_per_gpu=1, 
    train=dict(
        type=dataset_type,
        ann_file= None,                  # recoded dataset path     set in train component code
        img_prefix= None,               # org image dir path
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=None,     
        img_prefix= None,               
        pipeline=val_pipeline),
    test=dict(
        batch_size = 10,
        type=dataset_type,
        ann_file= None,                                   
        img_prefix= test_images_dir_path,                 # test할 image의 dir        
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])      # choise in ['bbox', 'segm']


