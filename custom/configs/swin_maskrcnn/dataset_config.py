img_scale = (1333, 800)     # resizing으로 기대하는 image의 shape (1280, 720)  width, height

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),                                     
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),                # flip_ratio = [0.3, 0.5, 0.2], direction = ['horizontal', 'vertical', 'diagonal']
                                                            # 전체 중 0.3%확률로 horizontal flip적용,
                                                            # 전체 중 0.5%확률로 vertical flip적용,
                                                            # 전체 중 0.2%확률로 diagonal flip적용,         
                                                            # 0.1확률로 flip적용 안함
    dict(type='Normalize', **img_norm_cfg),                
    dict(type='Pad', size_divisor=32),      # width, height가 각각 size_divisor의 배수가 되는 size까지 padding         
    dict(type='DefaultFormatBundle'),                      
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),   # 최종 학습에 사용할 data
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

data_root = "dataset"

data = dict(
    samples_per_gpu=2,  # batch_size
    workers_per_gpu=1, 
    train=dict(
        data_root = data_root,
        ann_file= 'train_dataset.json',
        img_prefix= "",          
        pipeline=train_pipeline),
    val=dict(
        data_root = data_root,
        ann_file= 'val_dataset.json',
        img_prefix= "",          
        pipeline=val_pipeline),
    test=dict(
        data_root = data_root,
        batch_size = 10,
        ann_file= None,                                                          # work_dir/model_dir/dataset.json
        img_prefix="",                 # test할 image의 dir        
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])      # choise in ['bbox', 'segm']

