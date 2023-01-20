img_scale = (1333, 800)     # expected resizing image shape (1280, 720)  width, height

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),                                     
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),                # flip_ratio = [0.3, 0.5, 0.2], direction = ['horizontal', 'vertical', 'diagonal']
                                                            # apply horizontal flip to 0.3% of the total,
                                                            # apply vertical flip to 0.5% of the total
                                                            # apply diagonal flip to 0.2% of the total
                                                            # kept as original to 0.1% of the total
    dict(type='Normalize', **img_norm_cfg),                
    dict(type='Pad', size_divisor=32),      # padding up to size where width and height are multiples of `size_divisor`         
    dict(type='DefaultFormatBundle'),                      
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),   # 
]

val_pipeline = [
    dict(type='LoadImageFromFile'),                                     
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(type='RandomFlip', flip_ratio=0.5),               
    # dict(type='Normalize', **img_norm_cfg),                
    dict(type='DefaultFormatBundle'),                      
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),  
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




data_root = "test_dataset"




data = dict(
    api = "COCO",
    samples_per_gpu=1,  # batch_size
    workers_per_gpu=1,  # set 0 when running by katib or in pod (not have enough shared memory) 
    train=dict(
        data_root = data_root,
        ann_file= 'train_dataset.json',
        img_prefix= "",          
        pipeline=train_pipeline
        ),
    val=dict(      
        data_root = data_root, 
        ann_file= 'val_dataset.json',
        img_prefix= "",          
        pipeline=val_pipeline,   
        batch_size = 4
        )
    )

models_dir = "models"
model_name = "last.pth"
model_path = models_dir + "/" + model_name


