


train_pipeline = [
    dict(type='LoadImageFromFile'),                                     
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', 
         img_scale=(1280, 800),        # expected resizing image shape (1280, 720)  width, height      1333, 800
         keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),                # flip_ratio = [0.3, 0.5, 0.2], direction = ['horizontal', 'vertical', 'diagonal']
                                                            # apply horizontal flip to 0.3% of the total,
                                                            # apply vertical flip to 0.5% of the total
                                                            # apply diagonal flip to 0.2% of the total
                                                            # kept as original to 0.1% of the total
    dict(type='Normalize', 
         mean = [123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375],
         to_rgb=True),                
    dict(type='Pad', size_divisor=32),      # padding up to size where width and height are multiples of `size_divisor`         
    dict(type='DefaultFormatBundle'),                      
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),   # 
]


val_infer_pipeline = [              # infernce for during validation
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1200, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', 
                 mean = [123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 to_rgb=True), 
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]


val_pipeline = [
    dict(type='LoadImageFromFile'),                                     
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),      
    dict(type='DefaultFormatBundle'),                      
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),  
]


data_root = "test_dataset"

data = dict(
    api = "COCO",
    samples_per_gpu=1,  # batch_size
    workers_per_gpu=0,  # set 0 when running by katib or in pod (not have enough shared memory) 
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



