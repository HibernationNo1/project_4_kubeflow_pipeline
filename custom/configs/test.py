
models_dir = "models"
model_name = "last.pth"

model_path = models_dir + "/" + model_name



img_scale = (1333, 800)     # expected resizing image shape (1280, 720)  width, height

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

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

test_result = "result/test"
eval_result = "result/evel"
data_root = "dataset"
test_data = "test"

data = dict(
    samples_per_gpu=2,  # batch_size
    workers_per_gpu=1, 
    test=dict(
        data_root = data_root + "/" + test_data,
        batch_size = 10,
        ann_file= None,                                                          # work_dir/model_dir/dataset.json
        img_prefix="",                 # test할 image의 dir        
        pipeline=test_pipeline))

workflow = [('test', None)]
