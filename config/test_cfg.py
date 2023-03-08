_base_ = [
    "utils/utils.py"
]


img_scale = (3000, 2000)     # expected resizing image shape (1280, 720)  width, height

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
model_path = None
device = 'cuda:0'

data = dict(
    data_root = "for_test/images",  # TODO
    batch_size = 2
)


show_score_thr = 0.7

evaluation = dict(metric=['bbox', 'segm'])      # choise in ['bbox', 'segm']


get_board_info = True
