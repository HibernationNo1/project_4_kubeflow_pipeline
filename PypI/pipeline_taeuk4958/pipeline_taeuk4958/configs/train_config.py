
_base_ = [
    '/opt/conda/lib/python3.7/site-packages/pipeline_taeuk4958/configs/swin_maskrcnn/mask_rcnn_r50_fpn.py',     # 'swin_maskrcnn/mask_rcnn_r50_fpn.py"
    '/opt/conda/lib/python3.7/site-packages/pipeline_taeuk4958/configs/swin_maskrcnn/dataset_config.py',     # 'swin_maskrcnn/dataset_config.py'
    '/opt/conda/lib/python3.7/site-packages/pipeline_taeuk4958/configs/swin_maskrcnn/schedule_1x.py',       # 'swin_maskrcnn/schedule_1x.py'
    '/opt/conda/lib/python3.7/site-packages/pipeline_taeuk4958/configs/default_runtime.py'        # './default_runtime.py'  
]




mode = "labelme"

pipeline = dict(
    pipeline_name = 'train',
    pipeline_version = "0.1"
)

gs = dict(
    client_secrets = "client_secrets.json",
    recoded_dataset_bucket_name = "pipeline_taeuk4958",
    )


train = dict(
    validate = False,
    finetun = True,
    model_version = '0.0.1'
)

device = 'cuda:0'

pretrained ='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),       # fine tuning
    neck=dict(in_channels=[96, 192, 384, 768]))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict(warmup_iters=1000, step=[8, 11])
runner = dict(max_epochs=3)
