_base_ = [
    'swin_solov2/solov2_r50_fpn_1x_coco.py',
    'swin_solov2/dataset_config.py',
    'swin_solov2/schedule_1x.py',
    './default_runtime.py'
]

work_dir = 'work_dir'
mode = 'train'


output = 'result.pkl'
show_dir = 'result_images'

device = 'cuda:0'

get_result_ann = False



pretrained = None
# model settings
model = dict(
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
        init_cfg=dict(type='Pretrained', checkpoint=None)),       
    mask_head=dict(
        mask_feature_head=dict(conv_cfg=dict(type='DCNv2')),
        dcn_cfg=dict(type='DCNv2'),
        dcn_apply_to_all_conv=True))

lr_config = dict(warmup_iters=1000, step=[8, 11])
runner = dict(max_epochs=3)