


pretrained ='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'

# model settings
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
        pm_kernel_size = 2,
        pm_dilation = 1,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,       # add backbone name before layer name when run weight initalization
                                    # if True : patch_embed.projection.weight >> backbone.patch_embed.projection.weight
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),       # fine tuning
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),    
    rpn_head=dict(              # RPNHead
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(  # AnchorGenerator
            type = 'AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(        # DeltaXYWHBBoxCoder
            type = 'DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(          # CrossEntropyLoss
            type = 'CrossEntropyLoss',
            use_sigmoid=True, 
            loss_weight=1.0),   
        loss_bbox=dict(         # L1Loss
            type = 'L1Loss',
            loss_weight=1.0)),      
    roi_head=dict(      # StandardRoIHead
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign',
                output_size=7, 
                sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=6,          # number of class type in dataset. will be fixed in the code.
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True, 
                loss_weight=1.0),  
            loss_bbox=dict(
                type='L1Loss',
                loss_weight=1.0)),                  
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlign',
                output_size=14, 
                sampling_ratio=0),   
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=6,          # number of class type in dataset. will be fixed in the code.
            loss_mask=dict(
                type='CrossEntropyLoss',
                use_mask=True, 
                loss_weight=1.0))),                                         
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(              # MaxIoUAssigner
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(               # RandomSampler
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,       # number of anchor to be selected for each level
            max_per_img=1000,   # image당 proposals되는 image의 최대 개수(넘더라도 이 선에서 자른다. 값 올리면 성능 올라가나?)
            nms=dict(type='nms', iou_threshold=0.7),        # 추가 가능: max_num (int): maximum number of boxes after NMS.
                                                            # score_threshold = 0 or 0 < fload < 1   : score threshold for NMS.
            min_bbox_size=0),   # 이 값이 크면 작은 object는 detection불가능하지만 성능 효율↑
        rcnn=dict(
            assigner=dict(              # MaxIoUAssigner
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(               # RandomSampler
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))

