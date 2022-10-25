# model settings
model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(              # RPNHead
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(      # AnchorGenerator
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(        # DeltaXYWHBBoxCoder
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict( use_sigmoid=True, loss_weight=1.0),   # CrossEntropyLoss
        loss_bbox=dict(loss_weight=1.0)),       # L1Loss
    roi_head=dict(      # RoIHead
        bbox_roi_extractor=dict(    # SingleRoIExtractor
            roi_layer=dict(         # RoIAlign
                output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(             # Shared2FCBBoxHead
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=6,          # dataset의 class개수. code에서 변경됨
            bbox_coder=dict(        # DeltaXYWHBBoxCoder
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(use_sigmoid=True, loss_weight=1.0),   # CrossEntropyLoss
            loss_bbox=dict(loss_weight=1.0)),                    # L1Loss
        mask_roi_extractor=dict(        # SingleRoIExtractor
            roi_layer=dict(output_size=14, sampling_ratio=0),   # RoIAlign
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(     # FCNMaskHead
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=6,          # dataset의 class개수. code에서 변경됨
            loss_mask=dict(use_mask=True, loss_weight=1.0))),   # CrossEntropyLoss
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
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
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