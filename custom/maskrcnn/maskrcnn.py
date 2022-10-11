from builder import build_backbone, MODELS
from base_module import BaseModule
from maskrcnn.rpn import RPNHead
from maskrcnn.fpn import FPN
from maskrcnn.roi import RoIHead

@MODELS.register_module()
class MaskRCNN(BaseModule):
    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        
        super(MaskRCNN, self).__init__(init_cfg)
        self.fp16_enabled = False
        
        self.backbone = build_backbone(backbone)
        
        
        self.neck = FPN(in_channels= neck.in_channels, out_channels= neck.out_channels, num_outs= neck.num_outs)
        
        # build_rpn_head : RPNHead
        rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
        rpn_head_cfg = rpn_head.copy()
        rpn_head_cfg.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
        self.rpn_head = RPNHead(**rpn_head_cfg)
        
        
        rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
        roi_head.update(train_cfg=rcnn_train_cfg)
        roi_head.update(test_cfg=test_cfg.rcnn)
        self.roi_head = RoIHead(**roi_head)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
