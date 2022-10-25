import  torch

from builder import build_backbone, MODELS
from base_module import BaseModule
from models.maskrcnn.rpn import RPNHead
from models.neck.fpn import FPN
from models.maskrcnn.roi import RoIHead

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
        self.fp16_enabled = False       # TODO fp16으로 변환하여 학습 진행해보기
        # mmcv > runner > ffp16_utils.py > def auto_fp16
        
        
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

    # @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.
        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            return self.forward_test(img, img_metas, **kwargs)  #  TODO
        
    
    def forward_train(self, img, img_metas, gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None, gt_masks=None, proposals=None,
                      **kwargs):
        # img: [B=2, C=3, H=768, W=1344]
        x = self.backbone(img)
        # type(x): list,        len(x) == cfg.model.backbone.depths
        # 각 elements의 channel은 cfg.model.neck.in_channels과 동일해야 한다
        # x[n]: [B, Cn, H/n, W/n],     Cn == cfg.model.neck.in_channels,    n = [4, 8, 16, 32]

        # [2, 96, 192, 336]
        # [2, 192, 96, 168]
        # [2, 384, 48, 84]
        # [2, 768, 24, 42]
        
        x = self.neck(x)
        # [2, 256, 192, 336]
        # [2, 256, 96, 168]
        # [2, 256, 48, 84]
        # [2, 256, 24, 42]
        # [2, 256, 12, 21]      # max_pool2d
        
        losses = dict()
        # RPN forward and loss
        proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
        rpn_losses, proposal_list = self.rpn_head.forward_train(x, img_metas, gt_bboxes,
                                                                gt_bboxes_ignore=gt_bboxes_ignore, proposal_cfg=proposal_cfg,
                                                                **kwargs)
        
        
    
    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head.with_shared_head
    
    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_bbox)
                or (hasattr(self, 'bbox_head') and self.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self, 'roi_head') and self.roi_head.with_mask)
                or (hasattr(self, 'mask_head') and self.mask_head is not None))
        
    def train_step(self, data, optimizer):  # TODO : optimizer 어따씀?
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """

        # type(data): dict,     ['img_metas', 'img', 'gt_bboxes', 'gt_labels', 'gt_masks'] 
        #      img.shape = (batch_size, channel, height, width)
        #      else, len(key) == batch_size
    
        
        # self.에 포함된 모든 module의 forward()를 실행
        losses = self(**data)       
        exit()
        
        loss, log_vars = self._parse_losses(losses)
        
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs
    
    
    def _parse_losses(self, losses):    
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = dict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')
            
        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)
        
        log_vars['loss'] = loss
        
        for loss_name, loss_value in log_vars.items():
            log_vars[loss_name] = loss_value.item()
        
        return loss, log_vars