import  torch

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
        losses = self(**data)
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