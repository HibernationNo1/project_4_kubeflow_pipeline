import  torch

from builder import build_backbone, MODELS
from modules.base_module import BaseModule
from modules.maskrcnn.rpn import RPNHead
from modules.neck.fpn import FPN
from modules.maskrcnn.roi import RoIHead

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
        self.fp16_enabled = False       # TODO delete
        # mmcv > runner > ffp16_utils.py > def auto_fp16        
        
        self.backbone = build_backbone(backbone)
        # from mmdet_taeuk4958.models.backbones.swin import SwinTransformer as SwinTransformer_   ####
        # if backbone.get("type", None) is not None:
        #     _ = backbone.pop("type")
        # self.backbone = SwinTransformer_(**backbone)
        
        if neck.get("type", None) is not None: _ = neck.pop("type")
        self.neck = FPN(**neck)
        # from mmdet_taeuk4958.models.necks.fpn import FPN as FPN_####
        # if neck.get("type", None) is not None:
        #     _ = neck.pop("type")
            
        
        # build_rpn_head : RPNHead
        # rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
        # rpn_head_cfg = rpn_head.copy()
        # rpn_head_cfg.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
        # self.rpn_head = RPNHead(**rpn_head_cfg)
        rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
        rpn_head_cfg = rpn_head.copy()
        rpn_train_cfg = {'assigner': {'type': 'MaxIoUAssigner', 
                                      'pos_iou_thr': rpn_train_cfg.assigner.pos_iou_thr, 
                                      'neg_iou_thr': rpn_train_cfg.assigner.neg_iou_thr, 
                                      'min_pos_iou': rpn_train_cfg.assigner.min_pos_iou, 
                                      'match_low_quality': rpn_train_cfg.assigner.match_low_quality, 
                                      'ignore_iof_thr': rpn_train_cfg.assigner.ignore_iof_thr}, 
                         'sampler': {'type': 'RandomSampler', 
                                     'num': rpn_train_cfg.sampler.num, 
                                     'pos_fraction': rpn_train_cfg.sampler.pos_fraction, 
                                     'neg_pos_ub': rpn_train_cfg.sampler.neg_pos_ub, 
                                     'add_gt_as_proposals': rpn_train_cfg.sampler.add_gt_as_proposals}, 
                         'allowed_border': rpn_train_cfg.allowed_border, 
                         'pos_weight': rpn_train_cfg.pos_weight, 
                         'debug': rpn_train_cfg.debug}
        rpn_head_cfg.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
        from mmdet_taeuk4958.models.dense_heads.rpn_head import RPNHead as RPNHead_
        from utils.config import Config
        rpn_head_cfg = Config(rpn_head_cfg) 
        self.rpn_head = RPNHead_(**rpn_head_cfg)    
        
        # rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
        # roi_head.update(train_cfg=rcnn_train_cfg)
        # roi_head.update(test_cfg=test_cfg.rcnn)
        # self.roi_head = RoIHead(**roi_head)   
        rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
        rcnn_train_cfg = {'assigner': {'type': 'MaxIoUAssigner', 
                                       'pos_iou_thr': train_cfg.rcnn.assigner.pos_iou_thr, 
                                       'neg_iou_thr': train_cfg.rcnn.assigner.neg_iou_thr, 
                                       'min_pos_iou': train_cfg.rcnn.assigner.min_pos_iou, 
                                       'match_low_quality': train_cfg.rcnn.assigner.match_low_quality, 
                                       'ignore_iof_thr': train_cfg.rcnn.assigner.ignore_iof_thr}, 
                          'sampler': {'type': 'RandomSampler', 
                                      'num': train_cfg.rcnn.sampler.num, 
                                      'pos_fraction': train_cfg.rcnn.sampler.pos_fraction, 
                                      'neg_pos_ub': train_cfg.rcnn.sampler.neg_pos_ub, 
                                      'add_gt_as_proposals': train_cfg.rcnn.sampler.add_gt_as_proposals}, 
                          'mask_size': train_cfg.rcnn.mask_size, 
                          'pos_weight': train_cfg.rcnn.pos_weight, 
                          'debug': train_cfg.rcnn.debug}
        roi_head.update(train_cfg=rcnn_train_cfg)
        roi_head.update(test_cfg=test_cfg.rcnn)
        from mmdet_taeuk4958.models.roi_heads.standard_roi_head import StandardRoIHead as StandardRoIHead_
        from utils.config import Config
        roi_head = Config(roi_head)  
        self.roi_head = StandardRoIHead_(**roi_head)     

    
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
            return self.forward_test(img, img_metas, **kwargs)    
    
    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')
        
        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')
        
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])
        
        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]

            
            
            assert self.with_bbox, 'Bbox head must be implemented.'
            
            img, img_meta = imgs[0], img_metas[0]
            x = self.backbone(img)
            if self.with_neck:
                x = self.neck(x)
            
            if kwargs.get("proposals", None) is None:
                # len: batch_size
                proposal_list = self.rpn_head.simple_test_rpn(x, img_meta)
            else:
                proposal_list = kwargs['proposals']
        
            # len: batch_size
            # result[n].size: (6, 6)
            results = self.roi_head.simple_test(x, proposal_list, img_meta, rescale=kwargs.get("rescale", False))           
            
            return results
        
        else:   # TODO
            pass
            # assert imgs[0].size(0) == 1, 'aug test does not support ' \
            #                              'inference with batch size ' \
            #                              f'{imgs[0].size(0)}'
            # # TODO: support test augmentation for predefined proposals
            # assert 'proposals' not in kwargs
            # return self.aug_test(imgs, img_metas, **kwargs)
            
        
    # gt_bboxes_ignore = None
    def forward_train(self, img, img_metas, gt_bboxes, gt_labels, gt_masks=None, proposals=None,
                      **kwargs):
        """
            len(gt_bboxe) = batch_size
                gt_bboxe: [num_gts, 4] in [x_min, y_min, x_max, y_max]x = self.neck
        """

        # img: [B=2, C=3, H=768, W=1344]
        x = self.backbone(img)
        # type(x): list,        len(x) == cfg.model.backbone.depths
        # each channel of elements must be equal to `cfg.model.neck.in_channels`과
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
        
        ## compute rpn loss and get proposal boxes
        # type:dict, keys = ['loss_cls', 'loss_bbox'],      each len = num_levels, value: tensor(float)
        # len(proposal_list) = batch_size
        # proposal: [proposal_cfg.max_per_img, 5],    5: [x_min, y_min, x_max, y_max, score]
        # rpn_losses, proposal_list = self.rpn_head.forward_train(x, img_metas, gt_bboxes, proposal_cfg, 
        #                                                         **kwargs)
        
        rpn_losses, proposal_list = self.rpn_head.forward_train(x,
                                                                img_metas,
                                                                gt_bboxes,
                                                                gt_labels=None,
                                                                gt_bboxes_ignore=None,
                                                                proposal_cfg=proposal_cfg,
                                                                **kwargs)
        
        losses.update(rpn_losses)
     
        # roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
        #                                          gt_bboxes, gt_labels, gt_masks,
        #                                          **kwargs)
        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels, None, gt_masks,
                                                 **kwargs)
        
        losses.update(roi_losses)

      
        return losses 
    
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
    
        
        # run `forward()` that all modules have 
        losses = self(**data)  
                
        loss, log_vars = self._parse_losses(losses)
        
        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

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

        # total loss
        loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
        
        log_vars['loss'] = loss         
            
        for loss_name, loss_value in log_vars.items():
            log_vars[loss_name] = loss_value.item()
            
        return loss, log_vars