import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from base_module import BaseModule, ModuleList
from basic_module import ConvModule, ConvTranspose2d, Conv2d
from loss import CrossEntropyLoss



def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,
                cfg):
    """Compute mask target for positive proposals in multiple images.

    Args:
        pos_proposals_list (list[Tensor]): Positive proposals in multiple
            images.
        pos_assigned_gt_inds_list (list[Tensor]): Assigned GT indices for each
            positive proposals.
        gt_masks_list (list[:obj:`BaseInstanceMasks`]): Ground truth masks of
            each image.
        cfg (dict): Config dict that specifies the mask size.

    Returns:
        list[Tensor]: Mask target of each image.

    Example:
        >>> import mmcv
        >>> import mmdet
        >>> from mmdet.core.mask import BitmapMasks
        >>> from mmdet.core.mask.mask_target import *
        >>> H, W = 17, 18
        >>> cfg = mmcv.Config({'mask_size': (13, 14)})
        >>> rng = np.random.RandomState(0)
        >>> # Positive proposals (tl_x, tl_y, br_x, br_y) for each image
        >>> pos_proposals_list = [
        >>>     torch.Tensor([
        >>>         [ 7.2425,  5.5929, 13.9414, 14.9541],
        >>>         [ 7.3241,  3.6170, 16.3850, 15.3102],
        >>>     ]),
        >>>     torch.Tensor([
        >>>         [ 4.8448, 6.4010, 7.0314, 9.7681],
        >>>         [ 5.9790, 2.6989, 7.4416, 4.8580],
        >>>         [ 0.0000, 0.0000, 0.1398, 9.8232],
        >>>     ]),
        >>> ]
        >>> # Corresponding class index for each proposal for each image
        >>> pos_assigned_gt_inds_list = [
        >>>     torch.LongTensor([7, 0]),
        >>>     torch.LongTensor([5, 4, 1]),
        >>> ]
        >>> # Ground truth mask for each true object for each image
        >>> gt_masks_list = [
        >>>     BitmapMasks(rng.rand(8, H, W), height=H, width=W),
        >>>     BitmapMasks(rng.rand(6, H, W), height=H, width=W),
        >>> ]
        >>> mask_targets = mask_target(
        >>>     pos_proposals_list, pos_assigned_gt_inds_list,
        >>>     gt_masks_list, cfg)
        >>> assert mask_targets.shape == (5,) + cfg['mask_size']
    """
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(mask_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    mask_targets = list(mask_targets)
    if len(mask_targets) > 0:
        mask_targets = torch.cat(mask_targets)
    return mask_targets


def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
    """Compute mask target for each positive proposal in the image.

    Args:
        pos_proposals (Tensor): Positive proposals.
        pos_assigned_gt_inds (Tensor): Assigned GT inds of positive proposals.
        gt_masks (:obj:`BaseInstanceMasks`): GT masks in the format of Bitmap
            or Polygon.
        cfg (dict): Config dict that indicate the mask size.

    Returns:
        Tensor: Mask target of each positive proposals in the image.

    Example:
        >>> import mmcv
        >>> import mmdet
        >>> from mmdet.core.mask import BitmapMasks
        >>> from mmdet.core.mask.mask_target import *  # NOQA
        >>> H, W = 32, 32
        >>> cfg = mmcv.Config({'mask_size': (7, 11)})
        >>> rng = np.random.RandomState(0)
        >>> # Masks for each ground truth box (relative to the image)
        >>> gt_masks_data = rng.rand(3, H, W)
        >>> gt_masks = BitmapMasks(gt_masks_data, height=H, width=W)
        >>> # Predicted positive boxes in one image
        >>> pos_proposals = torch.FloatTensor([
        >>>     [ 16.2,   5.5, 19.9, 20.9],
        >>>     [ 17.3,  13.6, 19.3, 19.3],
        >>>     [ 14.8,  16.4, 17.0, 23.7],
        >>>     [  0.0,   0.0, 16.0, 16.0],
        >>>     [  4.0,   0.0, 20.0, 16.0],
        >>> ])
        >>> # For each predicted proposal, its assignment to a gt mask
        >>> pos_assigned_gt_inds = torch.LongTensor([0, 1, 2, 1, 1])
        >>> mask_targets = mask_target_single(
        >>>     pos_proposals, pos_assigned_gt_inds, gt_masks, cfg)
        >>> assert mask_targets.shape == (5,) + cfg['mask_size']
    """
    device = pos_proposals.device
    mask_size = _pair(cfg.mask_size)
    binarize = not cfg.get('soft_mask_target', False)
    num_pos = pos_proposals.size(0)
    if num_pos > 0:
        proposals_np = pos_proposals.cpu().numpy()
        maxh, maxw = gt_masks.height, gt_masks.width
        proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
        proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()

        mask_targets = gt_masks.crop_and_resize(
            proposals_np,
            mask_size,
            device=device,
            inds=pos_assigned_gt_inds,
            binarize=binarize).to_ndarray()

        mask_targets = torch.from_numpy(mask_targets).float().to(device)
    else:
        mask_targets = pos_proposals.new_zeros((0, ) + mask_size)

    return mask_targets



class FCNMaskHead(BaseModule):
    def __init__(self,
                 num_convs=4,
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 conv_out_channels=256,
                 num_classes=80,
                 class_agnostic=False,
                 scale_factor=2,
                 predictor_cfg=dict(type='Conv'),
                 loss_mask=dict(use_mask=True, loss_weight=1.0),        # CrossEntropyLoss
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(FCNMaskHead, self).__init__(init_cfg)
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.scale_factor = scale_factor
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.predictor_cfg = predictor_cfg
        self.loss_mask = CrossEntropyLoss(**loss_mask)
        
        self.convs = ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding))
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        
        upsample_cfg_ = {}
        upsample_cfg_.update(
            in_channels=upsample_in_channels,
            out_channels=self.conv_out_channels,
            kernel_size=self.scale_factor,
            stride=self.scale_factor)
        self.upsample = ConvTranspose2d(**upsample_cfg_)
        
        out_channels = 1 if self.class_agnostic else self.num_classes
        self.conv_logits = Conv2d(self.conv_out_channels, out_channels, 1)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.debug_imgs = None
        
        self.fp16_enabled = False

    # @auto_fp16()
    def forward(self, x):
        # x.shape = [tmp, 256, win_size*2, win_size*2]
        for conv in self.convs:
            x = conv(x)  # [tmp, 256, win_size*4, win_size*4]
        if self.upsample is not None:   # True
            x = self.upsample(x)
            x = self.relu(x)
       
        mask_pred = self.conv_logits(x)     # [tmp, 6, win_size*4, win_size*4]d
        return mask_pred


    def get_targets(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets
    
    
    def init_weights(self):
        super(FCNMaskHead, self).init_weights()
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            elif hasattr(m, 'weight') and hasattr(m, 'bias'):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
                
    
    # @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels):
        """
        Example:
            >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
            >>> N = 7  # N = number of extracted ROIs
            >>> C, H, W = 11, 32, 32
            >>> # Create example instance of FCN Mask Head.
            >>> # There are lots of variations depending on the configuration
            >>> self = FCNMaskHead(num_classes=C, num_convs=1)
            >>> inputs = torch.rand(N, self.in_channels, H, W)
            >>> mask_pred = self.forward(inputs)
            >>> sf = self.scale_factor
            >>> labels = torch.randint(0, C, size=(N,))
            >>> # With the default properties the mask targets should indicate
            >>> # a (potentially soft) single-class label
            >>> mask_targets = torch.rand(N, H * sf, W * sf)
            >>> loss = self.loss(mask_pred, mask_targets, labels)
            >>> print('loss = {!r}'.format(loss))
        """
        loss = dict()
       
        if mask_pred.size(0) == 0:
            loss_mask = mask_pred.sum()
        else:   # True
            if self.class_agnostic:
                loss_mask = self.loss_mask(mask_pred, mask_targets,
                                           torch.zeros_like(labels))
            else:       # True
                print("@@@@")
                loss_mask = self.loss_mask(mask_pred, mask_targets, labels, tmp = True)
                print(f"loss_mask : {loss_mask}")
                exit()
            
        loss['loss_mask'] = loss_mask
        return loss
                