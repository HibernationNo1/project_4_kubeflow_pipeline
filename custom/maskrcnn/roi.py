import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

from base_module import BaseModule, ModuleList
from maskrcnn.coder import DeltaXYWHBBoxCoder
from loss import CrossEntropyLoss, L1Loss
from assigner import MaxIoUAssigner
from maskrcnn.rpn import ConvModule
from utils.sampler import RandomSampler

if torch.__version__ == 'parrots':
    TORCH_VERSION = torch.__version__
else:
    # torch.__version__ could be 1.3.1+cu92, we only need the first two
    # for comparison
    TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])
    
    
# StandardRoIHead
class RoIHead(BaseModule):  
    def __init__(self,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 mask_roi_extractor=dict(                               # SingleRoIExtractor
                    roi_layer=dict(output_size=14, sampling_ratio=0),   # RoIAlign
                    out_channels=256,
                    featmap_strides=[4, 8, 16, 32]),
                 mask_head=None,        # FCNMaskHead
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(RoIHead, self).__init__(init_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        if bbox_head is not None: self.init_bbox_head(bbox_roi_extractor, bbox_head)        
        if mask_head is not None: self.init_mask_head(mask_roi_extractor, mask_head)
        self.init_assigner_sampler()
    
    @property
    def with_bbox(self):
        """bool: whether the RoI head contains a `bbox_head`"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_mask(self):
        """bool: whether the RoI head contains a `mask_head`"""
        return hasattr(self, 'mask_head') and self.mask_head is not None

    @property
    def with_shared_head(self):
        """bool: whether the RoI head contains a `shared_head`"""
        return hasattr(self, 'shared_head') and self.shared_head is not None
    
       
    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            
            self.bbox_assigner = MaxIoUAssigner(**self.train_cfg.assigner)        # 여기서부터
            self.bbox_sampler = RandomSampler(**self.train_cfg.sampler)
            
    
    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        
        self.bbox_roi_extractor = SingleRoIExtractor(**bbox_roi_extractor)
        self.bbox_head = Shared2FCBBoxHead(**bbox_head)
        
        

        
    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = SingleRoIExtractor(**mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
   
        self.mask_head = FCNMaskHead(**mask_head)
        
        
        
    # TODO


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
        
    def init_weights(self):
        super(FCNMaskHead, self).init_weights()
        for m in [self.upsample, self.conv_logits]:
            if m is None:
                continue
            elif hasattr(m, 'weight') and hasattr(m, 'bias'):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
                

class Conv2d(nn.Conv2d):
    def forward(self, x):
        if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 4)):
            out_shape = [x.shape[0], self.out_channels]
            for i, k, p, s, d in zip(x.shape[-2:], self.kernel_size,
                                     self.padding, self.stride, self.dilation):
                o = (i + 2 * p - (d * (k - 1) + 1)) // s + 1
                out_shape.append(o)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                # produce dummy gradient to avoid DDP warning.
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super().forward(x)


class NewEmptyTensorOp(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return NewEmptyTensorOp.apply(grad, shape), None
    
    
def obsolete_torch_version(torch_version, version_threshold):
    print(f"version_threshold : {version_threshold}")
    return torch_version == 'parrots' or torch_version <= version_threshold
         
class ConvTranspose2d(nn.ConvTranspose2d):
    def forward(self, x):
        if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 4)):
            out_shape = [x.shape[0], self.out_channels]
            for i, k, p, s, d, op in zip(x.shape[-2:], self.kernel_size,
                                         self.padding, self.stride,
                                         self.dilation, self.output_padding):
                out_shape.append((i - 1) * s - 2 * p + (d * (k - 1) + 1) + op)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                # produce dummy gradient to avoid DDP warning.
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty
        return super().forward(x)
    

class Shared2FCBBoxHead(BaseModule):
    def __init__(self, 
                 fc_out_channels=1024,
                 num_shared_convs=0,
                 num_shared_fcs=2,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 with_avg_pool=False,
                 with_cls=True,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=80,
                 bbox_coder=dict(       # DeltaXYWHBBoxCoder
                     clip_border=True,
                     target_means=[0., 0., 0., 0.],
                     target_stds=[0.1, 0.1, 0.2, 0.2]),
                 reg_class_agnostic=False,
                 reg_decoded_bbox=False,
                 loss_cls=dict(         # CrossEntropyLoss
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_bbox=dict(        # default: SmoothL1Loss
                     beta=1.0, loss_weight=1.0)):
        super(Shared2FCBBoxHead, self).__init__(init_cfg) 
        
        """Simplest RoI head, with only two fc layers for classification and
            regression respectively."""
        assert with_cls or with_reg     # 두개 다 true로 고정, 지우기
        self.with_avg_pool = with_avg_pool
        self.with_cls = with_cls
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.reg_class_agnostic = reg_class_agnostic
        self.reg_decoded_bbox = reg_decoded_bbox

        self.bbox_coder = DeltaXYWHBBoxCoder(**bbox_coder)
        self.loss_cls = CrossEntropyLoss(**loss_cls)
        self.loss_bbox = L1Loss(**loss_bbox)
        
        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area
                            
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:     assert num_shared_fcs == 0
        if not self.with_cls:  assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:   assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        
        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim
        

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)
        
        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area
                
        self.relu = nn.ReLU(inplace=True)
        
        # construct fc_cls and fc_reg
        if self.with_cls:
            cls_channels = num_classes + 1
            self.fc_cls = nn.Linear(in_features=in_channels,
                                    out_features=cls_channels)
        
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear( in_features=in_channels, out_features=out_dim_reg)
        
        self.debug_imgs = None
        if init_cfg is None:    # model initialize시 사용할 임의의 config
            self.init_cfg = []
            if self.with_cls:
                self.init_cfg += [
                    dict(
                        type='Normal', std=0.01, override=dict(name='fc_cls'))
                ]
            if self.with_reg:
                self.init_cfg += [
                    dict(
                        type='Normal', std=0.001, override=dict(name='fc_reg'))
                ]
                
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]

        
        
    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim
            
class SingleRoIExtractor(BaseModule):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (dict): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (List[int]): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0. Default: 56.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 roi_layer,
                 out_channels,
                 featmap_strides,
                 finest_scale=56,
                 init_cfg=None):
        super(SingleRoIExtractor, self).__init__(init_cfg)
        cfg = roi_layer.copy()
        self.roi_layers = nn.ModuleList([RoIAlign(spatial_scale=1 / s, **cfg) for s in featmap_strides])
        
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.finest_scale = finest_scale
    
    @property
    def num_inputs(self):
        """int: Number of input feature maps."""
        return len(self.featmap_strides)
    
    def roi_rescale(self, rois, scale_factor):
        """Scale RoI coordinates by scale factor.

        Args:
            rois (torch.Tensor): RoI (Region of Interest), shape (n, 5)
            scale_factor (float): Scale factor that RoI will be multiplied by.

        Returns:
            torch.Tensor: Scaled RoI.
        """

        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1]
        h = rois[:, 4] - rois[:, 2]
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5
        x2 = cx + new_w * 0.5
        y1 = cy - new_h * 0.5
        y2 = cy + new_h * 0.5
        new_rois = torch.stack((rois[:, 0], x1, y1, x2, y2), dim=-1)
        return new_rois

    

class RoIAlign(nn.Module):
    """RoI align pooling layer.

    Args:
        output_size (tuple): h, w
        spatial_scale (float): scale the input boxes by this number
        sampling_ratio (int): number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
        pool_mode (str, 'avg' or 'max'): pooling mode in each bin.
        aligned (bool): if False, use the legacy implementation in
            MMDetection. If True, align the results more perfectly.
        use_torchvision (bool): whether to use roi_align from torchvision.

    Note:
        The implementation of RoIAlign when aligned=True is modified from
        https://github.com/facebookresearch/detectron2/

        The meaning of aligned=True:

        Given a continuous coordinate c, its two neighboring pixel
        indices (in our pixel model) are computed by floor(c - 0.5) and
        ceil(c - 0.5). For example, c=1.3 has pixel neighbors with discrete
        indices [0] and [1] (which are sampled from the underlying signal
        at continuous coordinates 0.5 and 1.5). But the original roi_align
        (aligned=False) does not subtract the 0.5 when computing
        neighboring pixel indices and therefore it uses pixels with a
        slightly incorrect alignment (relative to our pixel model) when
        performing bilinear interpolation.

        With `aligned=True`,
        we first appropriately scale the ROI and then shift it by -0.5
        prior to calling roi_align. This produces the correct neighbors;

        The difference does not make a difference to the model's
        performance if ROIAlign is used together with conv layers.
    """
    
    def __init__(self,
                 output_size,
                 spatial_scale=1.0,
                 sampling_ratio=0,
                 pool_mode='avg',
                 aligned=True,
                 use_torchvision=False):
        super().__init__()

        self.output_size = _pair(output_size)
        self.spatial_scale = float(spatial_scale)
        self.sampling_ratio = int(sampling_ratio)
        self.pool_mode = pool_mode
        self.aligned = aligned
        self.use_torchvision = use_torchvision