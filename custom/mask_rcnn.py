
from builder import build_backbone, MODELS
from base_module import BaseModule, ModuleList
from abc import ABCMeta
import numpy as np

import torch
import torch.nn as nn

from loss import CrossEntropyLoss, L1Loss
from assigner import MaxIoUAssigner

# TODO 나중에 init으로 옮기기
def kaiming_init(module: nn.Module,     
                 a: float = 0,
                 mode: str = 'fan_out',
                 nonlinearity: str = 'relu',
                 bias: float = 0,
                 distribution: str = 'normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
        
        
def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


        
class ConvModule(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Besides, we add some additional features in this module.
    1. Automatically set `bias` of the conv layer.
    2. Spectral norm is supported.
    3. More padding modes are supported. Before PyTorch 1.5, nn.Conv2d only
    supports zero and circular padding, and we add "reflect" padding mode.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (int): 
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        padding_mode (str): If the `padding_mode` has not been supported by
            current `Conv2d` in PyTorch, we will use our own padding layer
            instead. Currently, we support ['zeros', 'circular'] with official
            implementation and ['reflect'] with our own implementation.
            Default: 'zeros'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    _abbr_ = 'conv_block'

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=0,  
                 inplace=True,
                 padding_mode='zeros',      # support ['zeros' 'circular', 'reflect'] 찾아서 공부해서 적용해보기
                 order=('conv', 'norm', 'act')):
        super().__init__()
        
        
        official_padding_mode = ['zeros', 'circular']
        self.inplace = inplace
        self.with_explicit_padding = padding_mode not in official_padding_mode
        
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {'conv', 'norm', 'act'}
        
        # reset padding to 0 for conv module
        conv_padding = padding

        # build convolution layer
        self.conv = nn.Conv2d(in_channels=in_channels,         
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=conv_padding,
                                dilation=dilation,
                                groups=groups,
                                bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        
        # build normalization layers
        # norm layer is after conv layer
        if order.index('norm') > order.index('conv'):
            norm_channels = out_channels
        else:
            norm_channels = in_channels
        norm = nn.LayerNorm(norm_channels, eps = 1e-5)
        self.norm_name = 'norm_layer' 
        
        self.add_module(self.norm_name, norm)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):        
        nonlinearity = 'relu'
        a = 0
        kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        
        
    def forward(self, x, norm=True):
        for layer in self.order:
            if layer == 'conv':
                x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
        return x




class FPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.

        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 relu_before_extra_convs=False,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(FPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level

        self.lateral_convs = ModuleList()
        self.fpn_convs = ModuleList()

        
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                kernel_size= 1,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                kernel_size= 3,
                padding=1,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
                


class DeltaXYWHBBoxCoder(metaclass=ABCMeta):
    """Delta XYWH BBox coder.

    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes bbox (x1, y1, x2, y2) into delta (dx, dy, dw, dh) and
    decodes delta (dx, dy, dw, dh) back to original bbox (x1, y1, x2, y2).

    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.
    """

    def __init__(self,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(1., 1., 1., 1.),
                 clip_border=True,
                 add_ctr_clamp=False,
                 ctr_clamp=32):
        self.means = target_means
        self.stds = target_stds
        self.clip_border = clip_border
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp

    def encode(self, bboxes, gt_bboxes):
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.

        Args:
            bboxes (torch.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (torch.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas
        """

        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == gt_bboxes.size(-1) == 4

        # 제안된 bboxes(= proposals)를 gt_bboxes로 계산(변환)하는 delta값을 계산한다.
        assert bboxes.size() == gt_bboxes.size()
        
        bboxes = bboxes.float()
        gt = gt_bboxes.float()
        px = (bboxes[..., 0] + bboxes[..., 2]) * 0.5
        py = (bboxes[..., 1] + bboxes[..., 3]) * 0.5
        pw = bboxes[..., 2] - bboxes[..., 0]
        ph = bboxes[..., 3] - bboxes[..., 1]
        
        gx = (gt[..., 0] + gt[..., 2]) * 0.5
        gy = (gt[..., 1] + gt[..., 3]) * 0.5
        gw = gt[..., 2] - gt[..., 0]
        gh = gt[..., 3] - gt[..., 1]
        
        dx = (gx - px) / pw
        dy = (gy - py) / ph
        dw = torch.log(gw / pw)
        dh = torch.log(gh / ph)
        deltas = torch.stack([dx, dy, dw, dh], dim=-1)
        means = deltas.new_tensor(means).unsqueeze(0)
        stds = deltas.new_tensor(stds).unsqueeze(0)
        encoded_bboxes = deltas.sub_(means).div_(stds)      # shape: (N, 4),  N = (dx, dy, dw, dh)
    
        return encoded_bboxes

    def decode(self,
               bboxes,
               pred_bboxes,
               max_shape=None,
               wh_ratio_clip=16 / 1000):
        """Apply transformation `pred_bboxes` to `boxes`.

        Args:
            bboxes (torch.Tensor): Basic boxes. Shape (B, N, 4) or (N, 4)
            pred_bboxes (Tensor): Encoded offsets with respect to each roi.
               Has shape (B, N, num_classes * 4) or (B, N, 4) or
               (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
               when rois is a grid of anchors.Offset encoding follows [1]_.
            max_shape (Sequence[int] or torch.Tensor or Sequence[
               Sequence[int]],optional): Maximum bounds for boxes, specifies
               (H, W, C) or (H, W). If bboxes shape is (B, N, 4), then
               the max_shape should be a Sequence[Sequence[int]]
               and the length of max_shape should also be B.
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.

        Returns:
            torch.Tensor: Decoded boxes.
        """

        assert pred_bboxes.size(0) == bboxes.size(0)
        if pred_bboxes.ndim == 3:
            assert pred_bboxes.size(1) == bboxes.size(1)

        assert pred_bboxes.ndim == 2 and not torch.onnx.is_in_onnx_export()
            
        # single image decode
        # 제안된 bboxes(= proposals) delta값인 pred_bboxes를 적용한다.
        num_bboxes, num_classes = pred_bboxes.size(0), pred_bboxes.size(1) // 4
        if num_bboxes == 0:  decoded_bboxes = pred_bboxes
        
        deltas = pred_bboxes.reshape(-1, 4)
        means = deltas.new_tensor(self.means).view(1, -1)
        stds = deltas.new_tensor(self.stds).view(1, -1)
        denorm_deltas = deltas * stds + means
        
        dxy = denorm_deltas[:, :2]
        dwh = denorm_deltas[:, 2:]
        
        # Compute width/height of each roi
        rois_ = bboxes.repeat(1, num_classes).reshape(-1, 4)
        pxy = ((rois_[:, :2] + rois_[:, 2:]) * 0.5)
        pwh = (rois_[:, 2:] - rois_[:, :2])
        dxy_wh = pwh * dxy

        max_ratio = np.abs(np.log(wh_ratio_clip))
        if self.add_ctr_clamp:
            dxy_wh = torch.clamp(dxy_wh, max=self.ctr_clamp, min=-self.ctr_clamp)
            dwh = torch.clamp(dwh, max=max_ratio)
        else:
            dwh = dwh.clamp(min=-max_ratio, max=max_ratio)
            
        gxy = pxy + dxy_wh
        gwh = pwh * dwh.exp()
        x1y1 = gxy - (gwh * 0.5)
        x2y2 = gxy + (gwh * 0.5)
        bboxes = torch.cat([x1y1, x2y2], dim=-1)
        if self.clip_border and max_shape is not None:
            bboxes[..., 0::2].clamp_(min=0, max=max_shape[1])
            bboxes[..., 1::2].clamp_(min=0, max=max_shape[0])
        decoded_bboxes = bboxes.reshape(num_bboxes, -1)
    
        return decoded_bboxes







        
class RPNHead(BaseModule, metaclass=ABCMeta):
    def __init__(self,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Conv2d', std=0.01),
                 num_convs=1,
                 num_classes = 1,
                 feat_channels=256,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     scales=[8, 16, 32],
                     ratios=[0.5, 1.0, 2.0],
                     strides=[4, 8, 16, 32, 64]),
                 bbox_coder=dict(       # DeltaXYWHBBoxCoder
                     clip_border=True,
                     target_means=(.0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0)),
                 reg_decoded_bbox=False,
                 loss_cls=dict(         # CrossEntropyLoss
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None
                 ):
        self.num_convs = num_convs
        
        super(RPNHead, self).__init__(init_cfg)
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        
        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')
        self.reg_decoded_bbox = reg_decoded_bbox
        

        self.bbox_coder = DeltaXYWHBBoxCoder(**bbox_coder)
        self.loss_cls = CrossEntropyLoss(**loss_cls)
        self.loss_bbox = L1Loss(loss_bbox)
        
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = MaxIoUAssigner(**self.train_cfg.assigner)
            
        print("?!")# 완료
        exit()
        
    def init_weights(self):
        super(RPNHead, self).init_weights()
        # avoid init_cfg overwrite the initialization of `conv_offset`
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)
                
        
        
    
                
@MODELS.register_module()
class MaskRCNN:
    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):

        self.backbone = build_backbone(backbone)
        
        
        self.neck = FPN(in_channels= neck.in_channels, out_channels= neck.out_channels, num_outs= neck.num_outs)
        
        # build_rpn_head : RPNHead
        rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
        rpn_head_cfg = rpn_head.copy()
        rpn_head_cfg.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
        
        
        self.rpn_head = RPNHead(**rpn_head_cfg)
            
        
        exit()
        

