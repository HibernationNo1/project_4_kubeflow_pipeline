import torch.nn.functional as F
import torch.nn as nn

from base_module import BaseModule
from basic_module import ConvModule

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
        self.upsample_cfg = upsample_cfg.copy()
        
        self.fp16_enabled = False  # TODO fp16으로 변환하여 학습 진행해보기
        # mmcv > runner > ffp16_utils.py > def auto_fp16

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

        self.lateral_convs = nn.ModuleList()        #  from base_module import ModuleList 사용시 loss상승
        self.fpn_convs = nn.ModuleList()
        
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(    # lateral_convs
                in_channels[i],
                out_channels,
                kernel_size= 1,
                inplace=False,
                act_cfg=None)   # activate layer: None
            fpn_conv = ConvModule(  # fpn_convs
                out_channels,
                out_channels,
                kernel_size= 3,
                padding=1,
                inplace=False,
                act_cfg=None)   # activate layer: None

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
    
    # @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        # n의 길이를 가진 list를 elements별로 forward
        assert len(inputs) == len(self.in_channels)
        
        # build laterals        각각의 conv에 각각의 input을 입력
        laterals = [lateral_conv(inputs[i + self.start_level])
                    for i, lateral_conv in enumerate(self.lateral_convs)]
        
        
        
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):        # 3, 2, 1, 0 중 3을 제외한 lavel에만 interpolate(보간법)적용
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:     # TODO: scale_factor를 위한 cfg가 mmdet에 있는지 확인. 있으면 사용해보기 
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], 
                                                                  **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], 
                                                                  size=prev_shape, 
                                                                  **self.upsample_cfg)
               
        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels) ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)      # TODO: solo v2에선 해야하나?
            for i in range(self.num_outs - used_backbone_levels):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        
        return tuple(outs)
