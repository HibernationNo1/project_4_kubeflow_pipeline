import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from base_module import BaseModule
from basic_module import ConvModule
from loss import CrossEntropyLoss, L1Loss
from utils.utils import multi_apply, load_ext
from models.maskrcnn.coder import DeltaXYWHBBoxCoder

ext_module = load_ext('_ext',['roi_align_forward', 'roi_align_backward'])



  
def accuracy(pred, target, topk=1, thresh=None):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class)
        target (torch.Tensor): The target of each prediction, shape (N, )
        topk (int | tuple[int], optional): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thresh (float, optional): If not None, predictions with scores under
            this threshold are considered incorrect. Default to None.

    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    maxk = max(topk)
    if pred.size(0) == 0:
        accu = [pred.new_tensor(0.) for i in range(len(topk))]
        return accu[0] if return_single else accu
    assert pred.ndim == 2 and target.ndim == 1
    assert pred.size(0) == target.size(0)
    assert maxk <= pred.size(1), \
        f'maxk {maxk} exceeds pred dimension {pred.size(1)}'
    pred_value, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()  # transpose to shape (maxk, N)
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    if thresh is not None:
        # Only prediction values larger than thresh are counted as correct
        correct = correct & (pred_value > thresh).t()
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res
    
    
class RoIAlignFunction(Function):
    
    # @staticmethod
    # def symbolic(g, input, rois, output_size, spatial_scale, sampling_ratio,
    #              pool_mode, aligned):
    #     from ..onnx import is_custom_op_loaded
    #     has_custom_op = is_custom_op_loaded()
    #     if has_custom_op:
    #         return g.op(
    #             'mmcv::MMCVRoiAlign',
    #             input,
    #             rois,
    #             output_height_i=output_size[0],
    #             output_width_i=output_size[1],
    #             spatial_scale_f=spatial_scale,
    #             sampling_ratio_i=sampling_ratio,
    #             mode_s=pool_mode,
    #             aligned_i=aligned)
    #     else:
    #         from torch.onnx import TensorProtoDataType
    #         from torch.onnx.symbolic_helper import _slice_helper
    #         from torch.onnx.symbolic_opset9 import squeeze, sub

    #         # batch_indices = rois[:, 0].long()
    #         batch_indices = _slice_helper(
    #             g, rois, axes=[1], starts=[0], ends=[1])
    #         batch_indices = squeeze(g, batch_indices, 1)
    #         batch_indices = g.op(
    #             'Cast', batch_indices, to_i=TensorProtoDataType.INT64)
    #         # rois = rois[:, 1:]
    #         rois = _slice_helper(g, rois, axes=[1], starts=[1], ends=[5])
    #         if aligned:
    #             # rois -= 0.5/spatial_scale
    #             aligned_offset = g.op(
    #                 'Constant',
    #                 value_t=torch.tensor([0.5 / spatial_scale],
    #                                      dtype=torch.float32))
    #             rois = sub(g, rois, aligned_offset)
    #         # roi align
    #         return g.op(
    #             'RoiAlign',
    #             input,
    #             rois,
    #             batch_indices,
    #             output_height_i=output_size[0],
    #             output_width_i=output_size[1],
    #             spatial_scale_f=spatial_scale,
    #             sampling_ratio_i=max(0, sampling_ratio),
    #             mode_s=pool_mode)

    @staticmethod
    def forward(ctx,
                input,
                rois,
                output_size,
                spatial_scale=1.0,
                sampling_ratio=0,
                pool_mode='avg',
                aligned=True):
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        assert pool_mode in ('max', 'avg')
        ctx.pool_mode = 0 if pool_mode == 'max' else 1
        ctx.aligned = aligned
        ctx.input_shape = input.size()

        assert rois.size(1) == 5, 'RoI must be (idx, x1, y1, x2, y2)!'

        output_shape = (rois.size(0), input.size(1), ctx.output_size[0],
                        ctx.output_size[1])
        output = input.new_zeros(output_shape)
        if ctx.pool_mode == 0:
            argmax_y = input.new_zeros(output_shape)
            argmax_x = input.new_zeros(output_shape)
        else:
            argmax_y = input.new_zeros(0)
            argmax_x = input.new_zeros(0)

        ext_module.roi_align_forward(
            input,
            rois,
            output,
            argmax_y,
            argmax_x,
            aligned_height=ctx.output_size[0],
            aligned_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            pool_mode=ctx.pool_mode,
            aligned=ctx.aligned)

        ctx.save_for_backward(rois, argmax_y, argmax_x)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, argmax_y, argmax_x = ctx.saved_tensors
        grad_input = grad_output.new_zeros(ctx.input_shape)
        # complex head architecture may cause grad_output uncontiguous.
        grad_output = grad_output.contiguous()
        ext_module.roi_align_backward(
            grad_output,
            rois,
            argmax_y,
            argmax_x,
            grad_input,
            aligned_height=ctx.output_size[0],
            aligned_width=ctx.output_size[1],
            spatial_scale=ctx.spatial_scale,
            sampling_ratio=ctx.sampling_ratio,
            pool_mode=ctx.pool_mode,
            aligned=ctx.aligned)
        return grad_input, None, None, None, None, None, None


roi_align = RoIAlignFunction.apply




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
        
    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N.\
                The other 4 columns are xyxy.
        """
        if self.use_torchvision:
            from torchvision.ops import roi_align as tv_roi_align
            if 'aligned' in tv_roi_align.__code__.co_varnames:
                return tv_roi_align(input, rois, self.output_size,
                                    self.spatial_scale, self.sampling_ratio,
                                    self.aligned)
            else:
                if self.aligned:
                    rois -= rois.new_tensor([0.] +
                                            [0.5 / self.spatial_scale] * 4)
                return tv_roi_align(input, rois, self.output_size,
                                    self.spatial_scale, self.sampling_ratio)
        else:
            return roi_align(input, rois, self.output_size, self.spatial_scale,
                             self.sampling_ratio, self.pool_mode, self.aligned)
        
    def __repr__(self):
        s = self.__class__.__name__
        s += f'(output_size={self.output_size}, '
        s += f'spatial_scale={self.spatial_scale}, '
        s += f'sampling_ratio={self.sampling_ratio}, '
        s += f'pool_mode={self.pool_mode}, '
        s += f'aligned={self.aligned}, '
        s += f'use_torchvision={self.use_torchvision})'
        return s
    
    
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
    
    # @force_fp32(apply_to=('feats', ), out_fp16=True)
    def forward(self, feats, rois, roi_scale_factor=None):
        """Forward function."""
        out_size = self.roi_layers[0].output_size
        num_levels = len(feats)
        expand_dims = (-1, self.out_channels * out_size[0] * out_size[1])
        if torch.onnx.is_in_onnx_export():
            # Work around to export mask-rcnn to onnx
            roi_feats = rois[:, :1].clone().detach()
            roi_feats = roi_feats.expand(*expand_dims)
            roi_feats = roi_feats.reshape(-1, self.out_channels, *out_size)
            roi_feats = roi_feats * 0
        else:
            roi_feats = feats[0].new_zeros(
                rois.size(0), self.out_channels, *out_size)
        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        target_lvls = self.map_roi_levels(rois, num_levels)

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            mask = target_lvls == i
            if torch.onnx.is_in_onnx_export():
                # To keep all roi_align nodes exported to onnx
                # and skip nonzero op
                mask = mask.float().unsqueeze(-1)
                # select target level rois and reset the rest rois to zero.
                rois_i = rois.clone().detach()
                rois_i *= mask
                mask_exp = mask.expand(*expand_dims).reshape(roi_feats.shape)
                roi_feats_t = self.roi_layers[i](feats[i], rois_i)
                roi_feats_t *= mask_exp
                roi_feats += roi_feats_t
                continue
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            if inds.numel() > 0:
                rois_ = rois[inds]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)
                roi_feats[inds] = roi_feats_t
            else:
                # Sometimes some pyramid levels will not be used for RoI
                # feature extraction and this will cause an incomplete
                # computation graph in one GPU, which is different from those
                # in other GPUs and will cause a hanging error.
                # Therefore, we add it to ensure each feature pyramid is
                # included in the computation graph to avoid runtime bugs.
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
        return roi_feats
    
    def map_roi_levels(self, rois, num_levels):
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5).
            num_levels (int): Total level number.

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls
    
    

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
            self.fc_cls = nn.Linear(in_features=self.cls_last_dim,
                                    out_features=cls_channels)
         
        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear( in_features=self.cls_last_dim, out_features=out_dim_reg)
        
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
    @property
    def custom_activation(self):
        return getattr(self.loss_cls, 'custom_activation', False)

    # @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()
        return losses
    

    def get_targets(self,
                    sampling_results,
                    gt_bboxes,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True):
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Almost the same as the implementation in bbox_head, we passed
        additional parameters pos_inds_list and neg_inds_list to
        `_get_target_single` function.

        Args:
            sampling_results (List[obj:SamplingResults]): Assign results of
                all images in a batch after sampling.
            gt_bboxes (list[Tensor]): Gt_bboxes of all images in a batch,
                each tensor has shape (num_gt, 4),  the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            gt_labels (list[Tensor]): Gt_labels of all images in a batch,
                each tensor has shape (num_gt,).
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
            concat (bool): Whether to concatenate the results of all
                the images in a single batch.

        Returns:
            Tuple[Tensor]: Ground truth for proposals in a single image.
            Containing the following list of Tensors:

                - labels (list[Tensor],Tensor): Gt_labels for all
                  proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - label_weights (list[Tensor]): Labels_weights for
                  all proposals in a batch, each tensor in list has
                  shape (num_proposals,) when `concat=False`, otherwise
                  just a single tensor has shape (num_all_proposals,).
                - bbox_targets (list[Tensor],Tensor): Regression target
                  for all proposals in a batch, each tensor in list
                  has shape (num_proposals, 4) when `concat=False`,
                  otherwise just a single tensor has shape
                  (num_all_proposals, 4), the last dimension 4 represents
                  [tl_x, tl_y, br_x, br_y].
                - bbox_weights (list[tensor],Tensor): Regression weights for
                  all proposals in a batch, each tensor in list has shape
                  (num_proposals, 4) when `concat=False`, otherwise just a
                  single tensor has shape (num_all_proposals, 4).
        """
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        
        
        labels, label_weights, bbox_targets, bbox_weights = multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_labels_list,
            cfg=rcnn_train_cfg)

        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            bbox_targets = torch.cat(bbox_targets, 0)
            bbox_weights = torch.cat(bbox_weights, 0)
        return labels, label_weights, bbox_targets, bbox_weights


    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_labels, cfg):
        """Calculate the ground truth for proposals in the single image
        according to the sampling results.

        Args:
            pos_bboxes (Tensor): Contains all the positive boxes,
                has shape (num_pos, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            neg_bboxes (Tensor): Contains all the negative boxes,
                has shape (num_neg, 4), the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_bboxes (Tensor): Contains gt_boxes for
                all positive samples, has shape (num_pos, 4),
                the last dimension 4
                represents [tl_x, tl_y, br_x, br_y].
            pos_gt_labels (Tensor): Contains gt_labels for
                all positive samples, has shape (num_pos, ).
            cfg (obj:`ConfigDict`): `train_cfg` of R-CNN.

        Returns:
            Tuple[Tensor]: Ground truth for proposals
            in a single image. Containing the following Tensors:

                - labels(Tensor): Gt_labels for all proposals, has
                  shape (num_proposals,).
                - label_weights(Tensor): Labels_weights for all
                  proposals, has shape (num_proposals,).
                - bbox_targets(Tensor):Regression target for all
                  proposals, has shape (num_proposals, 4), the
                  last dimension 4 represents [tl_x, tl_y, br_x, br_y].
                - bbox_weights(Tensor):Regression weights for all
                  proposals, has shape (num_proposals, 4).
        """
        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg

        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_bboxes.new_full((num_samples, ),
                                     self.num_classes,
                                     dtype=torch.long)
        label_weights = pos_bboxes.new_zeros(num_samples)
        bbox_targets = pos_bboxes.new_zeros(num_samples, 4)
        bbox_weights = pos_bboxes.new_zeros(num_samples, 4)
        if num_pos > 0:
            labels[:num_pos] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[:num_pos] = pos_weight
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    pos_bboxes, pos_gt_bboxes)
            else:
                # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
                # is applied directly on the decoded bounding boxes, both
                # the predicted boxes and regression targets should be with
                # absolute coordinate format.
                pos_bbox_targets = pos_gt_bboxes
            bbox_targets[:num_pos, :] = pos_bbox_targets
            bbox_weights[:num_pos, :] = 1
        if num_neg > 0:
            label_weights[-num_neg:] = 1.0

        return labels, label_weights, bbox_targets, bbox_weights
    
    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

 
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred
        
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