import torch
import torch.nn as nn


from base_module import BaseModule, ModuleList
from loss import CrossEntropyLoss
from assigner import MaxIoUAssigner

from models.maskrcnn.bbox_head import SingleRoIExtractor, Shared2FCBBoxHead
from datasets.sampler import RandomSampler
from basic_module import ConvModule, ConvTranspose2d, Conv2d

if torch.__version__ == 'parrots':
    TORCH_VERSION = torch.__version__
else:
    # torch.__version__ could be 1.3.1+cu92, we only need the first two
    # for comparison
    TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])



    
def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois



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
    
    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
                proposals[n]: [proposal_cfg.max_per_img, 5],    5: [x_min, y_min, x_max, y_max, score]
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)   # batch_size
            sampling_results = []
            for i in range(num_imgs):
                # AssignResult(num_gts=num_gt_instance, gt_inds.shape=(proposal_cfg.max_per_img,),
                # max_overlaps.shape=(proposal_cfg.max_per_img,), labels.shape=(proposal_cfg.max_per_img,))
                assign_result = self.bbox_assigner.assign(proposal_list[i], gt_bboxes[i], gt_labels = gt_labels[i])
                
                sampling_result = self.bbox_sampler.sample(assign_result,  
                                                           proposal_list[i], gt_bboxes[i], gt_labels[i],
                                                           feats=[lvl_feat[i][None] for lvl_feat in x])
    
                sampling_results.append(sampling_result)
        
        # bbox head forward and loss
        losses = dict()
        if self.with_bbox:
            # bbox_results:dict
            #   cls_score: [1024, 7],        bbox_pred: [1024, 24],       bbox_feats: [1024, 256, 7, 7]
            #   loss_bbox:dict,   'loss_cls': tensor([float]), 'acc': tensor([float]), 'loss_bbox': tensor([float])
            bbox_results = self._bbox_forward_train(x, sampling_results, gt_bboxes, gt_labels)
            losses.update(bbox_results['loss_bbox'])
        
        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,        # 이거서부터
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])
            
            
    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        
        # bbox_results:dict,    cls_score: [1024, 7],        bbox_pred: [1024, 24],       bbox_feats: [1024, 256, 7, 7]
        bbox_results = self._bbox_forward(x, rois)

        # len:4 == labels([1024]), label_weights([1024]), bbox_targets([1024, 4]), bbox_weights([1024, 4])
        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        # loss_bbox:dict,   'loss_cls': tensor([float]), 'acc': tensor([float]), 'loss_bbox': tensor([float])
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)
        
        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
    
    
    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        # bbox_feats.shape: [1024, 256, 7, 7]
        bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)  
        
        cls_score, bbox_pred = self.bbox_head(bbox_feats)   
        bbox_results = dict(cls_score=cls_score, 
                            bbox_pred=bbox_pred, 
                            bbox_feats=bbox_feats)
        return bbox_results
    
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
            
            self.bbox_assigner = MaxIoUAssigner(**self.train_cfg.assigner)        
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
                




class NewEmptyTensorOp(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return NewEmptyTensorOp.apply(grad, shape), None
    
  


            





