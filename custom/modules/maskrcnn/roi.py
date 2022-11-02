from distutils.command.config import config
import torch
from assigner import MaxIoUAssigner

from modules.maskrcnn.bbox_head import SingleRoIExtractor, Shared2FCBBoxHead
from modules.maskrcnn.mask_head import FCNMaskHead
from datasets.sampler import RandomSampler
from modules.base_module import BaseModule

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

                # SamplingResult
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
            mask_results = self._mask_forward_train(x, sampling_results,       
                                                    bbox_results['bbox_feats'],
                                                    gt_masks)
            losses.update(mask_results['loss_mask'])
            
    
    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks):
        """Run forward function and calculate loss for mask head in
        training.
        """
        
        if not self.share_roi_extractor:        # True
            # sampling_results: list of SamplingResult
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])   # [num_instance in batch images, num_levels]
            mask_results = self._mask_forward(x, pos_rois)  
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(torch.ones(res.pos_bboxes.shape[0],device=device,
                                           dtype=torch.uint8)
                                )
                pos_inds.append(torch.zeros(res.neg_bboxes.shape[0],device=device,
                                            dtype=torch.uint8)
                                )
            pos_inds = torch.cat(pos_inds)

            # dict,     
            #   mask_pred (shape=[num_instance in batch images, 256, win_size*2, win_size*2])
            #   mask_feats (shape=[num_instance in batch images, 256, win_size*4, win_size*4])
            mask_results = self._mask_forward(x, pos_inds=pos_inds, bbox_feats=bbox_feats)
        
        # [num_instance in batch images, win_size*4, win_size*4]
        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg) 

        # [num_instance in batch images]
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)
        
        
     
            
            
    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))

        if rois is not None:    # True
            # mask_feats.shape= [num_instance in batch images, 256, win_size*2, win_size*2]
            mask_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], rois)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        # mask_pred.shape= [num_instance in batch images, 256, win_size*4, win_size*4]
        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results
            
            
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
            from mmdet_taeuk4958.core.bbox.assigners.max_iou_assigner import MaxIoUAssigner as MaxIoUAssigner_
            self.bbox_assigner = MaxIoUAssigner_(**self.train_cfg.assigner)
            # self.bbox_assigner = MaxIoUAssigner(**self.train_cfg.assigner)     
            
            from mmdet_taeuk4958.core.bbox.samplers.random_sampler import RandomSampler as RandomSampler_
            self.bbox_sampler = RandomSampler_(**self.train_cfg.sampler)
            # self.bbox_sampler = RandomSampler(**self.train_cfg.sampler)
            
    
    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        
        
        
        
        # ----
        from mmdet_taeuk4958.models.roi_heads.bbox_heads.convfc_bbox_head import Shared2FCBBoxHead as Shared2FCBBoxHead_####
        from utils.config import Config        
        bbox_head_1 = Config({'in_channels': 256, 
                            'fc_out_channels': 1024, 
                            'roi_feat_size': 7, 
                            'num_classes': 6, 
                            'bbox_coder': {'type': 'DeltaXYWHBBoxCoder', 
                                           'target_means': [0.0, 0.0, 0.0, 0.0], 
                                           'target_stds': [0.1, 0.1, 0.2, 0.2]}, 
                            'reg_class_agnostic': False, 
                            'loss_cls': {'type': 'CrossEntropyLoss', 
                                         'use_sigmoid': False, 
                                         'loss_weight': 1.0}, 
                            'loss_bbox': {'type': 'L1Loss', 'loss_weight': 1.0}})
        self.bbox_head = Shared2FCBBoxHead_(**bbox_head_1)
        # ---
        
        # self.bbox_roi_extractor = SingleRoIExtractor(**bbox_roi_extractor)
        # self.bbox_head = Shared2FCBBoxHead(**bbox_head)       # False
        
        

        
    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = SingleRoIExtractor(**mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
   
        self.mask_head = FCNMaskHead(**mask_head)
        
        








class NewEmptyTensorOp(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return NewEmptyTensorOp.apply(grad, shape), None
    
  


            





