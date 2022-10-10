from abc import ABCMeta
import torch
import numpy as np

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