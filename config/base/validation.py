# validation
val = dict(
    iou_thrs = [0.3, 0.9],          # range of iou threshold
    num_thrs_divi = 10,             # divide range of `iou_threshold` by number of `num_thrshd_divi` for compute mAP
    cfd_thrs = 0.6,                 # threshold of confidence score
    mask2polygon = None             # function for converte from mask to poltgon. need to run validation.
)
 
