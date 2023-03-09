_base_ = [
    'base/dataset_config.py',
    "utils/dvc.py",
    "utils/database.py",
    "utils/utils.py"
]

val_result = "result/evaluate"
model_path = None

device = 'cuda:0'


eval_cfg = dict(
    iou_thrs = [0.3, 0.9],          # range of iou threshold
    num_thrs_divi = 9,             # divide range of `iou_threshold` by number of `num_thrshd_divi` for compute mAP
    num_window = 3,
    batch_size = 4,
    save_plot = True,
    show_plot = False,
    show_score_thr = 0.7,           # threshold of score to draw image
    compare_board_info = True        
)

