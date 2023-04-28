_base_ = [
    'base/dataset_config.py',
    "utils/dvc.py",
    "utils/database.py",
    "utils/utils.py"
]

train_result = "result/train"
eval_result = "result/eval"
model_path = None
device = 'cuda:0'

# Key to determine if it is the best model. ex) mAP
# AIR: Exact Inference Rate 
key_name = ["dv_mAP", "EIR"]  

eval_cfg = dict(
    score_thrs = [0.3, 0.9],          # range of iou threshold
    num_thrs_divi = 9,             # divide range of `iou_threshold` by number of `num_thrshd_divi` for compute mAP
    iou_thrs = 0.6,
    num_window = 3,
    batch_size = 4,
    save_plot = True,
    show_plot = False,
    show_score_thr = 0.7,           # threshold of score to draw image
    compare_board_info = True        
)

