_base_ = [
    'swin_maskrcnn/mask_rcnn.py',
    'swin_maskrcnn/dataset_config.py',
    'swin_maskrcnn/schedule_1x.py',
    './default.py',
    "pipeline/pipeline.py",
    "pipeline/dvc.py",
    "pipeline/database.py",
    "pipeline/gs.py"
]









