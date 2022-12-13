_base_ = [
    'swin_maskrcnn/mask_rcnn.py',
    'swin_maskrcnn/dataset_config.py',
    'swin_maskrcnn/schedule_1x.py',
    './default.py',
]


pipeline = dict(
    pipeline_name = 'train',
    pipeline_version = "0.1"
)


gs = dict(
    client_secrets = "client_secrets.json",
    ann_bucket_name = "dataset_tesuk4958",
    recoded_dataset_bucket_name = "pipeline_taeuk4958",
    model_bucket_name = 'pipeline_models_taeuk4958',
    recoded_dataset_version = "0.1"   
    )

db = dict(
    host='localhost', 
    port=3306, 
    user='project-pipeline', 
    db='ann_dataset', 
    charset='utf8'   
)

