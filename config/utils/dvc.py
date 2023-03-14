dvc = dict(
    category = "test_dataset",
    ann = dict(
        name = "ann",
        remote = "ann_dataset",
        gs_bucket = "ann_dataset_taeuk4958",
        version = "0.0.6",
    ),
    
    record = dict(
        name = "record",
        remote = 'record_dataset',
        gs_bucket = "train_dataset_taeuk4958",
        version = "0.0.6",                 # match the version of the ann dataset you want to download. 
        train = 'train_dataset.json',      # name of recordd file(.json format) for training 
        val = 'val_dataset.json'           # name of recordd file(.json format) for validation 
    )
)

git = dict(
    remote = "origin",
    branch = "master",
    dataset_repo = "pipeline_dataset",
    package_repo = "hibernation_no1"
)
