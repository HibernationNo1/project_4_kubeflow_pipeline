dvc = dict(
    category = "board_dataset",
    
    ann = dict(
        dir = "ann_dataset",
        remote = "ann_dataset",
        gs_bucket = "ann_dataset_hibernation",
        version = "0.0.5"
    ),
    
    record = dict(
        dir = "train_dataset",
        remote = 'train_dataset',
        gs_bucket = "train_dataset_hibernation",
        version = "0.0.1",                 # match the version of the ann dataset you want to download. 
        train = 'train_dataset.json',      # name of recordd file(.json format) for training 
        val = 'val_dataset.json'           # name of recordd file(.json format) for validation 
    )
)
