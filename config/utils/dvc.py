dvc = dict(
    category = "board_dataset",
    
    ann = dict(
        dir = "ann_dataset",
        remote = "ann_dataset",
        gs_bucket = "ann_dataset_hibernation",
        version = "v0.0.7"				# match the version of the ann dataset you want to select from DB 
    ),
    
    record = dict(
        dir = "train_dataset",
        remote = 'train_dataset',
        gs_bucket = "train_dataset_hibernation",
        version = "v0.0.5",                # determine version of `train_dataset` for training. 
        train = 'train_dataset.json',      # name of recorded file(.json format) for training 
        val = 'val_dataset.json'           # name of recorded file(.json format) for validation 
    )
)
