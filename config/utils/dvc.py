dvc = dict(
    remote = "ann_dataset",
    
    
    category = "test_dataset",
    ann = dict(
        name = "ann",
        version = "0.0.1",
    ),
    
    recode = dict(
        name = "recode",
        version = "0.0.1",
        train = 'train_dataset.json',      # name of recoded file(.json format) for training 
        val = 'val_dataset.json'           # name of recoded file(.json format) for validation 
    ),
)

git = dict(
    remote = "origin",
    branch = "master"
)