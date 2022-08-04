import kfp
from kfp.components import InputPath, create_component_from_func

from config import SAVES3_IMAGE 

def save_dataset(args : dict,
                 cfg_path: InputPath("dict"),
                 train_dataset_path: InputPath("dict"),
                 val_dataset_path: InputPath("dict")):
    
    import json
    import os
    import boto3
    from pipeline_taeuk4958.utils.utils import NpEncoder
    from pipeline_taeuk4958.configs.config import Config
    
    
    ## load config
    cfg_pyformat_path = cfg_path + ".py"        # cfg_pyformat_path : {wiorkspace}/inputs/cfg/data.py
                                                # can't command 'mv' 
    # change format to .py
    with open(cfg_path, "r") as f:
        data = f.read()
    with open(cfg_pyformat_path, "w") as f:
        f.write(data)       # 
    f.close()

    cfg = Config.fromfile(cfg_pyformat_path)    # cfg_pyformat_path : must be .py format   
    
    # load dataset    
    with open(train_dataset_path, "r", encoding='utf-8') as f:
        train_dataset = json.load(f)
    train_dataset_to_upload = os.path.join(os.getcwd(), cfg.json.train_file_name)
    json.dump(train_dataset, open(train_dataset_to_upload, "w"), indent = 4, cls = NpEncoder)
    train_dataset_in_storage = cfg.json.train_file_name
    
    with open(val_dataset_path, "r", encoding='utf-8') as f:
        val_dataset = json.load(f)
    val_dataset_to_upload = os.path.join(os.getcwd(), cfg.json.val_file_name)
    json.dump(val_dataset, open(val_dataset_to_upload, "w"), indent = 4, cls = NpEncoder)
    val_dataset_in_storage = cfg.json.val_file_name

    # access s3 
    s3 = boto3.resource('s3',
                      aws_access_key_id= args['access_key_id'],
                      aws_secret_access_key= args['secret_access_key']
                      )
    # upload dataset to bucket
    bucket = s3.Bucket(args['bucket_name'])
    bucket.upload_file(train_dataset_to_upload, train_dataset_in_storage)
    bucket.upload_file(val_dataset_to_upload, val_dataset_in_storage)
    


save_dataset_op  = create_component_from_func(func =save_dataset,
                                              base_image = SAVES3_IMAGE,        
                                              output_component_file="save_dataset.component.yaml")