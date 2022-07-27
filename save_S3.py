import kfp
from kfp.components import InputPath, create_component_from_func

from configs.labelme_config import Labelme_Config as for_type
config = for_type
cfg_type = type(config)

def save_labelme(cfg : cfg_type, train_dataset_path: InputPath("dict"), val_dataset_path: InputPath("dict")):
    
    from utils import NpEncoder
    import json
    import os
    import boto3

    with open(train_dataset_path, "r", encoding='utf-8') as f:
        train_dataset = json.load(f)
    train_dataset_to_upload = os.path.join(os.getcwd(), cfg.json['train_file_name'])
    json.dump(train_dataset, open(train_dataset_to_upload, "w"), indent = 4, cls = NpEncoder)
    train_dataset_in_storage = cfg.json['train_file_name']
    
    with open(val_dataset_path, "r", encoding='utf-8') as f:
        val_dataset = json.load(f)
    val_dataset_to_upload = os.path.join(os.getcwd(), cfg.json['val_file_name'])
    json.dump(val_dataset, open(val_dataset_to_upload, "w"), indent = 4, cls = NpEncoder)
    val_dataset_in_storage = cfg.json['val_file_name']

    AWS_ACCESS_KEY_ID ="AKIAXZX44242SIJNTR5O"
    AWS_SECRET_ACCESS_KEY = "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"
    bucket_name = "hibernationproject"
    
    s3 = boto3.resource('s3',
                      aws_access_key_id=AWS_ACCESS_KEY_ID,
                      aws_secret_access_key=AWS_SECRET_ACCESS_KEY
                      )

    bucket = s3.Bucket(bucket_name)
    bucket.upload_file(train_dataset_to_upload, train_dataset_in_storage)
    bucket.upload_file(val_dataset_to_upload, val_dataset_in_storage)
    


save_labelme_op  = create_component_from_func(func =save_labelme,
                                              base_image = 'hibernation4958/labelme.0.1',        # 'hibernation4958/for_save.0.1'
                                              output_component_file="for_save.component.yaml")