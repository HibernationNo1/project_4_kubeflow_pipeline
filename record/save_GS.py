import kfp
from kfp.components import InputPath, create_component_from_func

from config import SAVE_GS_IMAGE, SAVE_GS_COM_FILE

def save_dataset(gs_secret : dict,
                 cfg_path: InputPath("dict"),
                 train_dataset_path: InputPath("dict"),
                 val_dataset_path: InputPath("dict")):
    
    import json
    import os
    from pipeline_taeuk4958.utils.utils import NpEncoder
    from pipeline_taeuk4958.configs.config import load_config_in_pipeline
    from pipeline_taeuk4958.cloud.gs import set_gs_credentials
    
    from google.cloud import storage
        

    def load_recorded_dataset(cfg):
        with open(train_dataset_path, "r", encoding='utf-8') as f:
            train_dataset = json.load(f)
        train_dataset_to_upload = os.path.join(os.getcwd(), cfg.dataset.train_file_name)
        json.dump(train_dataset, open(train_dataset_to_upload, "w"), indent = 4, cls = NpEncoder)
        train_dataset_in_storage_path = f'{cfg.gs.recoded_dataset_version}/{cfg.dataset.train_file_name}' 
        
        with open(val_dataset_path, "r", encoding='utf-8') as f:
            val_dataset = json.load(f)
        val_dataset_to_upload = os.path.join(os.getcwd(), cfg.dataset.val_file_name)
        json.dump(val_dataset, open(val_dataset_to_upload, "w"), indent = 4, cls = NpEncoder)
        val_dataset_in_storage_path = f'{cfg.gs.recoded_dataset_version}/{cfg.dataset.val_file_name}'
        
        return train_dataset_in_storage_path, val_dataset_in_storage_path


    
        
        
        
    def save_dataset_gs(cfg, train_dataset_path, val_dataset_path):        
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(cfg.gs.recoded_dataset_bucket_name)
        
        train_blob = bucket.blob(train_dataset_path)
        train_blob.upload_from_filename(cfg.dataset.train_file_name)
        
        val_blob = bucket.blob(val_dataset_path)
        val_blob.upload_from_filename(cfg.dataset.val_file_name)
        
        
        
    if __name__=="__main__":
        cfg = load_config_in_pipeline(cfg_path)
        set_gs_credentials(cfg.gs.client_secrets, gs_secret)
        
        train_dataset_path, val_dataset_path = load_recorded_dataset(cfg)
        
        save_dataset_gs(cfg, train_dataset_path, val_dataset_path)
    
    

    
    
    


save_dataset_op  = create_component_from_func(func =save_dataset,
                                              base_image = SAVE_GS_IMAGE,        
                                              output_component_file=SAVE_GS_COM_FILE)