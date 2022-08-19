
from kfp.components import InputPath, create_component_from_func, OutputPath

from pipeline_config import Pipeline_Config
pl_cfg = Pipeline_Config


def download_dataset(cfg_path: InputPath("dict"),
                     train_dataset_path: OutputPath("dict"),
                     val_dataset_path: OutputPath("dict"),):
    
    import json
    import os
    from pipeline_taeuk4958.configs.config import load_config_in_pipeline
    from pipeline_taeuk4958.utils.utils import NpEncoder
    from pipeline_taeuk4958.cloud.gs import gs_credentials
    from google.cloud import storage
    
    

    def download_dataset(cfg):
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(cfg.gs.recoded_dataset_bucket_name)
        
        train_dataset_in_gs = f'{cfg.dataset.dataset_version}/{cfg.dataset.train_file_name}' 
        val_dataset_in_gs = f'{cfg.dataset.dataset_version}/{cfg.dataset.val_file_name}'
        train_dataset_path = os.path.join(os.getcwd(), cfg.dataset.train_file_name)
        val_dataset_path = os.path.join(os.getcwd(), cfg.dataset.val_file_name)
        
        train_blob = bucket.blob(train_dataset_in_gs)
        val_blob = bucket.blob(val_dataset_in_gs)
        
        train_blob.download_to_filename(train_dataset_path)
        val_blob.download_to_filename(val_dataset_path)
        
    
    if __name__=="__main__":
        cfg = load_config_in_pipeline(cfg_path)
        
        gs_credentials(cfg.gs.client_secrets)
        
                           
        download_dataset(cfg)       # download train, val dataset.json from google cloud storage
  
        
        train_dataset_path = os.path.join(os.getcwd(), cfg.dataset.train_file_name)
        with open(train_dataset_path, "r", encoding='utf-8') as f:
            train_dataset = json.load(f)
        
        val_dataset_path = os.path.join(os.getcwd(), cfg.dataset.val_file_name)
        with open(val_dataset_path, "r", encoding='utf-8') as f:
            val_dataset = json.load(f)
            
                # save recorded dataset
        json.dump(train_dataset, open(train_dataset_path, "w"), indent=4, cls = NpEncoder)
        json.dump(val_dataset, open(val_dataset_path, "w"), indent=4, cls = NpEncoder)
        
        
        
    
  
download_dataset_op = create_component_from_func(func = download_dataset,
                                        base_image = pl_cfg.LOAD_DATA_IMAGE,
                                        output_component_file= pl_cfg.LOAD_DATA_COM_FILE)