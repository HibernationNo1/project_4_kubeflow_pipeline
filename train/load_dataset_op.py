
from kfp.components import create_component_from_func, OutputPath

from pipeline_config import Pipeline_Config
pl_cfg = Pipeline_Config


def download_dataset(cfg_dict : dict,                      
                     train_dataset_path: OutputPath("dict"),
                     val_dataset_path: OutputPath("dict"),):
    
    import json
    import os
    import subprocess
    import glob
    from pipeline_taeuk4958.configs.config import Config 
    from pipeline_taeuk4958.cloud.gs import gs_credentials, get_client_secrets
    from google.cloud import storage
 
    
    

    def download_dataset(cfg):
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(cfg.gs.recoded_dataset_bucket_name)
        
        train_dataset_in_gs = f'{cfg.gs.recoded_dataset_version}/{cfg.gs.train_file_name}' 
        val_dataset_in_gs = f'{cfg.gs.recoded_dataset_version}/{cfg.gs.val_file_name}'
        train_blob = bucket.blob(train_dataset_in_gs)
        val_blob = bucket.blob(val_dataset_in_gs)
        
        train_dataset_path = os.path.join(os.getcwd(), cfg.gs.train_file_name)
        val_dataset_path = os.path.join(os.getcwd(), cfg.gs.val_file_name)
        train_blob.download_to_filename(train_dataset_path)
        val_blob.download_to_filename(val_dataset_path)
        
        return train_dataset_path, val_dataset_path
    
        
    def download_images(cfg):
        # set client secret for dvc pull
              
        client_secrets_path = os.path.join(os.getcwd(), cfg.gs.client_secrets)
        gs_secret = get_client_secrets()
        
        json.dump(gs_secret, open(client_secrets_path, "w"), indent=4)   
        remote_bucket_command = f"dvc remote add -d -f bikes gs://{cfg.gs.ann_bucket_name}"
        credentials_command = f"dvc remote modify --local bikes credentialpath '{client_secrets_path}'" 
      
        subprocess.call([remote_bucket_command], shell=True)
        subprocess.call([credentials_command], shell=True)
        subprocess.call(["dvc pull"], shell=True)           # download dataset from GS by dvc
        
        # get annotations
        anns_dir = os.path.join(os.getcwd(), cfg.gs.anns_dir)
        anns_list = glob.glob(f"{anns_dir}/*.json")    
        if len(anns_list)==0 : raise OSError("Failed download images!!")
        print(f"\n number of annotations : {len(anns_list)} \n")
        
        anns_config_path = os.path.join(os.getcwd(), "ann_config.py")
        with open(anns_config_path, "r", encoding='utf-8') as f:
            anns_config = json.load(f)    
            
        return anns_list, anns_config
    
    
    if __name__=="__main__":
        
        config_file_path = os.path.join(os.getcwd(), cfg_dict['cfg_name'])
        with open(config_file_path, 'w') as f:
            f.write('\n')
        
        cfg = Config.fromfile(config_file_path, cfg_dict['cfg_dict'])        
       
        gs_credentials(cfg.gs.client_secrets)
        
                           
        train_dataset_path, val_dataset_path = download_dataset(cfg)       # download train, val dataset.json from google cloud storage
        anns_list, anns_config = download_images(cfg)
        
        print(f"len(train_dataset_path['annotations']), len(val_dataset_path['annotations']) : {len(train_dataset_path['annotations']), len(val_dataset_path['annotations'])}, len(anns_list) : {len(anns_list)}")
        
        print(f"anns_config : {anns_config}")
        print(f"train_dataset_path['info']['ann_version'] : {train_dataset_path['info']['ann_version']}")
        exit()
        # len(anns_list) len(train_dataset_path['annotations']) +  len(val_dataset_path['annotations'])
        
        
        
        
        
    
print(f"download_dataset base_image : {pl_cfg.LOAD_DATA_IMAGE}")    
download_dataset_op = create_component_from_func(func = download_dataset,
                                        base_image = pl_cfg.LOAD_DATA_IMAGE,
                                        output_component_file= pl_cfg.LOAD_DATA_COM_FILE)