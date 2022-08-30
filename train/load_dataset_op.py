
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
    import time 
    from pipeline_taeuk4958.configs.config import Config 
    from pipeline_taeuk4958.cloud.gs import gs_credentials, get_client_secrets
    from google.cloud import storage
    
    
    from mmcv.utils import get_logger
    from mmdet_taeuk4958.utils import collect_env, get_device
    from mmdet_taeuk4958.apis import init_random_seed, set_random_seed
    from mmdet_taeuk4958.datasets import build_dataset
    
    def get_config():
        config_file_path = os.path.join(os.getcwd(), cfg_dict['cfg_name'])
        with open(config_file_path, 'w') as f:
            f.write('\n')
 
        return Config.fromfile(config_file_path, cfg_dict['cfg_dict'])           

    def download_record(cfg):
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
        
        # set config for training
        cfg.data.train.ann_file = train_dataset_path
        cfg.data.val.ann_file = val_dataset_path
        
        return train_dataset_path, val_dataset_path, cfg
    
        
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
        
        anns_config_path = os.path.join(os.getcwd(), cfg.gs.anns_config_path)
        with open(anns_config_path, "r", encoding='utf-8') as f:
            anns_config = json.load(f)   
            
        cfg.data.train.img_prefix = anns_dir    
        cfg.data.val.img_prefix = anns_dir
            
        return anns_list, anns_config, cfg
    
    
    def download_dataset(cfg):
        train_dataset_path, val_dataset_path, cfg = download_record(cfg)       # download train, val dataset.json from google cloud storage
        anns_list, anns_config, cfg = download_images(cfg)
        
        with open(train_dataset_path, "r", encoding='utf-8') as f:
            train_dataset = json.load(f) 
        with open(val_dataset_path, "r", encoding='utf-8') as f:
            val_dataset = json.load(f) 

        dataset_num, ann_num= len(train_dataset['images']) + len(val_dataset['images']), len(anns_list)
       
        assert dataset_num == ann_num, f"\n Invalid annotations or dataset loaded. \n number of dataset: {dataset_num}, number of annotations: {ann_num}"
        assert anns_config['ann_version'] == train_dataset['info']['ann_version'], f"\n Invalid version for training. check dataset or annotations version \n \
                                                                                    dataset_version: {train_dataset['info']['ann_version']}, anns_config['ann_version']: {anns_config['ann_version']}"
        
        for image_info in train_dataset["images"]:
            filepath = os.path.join(os.path.join(os.getcwd(), cfg.gs.anns_dir), image_info['file_name'])
            print(f"filepath : {filepath}")
            image_info['file_name'] = filepath
            
        return train_dataset, val_dataset, cfg
    
    
    
    if __name__=="__main__":
        cfg = get_config()
        gs_credentials(cfg.gs.client_secrets)       # set client secrets
        train_dataset, val_dataset, cfg = download_dataset(cfg)
    
        def get_logger_set_meta(work_dir_gs_storage, cfg):
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            log_file = os.path.join(work_dir_gs_storage, f'{cfg.model_version}.log')
            logger = get_logger(name= cfg.pipeline.pipeline_name, log_file=log_file, log_level=cfg.log_level) 
            
            # log env info      
            env_info_dict = collect_env()
            env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
            dash_line = '-' * 60 + '\n'
            logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)          # delete '#' if you want print Environment info  
            
            cfg.device = get_device() 
            meta = dict()
            meta['env_info'] = env_info
            meta['config'] = cfg.pretty_text
            
            seed = init_random_seed(cfg.seed, device=cfg.device)
            logger.info(f'Set random seed to {seed}, '
                    f'deterministic: {cfg.deterministic}')
            
            set_random_seed(seed, deterministic=cfg.deterministic)
            cfg.seed = seed
            meta['seed'] = seed
            meta['exp_name'] = cfg_dict['cfg_name']
            return logger, meta, timestamp, cfg
        
        work_dir_gs_storage = os.path.join(cfg.gs.model_bucket_name, cfg.model_version)
        logger, meta, timestamp, cfg = get_logger_set_meta(work_dir_gs_storage, cfg)
        
        print(f"\n cfg.data : {cfg.data}")
        # datasets = [build_dataset(cfg.data.train)]      # <class 'mmdet.datasets.custom.CocoDataset'>
    
    
        
        
        
        
    
print(f"download_dataset base_image : {pl_cfg.LOAD_DATA_IMAGE}")    
download_dataset_op = create_component_from_func(func = download_dataset,
                                        base_image = pl_cfg.LOAD_DATA_IMAGE,
                                        output_component_file= pl_cfg.LOAD_DATA_COM_FILE)