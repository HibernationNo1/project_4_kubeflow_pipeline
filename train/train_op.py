
from kfp.components import create_component_from_func

from pipeline_config import Pipeline_Config
pl_cfg = Pipeline_Config


def train(cfg_dict : dict):
    
    import json
    import os
    import subprocess
    import glob
    import time
    from pipeline_taeuk4958.configs.utils import load_config 
    from pipeline_taeuk4958.cloud.gs import gs_credentials, get_client_secrets
    from google.cloud import storage
    
    
    from mmcv.utils import get_logger, get_git_hash
    from mmdet_taeuk4958 import __version__
    from mmdet_taeuk4958.utils import collect_env, get_device
    from mmdet_taeuk4958.apis import init_random_seed, set_random_seed, train_detector
    from mmdet_taeuk4958.datasets import build_dataset
    from mmdet_taeuk4958.models import build_detector
    
    
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
        
        imgs_dir = os.path.join(os.getcwd(), cfg.gs.imgs_dir)
        cfg.data.train.img_prefix = cfg.data.val.img_prefix = imgs_dir    
            
        return anns_list, anns_config, cfg
    
    
    def download_dataset(cfg):
        train_dataset_path, val_dataset_path, cfg = download_record(cfg)       # download train, val dataset.json from google cloud storage
        anns_list, anns_config, cfg = download_images(cfg)
        
        with open(train_dataset_path, "r", encoding='utf-8') as f:
            train_dataset = json.load(f) 
        cfg.data.train.ann_file = train_dataset_path
        
        with open(val_dataset_path, "r", encoding='utf-8') as f:
            val_dataset = json.load(f) 
        cfg.data.val.ann_file = val_dataset_path  

        dataset_num, ann_num= len(train_dataset['images']) + len(val_dataset['images']), len(anns_list)
       
        assert dataset_num == ann_num, f"\n Invalid annotations or dataset loaded. \n number of dataset: {dataset_num}, number of annotations: {ann_num}"
        assert anns_config['ann_version'] == train_dataset['info']['ann_version'], \
            f"\n Invalid version for training. check dataset or annotations version \n \
              dataset_version: {train_dataset['info']['ann_version']}, anns_config['ann_version']: {anns_config['ann_version']}"
        
        for image_info_t in train_dataset["images"]:
            filepath_t = os.path.join(os.path.join(os.getcwd(), cfg.gs.imgs_dir), image_info_t['file_name'])
            image_info_t['file_name'] = filepath_t
        for image_info_v in val_dataset["images"]:
            filepath_v = os.path.join(os.path.join(os.getcwd(), cfg.gs.imgs_dir), image_info_v['file_name'])
            image_info_v['file_name'] = filepath_v
 
    
    def get_logger_set_meta(cfg):
        cfg.work_dir = os.path.join(os.getcwd(), cfg.model_version)
        os.makedirs(cfg.work_dir, exist_ok= True)           
        
        log_file = os.path.join(cfg.work_dir, f'{cfg.model_version}.log')
        logger = get_logger(name= cfg.pipeline.pipeline_name, log_file=log_file, log_level=cfg.log_level) 
        
        # log env info      
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info(f'\nEnvironment info:\n{dash_line}{env_info}\n{dash_line}\n')          # delete '#' if you want print Environment info  
        
        cfg.device = get_device() 
        meta = dict()
        meta['env_info'] = env_info
        meta['config'] = cfg.pretty_text
        
        seed = init_random_seed(cfg.seed, device=cfg.device)
        logger.info(f'\nSet random seed to {seed}, '
                f'deterministic: {cfg.deterministic}\n')
        
        set_random_seed(seed, deterministic=cfg.deterministic)
        cfg.seed = seed
        meta['seed'] = seed
        meta['exp_name'] = cfg_dict['cfg_name']
        return logger, log_file, meta
   
   
    
    if __name__=="__main__":
        cfg = load_config(cfg_dict)
        gs_credentials(cfg.gs.client_secrets)       # set client secrets
        download_dataset(cfg)
        
        logger, log_file_path, meta = get_logger_set_meta(cfg)
        
   
        datasets = [build_dataset(cfg.data.train)]      # <class 'mmdet.datasets.custom.CocoDataset'>
 
        if cfg.model.type =='MaskRCNN' :
            cfg.model.roi_head.bbox_head.num_classes = len(datasets[0].CLASSES)
            cfg.model.roi_head.mask_head.num_classes = len(datasets[0].CLASSES)   
            
        model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        
        if cfg.finetun is not None:  # fine tuning
            model.init_weights()
        
        if cfg.checkpoint_config is not None:   
            # save mmdet version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmdet_version=__version__ + get_git_hash()[:7],
                CLASSES=datasets[0].CLASSES)
        
        start_time = str(time.strftime('%Y%m%d_%H%M%S', time.localtime()))
        # train_detector
        train_detector(
            model,
            datasets,
            cfg,
            distributed= False,      
            validate= cfg.validate ,
            timestamp= start_time,
            meta=meta)
        
        gs_dir = cfg.model_version
        
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(cfg.gs.model_bucket_name)
        
        print(f"os.listdir(cfg.work_dir) : {os.listdir(cfg.work_dir)}")

        assert os.path.isfile(log_file_path)
        log_blob = bucket.blob(f"{gs_dir}/log_file.log")
        log_blob.upload_from_filename(log_file_path)
        
        log_json = os.path.join(os.getcwd(), cfg.work_dir, start_time) + ".log.json"
        assert os.path.isfile(log_json)
        jsonlog_blob = bucket.blob(f"{gs_dir}/train.log.json")
        jsonlog_blob.upload_from_filename(log_json)
        
        
        
        model_list = glob.glob(f"{cfg.work_dir}/*.pth")  
        for model_file in model_list:
            model_blob = bucket.blob(f"{gs_dir}/{model_file.split('/')[-1]}")
            model_blob.upload_from_filename(model_file)
        
    
        
    
print(f"train base_image : {pl_cfg.LOAD_DATA_IMAGE}")    
train_op = create_component_from_func(func = train,
                                        base_image = pl_cfg.LOAD_DATA_IMAGE,
                                        output_component_file= pl_cfg.LOAD_DATA_COM_FILE)