
from kfp.components import create_component_from_func, OutputPath

from pipeline_base_config import Base_Image_cfg
base_image = Base_Image_cfg()

def train(cfg : dict):    
    import torch
    import psutil
    import numpy as np
    import os, os.path as osp
    import pymysql
    import pandas as pd
    from google.cloud import storage
    from git.repo import Repo
    import sys
    
    WORKSPACE = dict(volume = cfg['path']['volume'],       # pvc volume path
                     work =  cfg['path']['work_space'],     # path if workspace in docker container
                     local_package = cfg['path']['local_volume'])    

    if __name__=="train.train_op": 
        assert osp.isdir(WORKSPACE['local_package']), f"The path '{WORKSPACE['local_package']}' is not exist!"
        sys.path.append(f"{WORKSPACE['local_package']}")    
              
    if __name__=="__main__":    
        assert osp.isdir(WORKSPACE['work']), f"The path '{WORKSPACE['work']}' is not exist!"
        assert osp.isdir(WORKSPACE['volume']), f"The path '{WORKSPACE['volume']}' is not exist!"
        # for import hibernation_no1
        package_path = osp.join(WORKSPACE['volume'], cfg['git_repo']['package'])
        if not osp.isdir(package_path):
            print(f" git clone 'hibernation_no1' to {package_path}")
            Repo.clone_from(f"git@github.com:HibernationNo1/{cfg['git_repo']['package']}.git", package_path)
        
        sys.path.append(f"{WORKSPACE['volume']}")    
        
 
    from hibernation_no1.configs.utils import change_to_tuple
    from hibernation_no1.configs.config import Config
    from hibernation_no1.database.mysql import check_table_exist
    from hibernation_no1.cloud.google.storage import set_gs_credentials, get_client_secrets
    from hibernation_no1.cloud.google.dvc import dvc_pull
    
    from hibernation_no1.utils.utils import get_environ
    
    from hibernation_no1.utils.log import LOGGERS, get_logger, collect_env_cuda
    from hibernation_no1.mmdet.data.dataset import build_dataset
    from hibernation_no1.mmdet.data.dataloader import build_dataloader
    from hibernation_no1.mmdet.modules.dataparallel import build_dp
    from hibernation_no1.mmdet.optimizer import build_optimizer
    from hibernation_no1.mmdet.runner import build_runner
            
        
    def main(cfg, in_pipeline = False):    
        assert torch.cuda.is_available()
        train_result = osp.join(os.getcwd(), cfg.train_result)
        os.makedirs(train_result, exist_ok=True)
        
        set_logs(cfg, train_result)
        cfg.seed = np.random.randint(2**31)
        cfg.device = "cuda"
        
        if in_pipeline:
            train_dataset, val_dataset = build_dataset(**set_dataset_cfg(cfg, load_dataset_from_dvc_db(cfg)))
        else:
            val_cfg = cfg.data.val.copy()
            val_cfg.pop("batch_size")
            dataset_cfg = dict(train_cfg = cfg.data.train, val_cfg = val_cfg)
            train_dataset, val_dataset = build_dataset(**dataset_cfg)
        
         
        dataloader_cfg = dict(train_dataset = train_dataset,
                                val_dataset = val_dataset,
                                train_batch_size=cfg.data.samples_per_gpu,
                                val_batch_size = cfg.data.val.batch_size,
                                num_workers=cfg.data.workers_per_gpu,
                                seed = cfg.seed,
                                shuffle = True)
        train_dataloader, val_dataloader = build_dataloader(**dataloader_cfg)
        
        # build model
        assert cfg.get('train_cfg') is None , 'train_cfg must be specified in both outer field and model field'
        
        # TODO : build with registry
        if cfg.model.type == 'MaskRCNN':
            cfg.model.roi_head.bbox_head.num_classes = len(train_dataset.CLASSES) 
            cfg.model.roi_head.mask_head.num_classes = len(train_dataset.CLASSES)
            
            model_cfg = cfg.model.copy()
        
            model_cfg.pop("type")
            from hibernation_no1.mmdet.modules.detector.maskrcnn import MaskRCNN
            model = MaskRCNN(**model_cfg)
        
        
        dp_cfg = dict(model = model, 
                      device = cfg.device,
                      cfg = cfg,
                      classes = train_dataset.CLASSES)
        model = build_dp(**dp_cfg)
        
        optimizer = build_optimizer(model, cfg, LOGGERS[cfg.log.train]['logger'])          
        
        # build runner
        runner_build_cfg = dict(model = model,
                                optimizer = optimizer,
                                work_dir = train_result,
                                logger = LOGGERS[cfg.log.train]['logger'],
                                meta = dict(config = cfg.pretty_text, seed = cfg.seed),
                                batch_size = cfg.data.samples_per_gpu,
                                max_epochs = cfg.max_epochs,
                                iterd_per_epochs = len(train_dataloader),
                                in_pipeline = in_pipeline)
        train_runner = build_runner(runner_build_cfg)
        
        # get config about each hooks
        for hook_cfg in cfg.hook_config:     
            if hook_cfg.type == 'CheckpointHook': 
                hook_cfg.model_cfg = cfg.model
                
            if hook_cfg.type == 'Validation_Hook': 
                hook_cfg.val_dataloader = val_dataloader
                hook_cfg.logger = get_logger("validation")
                hook_cfg.result_dir = train_result
                
                 
            if hook_cfg.type == 'TensorBoard_Hook' and in_pipeline: 
                hook_cfg.pvc_dir = osp.join(WORKSPACE['volume'], hook_cfg.pvc_dir) 
                                    
        train_runner.register_training_hooks(cfg.hook_config)  

        
        resume_from = cfg.get('resume_from', None)
        load_from = cfg.get('load_from', None)
        
        # TODO
        if resume_from is not None:
            train_runner.resume(cfg.resume_from)
        elif cfg.load_from:
            train_runner.load_checkpoint(cfg.load_from)

        # cfg.val.mask2polygon = mask_to_polygon 
        run_cfg = dict(train_dataloader = train_dataloader)
                       # val_dataloader = val_dataloader

        train_runner.run(**run_cfg)
        

             
        
        
    
    def set_dataset_cfg(cfg, database):
        def get_dataframe(table, version):
            base_sql = f"SELECT * FROM {table} WHERE recode_version = '{version}'"
            df = dict()
            for perpose in ["train", "val"]:
                sql = base_sql + f" AND dataset_purpose = '{perpose}'"
                df[perpose] = pd.read_sql(sql, database)
            return df
    
        df_image = get_dataframe(cfg.db.table.image_data, cfg.dvc.recode.version)
        # df_dataset = get_dataframe(cfg.db.table.dataset, cfg.dvc.recode.version)
        
        val_data_cfg = cfg.data.val.copy()
        _ = val_data_cfg.pop("batch_size", None)       # TODO: check batch_size is not using 
        
        cfg.data.train.data_root = osp.join(os.getcwd(),
                                            df_image['train'].category[0], 
                                            "recode", 
                                            df_image['train'].recode_version[0])
        cfg.data.train.ann_file = osp.join(cfg.data.train.data_root,
                                        df_image['train'].recode_file[0])
        
        
        val_data_cfg.data_root = osp.join(os.getcwd(), 
                                        df_image['val'].category[0], 
                                        "recode", 
                                        df_image['val'].recode_version[0])
        val_data_cfg.ann_file = osp.join(cfg.data.train.data_root, 
                                        df_image['val'].recode_file[0])
    
        dataset_cfg = dict(dataset_api = cfg.data.api,
                        train_cfg = cfg.data.train,
                        val_cfg = val_data_cfg)
        return dataset_cfg
        
    
    def load_dataset_from_dvc_db(cfg):
        data_root = osp.join(os.getcwd(), cfg.dvc.category,
                                            cfg.dvc.recode.name,
                                            cfg.dvc.recode.version)
        
        dvc_cfg = dict(remote = cfg.dvc.recode.remote,
                       bucket_name = cfg.dvc.recode.gs_bucket,
                       client_secrets = get_client_secrets(),
                       data_root = data_root)
        dvc_pull(**dvc_cfg)

        database = pymysql.connect(host=get_environ(cfg.db, 'host'), 
                        port=int(get_environ(cfg.db, 'port')), 
                        user=cfg.db.user, 
                        passwd=os.environ['password'], 
                        database=cfg.db.name, 
                        charset=cfg.db.charset)
        
        cursor = database.cursor() 
        check_table_exist(cursor, [cfg.db.table.image_data, cfg.db.table.dataset])  
        return database
            
            
    def set_logs(cfg, train_result):
        env_logger = get_logger(cfg.log.env, log_level = cfg.log.level,
                                log_file = osp.join(train_result, f"{cfg.log.env}.log"))       # TODO: save log file
        env_info_dict = collect_env_cuda()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        # env_logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)           
        # env_logger.info(f'Config:\n{cfg.pretty_text}')
        
        get_logger(cfg.log.train, log_level = cfg.log.level,
                    log_file = osp.join(train_result, f"{cfg.log.train}.log"))      # TODO: save log file
            
            
    
    def dict2Config(cfg):
        cfg_flag = cfg.get('flag', None)
        if cfg_flag is not None:
            cfg = change_to_tuple(cfg, cfg_flag)
        cfg = Config(cfg)
        return cfg
    
    
    def upload_models(cfg):
        set_gs_credentials(get_client_secrets())
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(cfg.gs.models_bucket)

        dir_bucket = cfg.gs.get('path', None)  
        if dir_bucket is None:
            import time 
            # yyyy_mm_dd_hh_mm
            dir_bucket = time.strftime('%Y-%m-%d', time.localtime(time.time())) \
                        + " "+ str(time.localtime(time.time()).tm_hour) \
                        + ":" + str(time.localtime(time.time()).tm_min)     
        
        # sort file to upload non-models first
        result_list = os.listdir(osp.join(os.getcwd(), cfg.train_result))
        upload_list = []
        model_list = list()
        for i, result in enumerate(result_list):
            if result.split("_")[0] == 'model':
                continue        ###
                model_list.append(result)
            else:
                upload_list.append(result)
            
            if i == len(result_list)-1:
                for model in model_list:
                    upload_list.append(model)
                            
        for file_name in upload_list:
            print(f"upload {file_name} to google storage...")
            blob = bucket.blob(os.path.join(dir_bucket, file_name))
            blob.upload_from_filename(osp.join(cfg.train_result, file_name))
            
    
    def git_clone_dataset(cfg):
        repo_path = osp.join(WORKSPACE['work'], cfg.git_repo.dataset)
        if len(os.listdir(repo_path)) != 0:
            # ----
            # repo = Repo(osp.join(WORKSPACE['work'], cfg.git_repo.dataset))
            # origin = repo.remotes.origin  
            # repo.config_writer().set_value("user", "email", "taeuk4958@gmail.com").release()
            # repo.config_writer().set_value("user", "name", "HibernationNo1").release()
            
            # import subprocess       # this command working only shell, not gitpython.
            # safe_directory_str = f"git config --global --add safe.directory {repo_path}"
            # subprocess.call([safe_directory_str], shell=True)      

            # # raise: stderr: 'fatal: could not read Username for 'https://github.com': No such device or address'  
            # origin.pull()   
            # ----
            
            # ssh key not working when trying git pull with gitpython
            # delete all file cloned before and git clone again  
            import shutil
            shutil.rmtree(repo_path, ignore_errors=True)
            os.makedirs(repo_path, exist_ok=True)

        Repo.clone_from(f'git@github.com:HibernationNo1/{cfg.git_repo.dataset}.git', os.getcwd())  
            
            
    
    if __name__=="train.train_op":        
        cfg = dict2Config(cfg)
        main(cfg)
        
        
    if __name__=="__main__":       
        cfg = dict2Config(cfg)       
         
        git_clone_dataset(cfg)              
        main(cfg, in_pipeline = True)        
        upload_models(cfg)
        
      
     
        
            
train_op = create_component_from_func(func = train,
                                        base_image = base_image.train,
                                        output_component_file= base_image.train_cp)
