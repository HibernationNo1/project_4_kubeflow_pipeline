
from kfp.components import create_component_from_func, InputPath, OutputPath

from pipeline_base_config import Base_Image_cfg
base_image = Base_Image_cfg()

def train(cfg : dict, input_run_flag: InputPath("dict"),
          run_flag_path: OutputPath('dict')):       # Error when using variable name 'run_flag_path'
          
    import json
    import torch
    import psutil
    import numpy as np
    import os, os.path as osp
    import pymysql
    import pandas as pd
    from google.cloud import storage
    from git.repo import Repo
    import sys
    
    WORKSPACE = dict(component_volume = cfg['path']['component_volume'],       # pvc volume path on component container
                     local_volume = cfg['path']['local_volume'],     # pvc volume path on local
                     katib_volume = cfg['path']['katib_volume'],     # volume path on katib container
                     work = cfg['path']['work_space']
                     )    

    # set package path to 'import {custom_package}'
    if __name__=="component.train.train_op":
        local_volume = osp.join('/opt/local-path-provisioner', WORKSPACE['local_volume']) 
        katib_volume = f"/{WORKSPACE['katib_volume']}"
        if osp.isdir(local_volume):
            PACKAGE_PATH = local_volume
            print(f"    Run `train` locally")
            
        elif osp.isdir(katib_volume):
            PACKAGE_PATH = katib_volume
            print(f"    Run `train` in container for katib")
            
        else:
            raise OSError(f"Paths '{katib_volume}' and '{local_volume}' do not exist!")

    if __name__=="__main__":    
        assert osp.isdir(WORKSPACE['work']), f"The path '{WORKSPACE['work']}' is not exist!"
        assert osp.isdir(WORKSPACE['component_volume']), f"The path '{WORKSPACE['component_volume']}' is not exist!"
        print(f"    Run `train` in component for pipeline")
        PACKAGE_PATH = WORKSPACE['component_volume']
        # for import hibernation_no1
        package_repo_path = osp.join(WORKSPACE['component_volume'], cfg['git']['package_repo'])
        if not osp.isdir(package_repo_path):
            print(f" git clone 'hibernation_no1' to {package_repo_path}")
            
            Repo.clone_from(f"git@github.com:HibernationNo1/{cfg['git']['package_repo']}.git", package_repo_path)
     
    sys.path.append(PACKAGE_PATH) 
        
    from hibernation_no1.configs.pipeline import dict2Config, replace_config
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
        set_model_config(cfg)
        
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
        

             
    def set_model_config(cfg):
        """
            Get the model configuration to run training with `Config.fromfile` 
            
            Why bring configuration with `Config.fromfile` rather then pass parameters in pipeline?
            >> The pipeline cannot run if the input parameter size exceeds 10,000.
            >> And the configuration of model occupies a lot of the input parameter size, rasing `size exceeds 10,000` Error.
        """
            
        package_name = cfg.git.get('package_repo', 'hibernation_no1')
        package_dir = osp.join(PACKAGE_PATH, package_name)    
        model_cfg_dir = osp.join(package_dir, 'mmdet', 'modules', 'configs')
        
        assert osp.isdir(model_cfg_dir), f"Path of model config files dose not exist!! \n   Path: {model_cfg_dir}"
        
        main_module_cfg = f"{cfg.model.get('type', 'MaskRCNN')}.py"
        backbone_module_cfg =  f"{cfg.model.backbone.get('type', 'SwinTransformer')}.py"
        neck_module_cfg =  f"{cfg.model.neck.get('type', 'FPN')}.py"

        main_cfg_file = osp.join(model_cfg_dir, main_module_cfg)
        backbone_cfg_file = osp.join(model_cfg_dir, 'backbone', backbone_module_cfg)
        neck_cfg_file = osp.join(model_cfg_dir, 'neck', neck_module_cfg)

        if osp.isfile(main_cfg_file) and\
            osp.isfile(backbone_cfg_file) and\
            osp.isfile(neck_cfg_file):


            model_cfg = Config.fromfile(main_cfg_file).model
            model_cfg.backbone = dict(Config.fromfile(backbone_cfg_file).backbone)
            model_cfg.neck = dict(Config.fromfile(neck_cfg_file).neck)
        else:
            assert osp.isfile(main_cfg_file), f"Path of model config file dose not exist!! \n   Path: {main_cfg_file}"
            assert osp.isfile(backbone_cfg_file), f"Path of model config file dose not exist!! \n   Path: {backbone_cfg_file}"
            assert osp.isfile(neck_cfg_file), f"Path of model config file dose not exist!! \n   Path: {neck_cfg_file}"
        

        replaced_cfg = replace_config(from_cfg = model_cfg, to_cfg = cfg.model, init=True)
        if replaced_cfg is None:
            raise KeyError(f"cfg.model.type is not same as model_cfg.type!! check configuration."
                           f"\n cfg.model.type: {cfg.model.type},  model_cfg.type: {model_cfg.type}")

        cfg.model = replaced_cfg     
        
    
    def set_dataset_cfg(cfg, database):
        def get_dataframe(table, version):
            base_sql = f"SELECT * FROM {table} WHERE record_version = '{version}'"
            df = dict()
            for perpose in ["train", "val"]:
                sql = base_sql + f" AND dataset_purpose = '{perpose}'"
                df[perpose] = pd.read_sql(sql, database)
            return df
    
        df_image = get_dataframe(cfg.db.table.image_data, cfg.dvc.record.version)
        # df_dataset = get_dataframe(cfg.db.table.dataset, cfg.dvc.record.version)
        
        val_data_cfg = cfg.data.val.copy()
        _ = val_data_cfg.pop("batch_size", None)       # TODO: check batch_size is not using 
        
        cfg.data.train.data_root = osp.join(os.getcwd(),
                                            df_image['train'].category[0], 
                                            "record", 
                                            df_image['train'].record_version[0])
        cfg.data.train.ann_file = osp.join(cfg.data.train.data_root,
                                        df_image['train'].record_file[0])
        
        
        val_data_cfg.data_root = osp.join(os.getcwd(), 
                                        df_image['val'].category[0], 
                                        "record", 
                                        df_image['val'].record_version[0])
        val_data_cfg.ann_file = osp.join(cfg.data.train.data_root, 
                                        df_image['val'].record_file[0])
    
        dataset_cfg = dict(dataset_api = cfg.data.api,
                        train_cfg = cfg.data.train,
                        val_cfg = val_data_cfg)
        return dataset_cfg
        
    
    def load_dataset_from_dvc_db(cfg):
        data_root = osp.join(os.getcwd(), cfg.dvc.category,
                                            cfg.dvc.record.name,
                                            cfg.dvc.record.version)
        
        dvc_cfg = dict(remote = cfg.dvc.record.remote,
                       bucket_name = cfg.dvc.record.gs_bucket,
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
        repo_path = osp.join(WORKSPACE['work'], cfg.git.dataset_repo)
        if len(os.listdir(repo_path)) != 0:
            # ----
            # repo = Repo(osp.join(WORKSPACE['work'], cfg.git.dataset_repo))
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

        Repo.clone_from(f'git@github.com:HibernationNo1/{cfg.git.dataset_repo}.git', os.getcwd())  
            
            
    
    if __name__=="component.train.train_op":   
        cfg = Config(cfg)
        main(cfg)

        
        
        
    if __name__=="__main__": 
        with open(input_run_flag, "r", encoding='utf-8') as f:
            input_run_flag = json.load(f) 

        if 'train' in input_run_flag['pipeline_run_flag']:
            cfg = dict2Config(cfg, key_name ='flag_list2tuple')       
         
            # git_clone_dataset(cfg)              
            # main(cfg, in_pipeline = True)        
            # upload_models(cfg)
        else:
            print(f"Pass component: train")
        
        return json.dump(input_run_flag, open(run_flag_path, "w"), indent=4)
        


        


        
        
      
     
        
            
train_op = create_component_from_func(func = train,
                                        base_image = base_image.train,
                                        output_component_file= base_image.train_cp)
