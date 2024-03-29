
from kfp.components import create_component_from_func, InputPath, OutputPath

from pipeline_base_config import Base_Image_cfg
base_image = Base_Image_cfg()

def train(cfg : dict, input_run_flag: InputPath("dict"),
          run_flag_path: OutputPath('dict')):       # Error when using variable name 'run_flag_path'
          
    import json
    import torch
    import psutil
    import numpy as np
    import glob
    import os, os.path as osp
    import pymysql
    import pandas as pd
    from google.cloud import storage
    from git.repo import Repo
    import sys
    
    WORKSPACE = dict(component_volume = cfg['path']['component_volume'],       # pvc volume path on component container
                     local_volume = cfg['path'].get('local_volume', None),     # pvc volume path on local
                     docker_volume = cfg['path'].get('docker_volume', None),   # volume 
                     package_dir = cfg['git'].get('package_repo', None),
                     work = cfg['path']['work_space']
                     )    


    # set package path to 'import {custom_package}'
    if __name__=="component.train.train_op":
        docker_volume = f"/{WORKSPACE['docker_volume']}"
        
        if WORKSPACE['local_volume'] is not None:
            local_module_path = osp.join('/opt/local-path-provisioner', WORKSPACE['local_volume']) 
        else:
            local_module_path = osp.join(os.getcwd(), cfg['git']['package_repo']) 
            
        package_dir = osp.join(os.getcwd(), WORKSPACE['package_dir'])       
        
        if osp.isdir(local_module_path):
            PACKAGE_PATH = os.getcwd()
            print(f"    Run `train` locally")
            
        elif osp.isdir(docker_volume):
            PACKAGE_PATH = docker_volume
            print(f"    Run `train` in docker container with volume")
        
        elif osp.isdir(package_dir):
            PACKAGE_PATH = os.getcwd()      # got packages with git clone
            print(f"    Run `train` in docker container")
            
        else:
            raise OSError(f"All paths to import packages do not exist!"
                          f"\n docker_volume: {docker_volume}"
                          f"\n package_dir: {package_dir}"
                          f"\n local_module_path:'{local_module_path}'")

    if __name__=="__main__":           
        if os.getcwd() != WORKSPACE['work']:
            workspace = WORKSPACE['work']
            os.makedirs(workspace, exist_ok = True)
            os.chdir(workspace)
         
        assert osp.isdir(WORKSPACE['work']), f"The path '{WORKSPACE['work']}' is not exist!"
        assert osp.isdir(WORKSPACE['component_volume']), f"The path '{WORKSPACE['component_volume']}' is not exist!"
        print(f"    Run `train` in component for pipeline")
        PACKAGE_PATH = WORKSPACE['component_volume']
        # for import sub_module
        package_repo_path = osp.join(WORKSPACE['component_volume'], cfg['git']['package_repo'])
        if not osp.isdir(package_repo_path):
            print(f" git clone 'sub_module' to {package_repo_path}")
            
            Repo.clone_from(f"git@github.com:HibernationNo1/{cfg['git']['package_repo']}.git", package_repo_path)
     
    sys.path.append(PACKAGE_PATH) 
        
    from sub_module.configs.pipeline import dict2Config, replace_config
    from sub_module.configs.config import Config
    from sub_module.database.mysql import check_table_exist
    from sub_module.cloud.google.storage import set_gs_credentials, get_client_secrets
    from sub_module.cloud.google.dvc import dvc_pull
    
    from sub_module.utils.utils import get_environ
    
    from sub_module.utils.log import LOGGERS, get_logger, collect_env_cuda
    from sub_module.mmdet.data.dataset import build_dataset
    from sub_module.mmdet.data.dataloader import build_dataloader
    from sub_module.mmdet.modules.dataparallel import build_dp
    from sub_module.mmdet.optimizer import build_optimizer
    from sub_module.mmdet.runner import build_runner
        
    def main(cfg, train_result, in_pipeline = False):    
        assert torch.cuda.is_available()
        set_model_config(cfg)
    
        os.makedirs(train_result, exist_ok=True)
        
        set_logs(cfg, train_result)
        cfg.seed = np.random.randint(2**31)
        cfg.device = "cuda"
        
        if in_pipeline:
            git_clone_dataset(cfg) 
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
            from sub_module.mmdet.modules.detector.maskrcnn import MaskRCNN
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
                                in_pipeline = in_pipeline,
                                katib = cfg.get("katib", False))
        train_runner = build_runner(runner_build_cfg)
        
        # get config about each hooks
        for hook_cfg in cfg.hook_configs:     
            if hook_cfg.type == 'CheckpointHook': 
                hook_cfg.model_cfg = cfg.model
                hook_cfg.out_dir = train_result
                
            if hook_cfg.type == 'Validation_Hook': 
                hook_cfg.val_dataloader = val_dataloader
                hook_cfg.model_cfg = cfg.model
                hook_cfg.logger = get_logger("validation")
                
                 
            if hook_cfg.type == 'TensorBoard_Hook' and in_pipeline: 
                hook_cfg.pvc_dir = osp.join(WORKSPACE['component_volume'], hook_cfg.pvc_dir) 
                                    
        train_runner.register_training_hooks(cfg.hook_configs)  

        
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
        
        return train_result
        

             
    def set_model_config(cfg):
        """
            Get the model configuration to run training with `Config.fromfile` 
            
            Why bring configuration with `Config.fromfile` rather then pass parameters in pipeline?
            >> The pipeline cannot run if the input parameter size exceeds 10,000.
            >> And the configuration of model occupies a lot of the input parameter size, rasing `size exceeds 10,000` Error.
        """
            
        package_name = cfg.git.get('package_repo', 'sub_module')
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
            base_sql = f"SELECT * FROM {table} WHERE train_version = '{version}'"
            df = dict()
            for perpose in ["train", "val"]:
                sql = base_sql + f" AND dataset_purpose = '{perpose}'"
                df[perpose] = pd.read_sql(sql, database)
            return df
    
        df_image = get_dataframe(cfg.db.table.image_data, cfg.dvc.record.version)
        # df_dataset = get_dataframe(cfg.db.table.dataset, cfg.dvc.record.version)
        
        val_data_cfg = cfg.data.val.copy()
        _ = val_data_cfg.pop("batch_size", None)       
        
        if not osp.isdir(cfg.data.train.data_root): 
            cfg.data.train.data_root = osp.join(os.getcwd(),
                                                cfg.git.dataset.repo,
                                                cfg.dvc.record.dir,
                                                df_image['train'].category[0])
        
        print(f"Number of images: {len(glob.glob(cfg.data.train.data_root +'/*.jpg'))}")
            
        assert osp.isdir(cfg.data.train.data_root), f"Path of train dataset dir is not exist! \npath: {cfg.data.train.data_root}"
        cfg.data.train.ann_file = osp.join(cfg.data.train.data_root,
                                        df_image['train'].record_file[0])
        assert osp.isfile(cfg.data.train.ann_file), f"Path of train dataset is not exist! \npath: {cfg.data.train.ann_file}"
      
        
        val_data_cfg.data_root = cfg.data.train.data_root
        val_data_cfg.ann_file = osp.join(val_data_cfg.data_root, 
                                        df_image['val'].record_file[0])
        assert osp.isfile(val_data_cfg.ann_file), f"Path of validaiton dataset is not exist! \npath: {val_data_cfg.ann_file}"
        dataset_cfg = dict(dataset_api = cfg.data.api,
                           train_cfg = cfg.data.train,
                           val_cfg = val_data_cfg)
        return dataset_cfg
        
    
    def load_dataset_from_dvc_db(cfg):
        dataset_dir = osp.join(os.getcwd(),
                               cfg.git.dataset.repo)
        
        data_root = osp.join(dataset_dir,
                             cfg.dvc.record.dir,
                             cfg.dvc.category)
        
        dvc_cfg = dict(remote = cfg.dvc.record.remote,
                       bucket_name = cfg.dvc.record.gs_bucket,
                       client_secrets = get_client_secrets(),
                       data_root = data_root,
                       dvc_path = osp.join(dataset_dir, ".dvc"))
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
        # get logger for environment
        env_logger = get_logger(cfg.log.env, log_level = cfg.log.level,
                                log_file = osp.join(train_result, f"{cfg.log.env}.log"))   
        
     
        env_info_dict = collect_env_cuda()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        # env_logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)           
        # env_logger.info(f'Config:\n{cfg.pretty_text}')
        
        # get logger for training
        get_logger(cfg.log.train, log_level = cfg.log.level,
                    log_file = osp.join(train_result, f"{cfg.log.train}.log"))      
            




    def upload_models(cfg, train_result_path):
        if not osp.isdir(train_result_path):
            raise OSError(f"Result of training path is not exist! \nPath: {train_result_path}")
        
        set_gs_credentials(get_client_secrets())
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(cfg.gs.result_bucket)
        
        dir_bucket = cfg.gs.train.get('path', None)  
        if dir_bucket is None:
            import time 
            # yyyy_mm_dd_hh_mm_ss
            dir_bucket = time.strftime('%Y_%m_%d_%H_%M_%S')  
        path_bucket = osp.join(cfg.gs.train.dir, dir_bucket) 

        models_dict, others_list = dict(accepted = [], important = []), []
        def set_upload_list(upload_path_list, path):
            # Accept only the pattern named model_{epoch}.pth 
            # or in `cfg.gs.train.model.important`
            for upload_path in upload_path_list:
                current_path = osp.join(path, upload_path)
                
                if osp.isdir(current_path):
                    set_upload_list(os.listdir(current_path), path = current_path)
                else:
                    formmat = osp.splitext(osp.basename(current_path))[1]
                    if formmat in cfg.gs.train.accept_formmat:
                        if formmat == '.pth':
                            try: 
                                _ = int(osp.basename(current_path).split('.')[0].split('_')[-1])      # for only `try`
                                models_dict['accepted'].append(current_path)
                            except:
                                if osp.basename(current_path) in cfg.gs.train.model.important:
                                    models_dict['important'].append(current_path)
                                else:
                                    print(f"\nUpload to GS: `{current_path}` is not accepted path!")
                                
                        else:
                            others_list.append(current_path)
                            
        set_upload_list(os.listdir(train_result_path), train_result_path)

        # Filter models to upload 
        sorted_models_list = sorted(models_dict['accepted'], 
                                    key=lambda x: int(osp.basename(x).split('.')[0].split('_')[-1]), 
                                    reverse=True)
        
        accept_model_list = list()
        for i, model_name in enumerate(sorted_models_list):
            if cfg.gs.train.model.min_epoch > int(osp.basename(model_name).split('.')[0].split('_')[-1]):
                break
            
            if i >= cfg.gs.train.model.count:
                break
            
            accept_model_list.append(model_name)
        accept_model_list += models_dict['important']
        
        # Upload models
        for accept_model in accept_model_list:
            # Convert path of the model to path that upload to bucket
            if WORKSPACE['component_volume'] in accept_model:
                model_path = accept_model.replace(WORKSPACE['component_volume'], "", 1)
            elif os.getcwd() in accept_model:
                model_path = accept_model.replace(os.getcwd(), "", 1)
                
            if os.path.isabs(model_path):
                model_path = model_path.lstrip('/')
                
            if len(model_path.split(f'{cfg.gs.train.dir}/'))>1:
                model_path = model_path.split(f'{cfg.gs.train.dir}/')[-1]
                
            gs_model_path = osp.join(path_bucket, model_path)
            print(f"\nUpload to GS: `{accept_model}` to google storage `{cfg.gs.result_bucket}` >> `{gs_model_path}`...")
    
            blob = bucket.blob(gs_model_path)
            blob.upload_from_filename(accept_model)
        
        # Upload others
        for others in others_list:
            if WORKSPACE['component_volume'] in others:
                other_path = others.replace(WORKSPACE['component_volume'], "", 1)
            elif os.getcwd() in others:
                other_path = others.replace(os.getcwd(), "", 1)
            
            if os.path.isabs(other_path):
                other_path = other_path.lstrip('/')
            
            if len(other_path.split(f'{cfg.gs.train.dir}/'))>1:
                other_path = other_path.split(f'{cfg.gs.train.dir}/')[-1]
                
            path_other_bucket = osp.join(path_bucket, other_path)
            
            print(f"\nUpload to GS: `{others}` to google storage `{path_other_bucket}`...")
           
            blob = bucket.blob(bucket_path)
            blob.upload_from_filename(others)
            
            
    
    def git_clone_dataset(cfg):
        repo_path = osp.join(WORKSPACE['work'], cfg.git.dataset.repo)
        if osp.isdir(repo_path):
            if len(os.listdir(repo_path)) != 0: 
                # ssh key not working when trying git pull with gitpython
                # delete all file cloned before and git clone again  
                import shutil
                shutil.rmtree(repo_path, ignore_errors=True)
                os.makedirs(repo_path, exist_ok=True)

        try:
            print(f"Run `$ git clone git@github.com:HibernationNo1/{cfg.git.dataset.repo}.git`")
            repo = Repo.clone_from(f'git@github.com:HibernationNo1/{cfg.git.dataset.repo}.git', repo_path)  
        except:
            print(f"Can't git clone with ssh!")
            print(f"Run `$ git clone https://github.com/HibernationNo1/{cfg.git.dataset.repo}.git`")
            repo = Repo.clone_from(f"https://github.com/HibernationNo1/{cfg.git.dataset.repo}.git", repo_path)

        remote_tags = repo.git.ls_remote("--tags").split("\n")
        tag_names = [tag.split('/')[-1] for tag in remote_tags if tag]
        if cfg.git.dataset.train.tag not in tag_names:
            raise KeyError(f"The `{cfg.git.dataset.train.tag}` is not exist in tags of repository `Hibernation/{cfg.git.dataset.repo}`")

        # checkout HEAD to tag
        repo.git.checkout(cfg.git.dataset.train.tag)
        
        return repo
            
            
    
    if __name__=="component.train.train_op":   
        cfg = Config(cfg)
        
        main(cfg, osp.join(os.getcwd(), cfg.train_result))

        
    if __name__=="__main__": 
        with open(input_run_flag, "r", encoding='utf-8') as f:
            input_run_flag = json.load(f) 
        if 'train' in input_run_flag['pipeline_run_flag']:
            cfg = dict2Config(cfg, key_name ='flag_list2tuple')          
            
            train_result = main(cfg, 
                                osp.join(WORKSPACE['component_volume'], cfg.train_result),
                                in_pipeline = True)    
            if cfg.gs.upload:    
                upload_models(cfg, train_result)
        else:
            print(f"Pass component: train")
        
        return json.dump(input_run_flag, open(run_flag_path, "w"), indent=4)
        


        


        
        
      
     
        
            
train_op = create_component_from_func(func = train,
                                        base_image = base_image.train,
                                        output_component_file= base_image.train_cp)
