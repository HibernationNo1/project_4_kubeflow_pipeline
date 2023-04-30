from kfp.components import create_component_from_func, InputPath, OutputPath

from pipeline_base_config import Base_Image_cfg
base_image = Base_Image_cfg()

def evaluate(cfg : dict, input_run_flag: InputPath("dict"),
             run_flag_path: OutputPath('dict')):       # Error when using variable name 'run_flag_path'
    
    import os, os.path as osp
    import sys
    import pymysql
    import json
    import pandas as pd
    from git.repo import Repo
    from google.cloud import storage

    WORKSPACE = dict(component_volume = cfg['path']['component_volume'],       # pvc volume path on component container
                     local_volume = cfg['path'].get('local_volume', None),     # pvc volume path on local
                     docker_volume = cfg['path']['docker_volume'],     # volume path on katib container
                     work = cfg['path']['work_space']
                     )    

    # set package path to 'import {custom_package}'
    if __name__=="component.evaluate.evaluate_op":
        docker_volume = f"/{WORKSPACE['docker_volume']}"
        
        if WORKSPACE['local_volume'] is not None:
            local_module_path = osp.join('/opt/local-path-provisioner', WORKSPACE['local_volume']) 
        else:
            local_module_path = osp.join(os.getcwd(), cfg['git']['package_repo'])       
        
        if osp.isdir(local_module_path):
            PACKAGE_PATH = os.getcwd()
            print(f"    Run `evaluate` locally")
            
        elif osp.isdir(docker_volume):
            PACKAGE_PATH = docker_volume
            print(f"    Run `evaluate` in docker container")
            
        else:
            raise OSError(f"Paths '{docker_volume}' and '{local_module_path}' do not exist!")


    if __name__=="__main__":    
        assert osp.isdir(WORKSPACE['work']), f"The path '{WORKSPACE['work']}' is not exist!"
        assert osp.isdir(WORKSPACE['component_volume']), f"The path '{WORKSPACE['component_volume']}' is not exist!"
        print(f"    Run `evaluate` in component for pipeline")
        PACKAGE_PATH = WORKSPACE['component_volume']
        # for import sub_module
        package_repo_path = osp.join(WORKSPACE['component_volume'], cfg['git']['package_repo'])
        if not osp.isdir(package_repo_path):
            print(f" git clone 'sub_module' to {package_repo_path}")
            
            Repo.clone_from(f"git@github.com:HibernationNo1/{cfg['git']['package_repo']}.git", package_repo_path)
     
    sys.path.append(PACKAGE_PATH)   

    from sub_module.configs.pipeline import dict2Config
    from sub_module.configs.config import Config
    from sub_module.database.mysql import check_table_exist
    from sub_module.cloud.google.storage import set_gs_credentials, get_client_secrets
    from sub_module.cloud.google.dvc import dvc_pull
    from sub_module.mmdet.inference import build_detector
    from sub_module.mmdet.modules.dataparallel import build_dp
    from sub_module.mmdet.visualization import draw_to_img
    
    from sub_module.utils.utils import get_environ

    from sub_module.mmdet.data.dataset import build_dataset
    from sub_module.mmdet.data.dataloader import build_dataloader
    from sub_module.mmdet.eval import Evaluate

    import numpy as np
    
    
    
    
    def main(cfg, eval_result, in_pipeline = False):
        cfg.seed = np.random.randint(2**31)

        if in_pipeline:
            git_clone_dataset(cfg) 
            _, val_dataset = build_dataset(**set_dataset_cfg(cfg, load_dataset_from_dvc_db(cfg)))
        else:
            val_cfg = cfg.data.val.copy()
            val_cfg.pop("batch_size")
            _, val_dataset = build_dataset(val_cfg = val_cfg)
        
        
        dataloader_cfg = dict(val_dataset = val_dataset,
                            val_batch_size = cfg.data.val.batch_size,
                            num_workers=cfg.data.workers_per_gpu,
                            seed = cfg.seed,
                            shuffle = True)
        _, val_dataloader = build_dataloader(**dataloader_cfg)
        
        if in_pipeline:
            if cfg.gs.download:
                model_dir = download_model(cfg)
            else:
                model_dir = osp.join(WORKSPACE['component_volume'], cfg.train_result)
        else:
            if cfg.get('model_path', None) is None:
                model_dir = osp.join(os.getcwd(), cfg.train_result)
            else:
                model_dir = osp.join(os.getcwd(), osp.dirname(cfg.model_path))
        
        model_path_list = list()
        def get_model_list(dir_path):
            # Get path only models
            if osp.isdir(dir_path):
                for path_ in os.listdir(dir_path):
                    get_model_list(osp.join(dir_path, path_))       
            elif osp.isfile(dir_path):
                if osp.splitext(dir_path)[-1] == ".pth":
                    model_path_list.append(dir_path)
        get_model_list(model_dir)   
        
        best_model = dict()
        for key_name in cfg.key_name:
            best_model[key_name] = dict(path = None,
                                        value = -1,
                                        output_path = None)
        
        for model_path in model_path_list:
            result_dir = osp.join(eval_result, osp.splitext(osp.basename(model_path))[0]) 
            os.makedirs(result_dir, exist_ok = True) 

            model = build_detector(cfg, model_path, device = cfg.device)
            dp_cfg = dict(model = model, 
                        cfg = cfg,
                        device = cfg.device,
                        classes = model.CLASSES)
            model = build_dp(**dp_cfg)


            eval_cfg = dict(model= model, 
                            output_path = result_dir,
                            cfg= cfg.eval_cfg,
                            dataloader= val_dataloader)
            eval_ = Evaluate(**eval_cfg)   
            summary = eval_.get_mAP()
            summary['dv_mAP'] = summary['dv']["mAP"]
            summary['mAP'] = summary['normal']["mAP"]
            
            exact_inference_rate = eval_.run_inference(compare_board = True)
            if exact_inference_rate is None:
                json.dump(summary, open(osp.join(result_dir, "summary.json"), "w"), indent=4)
            else: 
                summary['EIR'] = exact_inference_rate
                json.dump(summary, open(osp.join(result_dir, "summary.json"), "w"), indent=4)
            
            print(f"\nResult - model name: {osp.basename(model_path)}")
            for key_name, key_dict in best_model.items():
                key_value = summary.get(key_name, None)
                if key_value is None: continue
                else:
                    print(f"    >> {key_name}: {key_value}")
                    
                    if key_dict['value'] < key_value:
                        best_model[key_name]['value'] = key_value
                        best_model[key_name]['path'] = model_path
                        best_model[key_name]['result_dir'] = result_dir
        
        if in_pipeline:
        	upload_model(cfg, best_model)
  

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
        
        if not osp.isdir(cfg.data.val.data_root): 
            cfg.data.val.data_root = osp.join(os.getcwd(),
                                                cfg.git.dataset.repo,
                                                cfg.dvc.record.dir,
                                                df_image['train'].category[0])
                    
        val_data_cfg.ann_file = osp.join(val_data_cfg.data_root, 
                                        df_image['val'].record_file[0])
        assert osp.isfile(val_data_cfg.ann_file), f"Path of validaiton dataset is not exist! \npath: {val_data_cfg.ann_file}"
        dataset_cfg = dict(dataset_api = cfg.data.api,
                           val_cfg = val_data_cfg)
        return dataset_cfg
    
    
    def upload_model(cfg, best_model):
        for key_name, key_dict in best_model.items():
            if key_dict['path'] is None: continue
            
            model_path = key_dict['path']            
            if not osp.isfile(model_path):
                print(f"The path of model is not vaild. \n>> Path: {model_path}")
                return None
            
            summary_file_path = osp.join(key_dict['result_dir'], 'summary.json')
            if not osp.isfile(summary_file_path):
                print(f"Invaild path: {summary_file_path}")
                return None
            
            print(f"model name: {osp.basename(model_path)}      key_name : {key_name}       value: {key_dict['value']}")
            
            set_gs_credentials(get_client_secrets())
            storage_client = storage.Client()
            bucket = storage_client.get_bucket(cfg.gs.result_bucket)
            
            dir_bucket = cfg.gs.eval.get('path', None)  
            if dir_bucket is None:
                import time 
                # yyyy_mm_dd_hh_mm_ss
                dir_bucket = time.strftime('%Y_%m_%d_%H_%M_%S')  
            gs_path = osp.join(cfg.gs.eval.dir, dir_bucket, key_name) 
            
            # upload model to gs
            model_gs_path = osp.join(gs_path, osp.basename(model_path))
            print(f"Upload model from `{model_path}` to `{model_gs_path}`")
            
            blob = bucket.blob(model_gs_path)
            blob.upload_from_filename(model_path)    
            
            # upload summary file
            summary_file_gs_path = osp.join(gs_path, osp.basename(summary_file_path))
            print(f"Upload model from `{summary_file_path}` to `{summary_file_gs_path}`")
            
            blob = bucket.blob(summary_file_gs_path)
            blob.upload_from_filename(summary_file_path)    
            
                
            
    
    def download_model(cfg):
        set_gs_credentials(get_client_secrets())
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(cfg.gs.result_bucket)
        
        if cfg.gs.eval.get('target', None) is None:
            gs_path = cfg.gs.train.dir
        else:
            gs_path = osp.join(cfg.gs.train.dir, cfg.gs.eval.target)
        
        blobs = bucket.list_blobs(prefix=gs_path)
        model_dir = osp.join(os.getcwd(), cfg.gs.eval.get('download_dir', "download"))   
        for blob in blobs:
            bucket_path = blob.name[len(gs_path):]
            if osp.isabs(bucket_path):
                bucket_path = bucket_path.lstrip('/')
            local_file_path = os.path.join(model_dir, bucket_path)
            print(f"Download From `gs:{cfg.gs.result_bucket}/{bucket_path}` to `{local_file_path}`")
            os.makedirs(osp.dirname(local_file_path), exist_ok = True)
            blob.download_to_filename(local_file_path)

        return model_dir
    
    
    if __name__=="component.evaluate.evaluate_op":  
        cfg = Config(cfg)
        main(cfg, osp.join(os.getcwd(), cfg.eval_result))
        
        
    if __name__=="__main__":    
        with open(input_run_flag, "r", encoding='utf-8') as f:
            input_run_flag = json.load(f) 

        if 'evaluate' in input_run_flag['pipeline_run_flag']:
            cfg = dict2Config(cfg, key_name ='flag_list2tuple')    
            main(cfg, osp.join(WORKSPACE['component_volume'], cfg.eval_result), in_pipeline = True)
        else:
            print(f"Pass component: evaluate")
        
        return json.dump(input_run_flag, open(run_flag_path, "w"), indent=4)

   
evaluate_op = create_component_from_func(func = evaluate,
                                         base_image = base_image.evaluate,
                                         output_component_file= base_image.evaluate_cp)