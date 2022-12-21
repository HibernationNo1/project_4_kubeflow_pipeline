
from kfp.components import create_component_from_func

from pipeline_base_image_cfg import Base_Image_Cfg
base_image = Base_Image_Cfg()


def train(cfg : dict):
    
    import numpy as np
    import os, os.path as osp
    import pymysql
    import pandas as pd
    
    from hibernation_no1.configs.utils import change_to_tuple
    from hibernation_no1.configs.config import Config
    from hibernation_no1.database.mysql import check_table_exist
    from hibernation_no1.cloud.google.storage import get_client_secrets
    from hibernation_no1.cloud.google.dvc import dvc_pull
    
    from hibernation_no1.utils.utils import get_environ
    
    
    from hibernation_no1.utils.log import logger, get_logger
    from hibernation_no1.mmdet.data.dataset import build_dataset
    
    
    
    
    if __name__=="__main__":
        cfg_flag = cfg.pop('flag')
        cfg = change_to_tuple(cfg, cfg_flag)
        cfg = Config(cfg)
        
        print(f"cfg : {cfg}")
        exit()
        data_root = osp.join(os.getcwd(), cfg.dvc.category,
                                               cfg.dvc.ann.name,
                                               cfg.dvc.ann.version)
        
        dvc_cfg = dict(remote = cfg.dvc.remote,
                       bucket_name = cfg.gs.bucket.recoded,
                       client_secrets = get_client_secrets(),
                       data_root = data_root)
        dataset_dir_path = dvc_pull(**dvc_cfg)
        
        print(f"os.listdir(os.getcwd()) ; {os.listdir(os.getcwd())}")
      
        database = pymysql.connect(host=get_environ(cfg.db, 'host'), 
                        port=int(get_environ(cfg.db, 'port')), 
                        user=cfg.db.user, 
                        passwd=os.environ['password'], 
                        database=cfg.db.name, 
                        charset=cfg.db.charset)
        
        cursor = database.cursor() 
        check_table_exist(cursor, [cfg.db.table.image_data, cfg.db.table.dataset])
        
        base_sql = f"SELECT * FROM {cfg.db.table.image_data} WHERE recode_version = '{cfg.dvc.recode.version}'"
        train_sql = base_sql + f" AND dataset_purpose = 'train'"
        val_sql = base_sql + f" AND dataset_purpose = 'val'"
        
        train_df = pd.read_sql(train_sql, database)
        val_df = pd.read_sql(train_sql, database)
        
        print(f"len(train_df.image_name) : {len(train_df.image_name)}")
        print(f"train_df.image_name[0] : {train_df.image_name[0]}")
        exit()
        
        cfg.seed = np.random.randint(2**31)
        cfg.device = "cuda"    
        
        
        get_logger('enviroments', log_leval = cfg.log_level)
        get_logger(f'training', log_leval = cfg.log_level)
        print(f"logger")
        print(logger)
        # check log
        
 
        val_data_cfg = cfg.data.val.copy()
        _ = val_data_cfg.pop("batch_size", None)       # TODO: check batch_size is not using 
        dataset_cfg = dict(dataset_api = cfg.data.api,
                           train_cfg = cfg.data.train,
                           val_cfg = val_data_cfg)
        train_dataset, val_dataset = build_dataset(**dataset_cfg)
        
        # for epoch in cfg.epoch:
            
        
    
train_op = create_component_from_func(func = train,
                                        base_image = base_image.train,
                                        output_component_file= base_image.train_cp)
