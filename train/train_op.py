
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
    
    
    from hibernation_no1.utils.log import LOGGERS, get_logger
    from hibernation_no1.mmdet.data.dataset.dataset import build_dataset
    from hibernation_no1.mmdet.data.dataset.dataloader import build_dataloader
    
    # def get_images(dataset_df):
    #     category = dataset_df.category[0]
    #     recode_version = dataset_df.recode_version[0]
    #     image_path_list = []
    #     for image_name in dataset_df.image_name:
    #         image_path_list.append(osp.join(os.getcwd(),
    #                         category,
    #                         "recode",     # TODO: bring using db.column
    #                         recode_version,
    #                         image_name
    #                         ))
    #     return image_path_list
    
    # train_images = get_images(df_image["train"])
    # val_images = get_images(df_image["val"])
        
        
    
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
        
        
    
    if __name__=="__main__":
        cfg_flag = cfg.pop('flag')
        cfg = change_to_tuple(cfg, cfg_flag)
        cfg = Config(cfg)
        
        data_root = osp.join(os.getcwd(), cfg.dvc.category,
                                               cfg.dvc.ann.name,
                                               cfg.dvc.ann.version)
        
        dvc_cfg = dict(remote = cfg.dvc.remote,
                       bucket_name = cfg.gs.bucket.recoded,
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
        
 
        get_logger('enviroments', log_level = cfg.log_level)
        get_logger(f'training', log_level = cfg.log_level)      # TODO: save log file


        cfg.seed = np.random.randint(2**31)
        cfg.device = "cuda"            
        
        train_dataset, val_dataset = build_dataset(**set_dataset_cfg(cfg, database))
        
        if cfg.model.type == 'MaskRCNN':
            cfg.model.roi_head.bbox_head.num_classes = len(train_dataset.CLASSES) 
            cfg.model.roi_head.mask_head.num_classes = len(train_dataset.CLASSES) 
            
        train_loader_cfg = dict(train_dataset = train_dataset,
                                    val_dataset = val_dataset,
                                    train_batch_size=cfg.data.samples_per_gpu,
                                    val_batch_size = cfg.data.val.get("batch_size", None),
                                    num_workers=cfg.data.workers_per_gpu,
                                    seed = cfg.seed,
                                    shuffle = True)
        train_dataloader, val_dataloader = build_dataloader(**train_loader_cfg)
        
        # build model
        assert cfg.get('train_cfg') is None , 'train_cfg must be specified in both outer field and model field'
        
        if cfg.model.type == 'MaskRCNN':
            cfg.model.pop("type")
            from hibernation_no1.mmdet.modules.maskrcnn.maskrcnn import MaskRCNN
            model = MaskRCNN(**cfg.model)
            exit()
            
            
        
    
train_op = create_component_from_func(func = train,
                                        base_image = base_image.train,
                                        output_component_file= base_image.train_cp)
