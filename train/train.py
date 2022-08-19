from kfp.components import InputPath, create_component_from_func

from pipeline_config import Pipeline_Config
pl_cfg = Pipeline_Config

def download_dataset(cfg_path: InputPath("dict"),
                     train_dataset_path: InputPath("dict"),
                     val_dataset_path: InputPath("dict")):
    
    import torch
    import json
    import os
    from pipeline_taeuk4958.configs.config import load_config_in_pipeline

    
        
    
    if __name__=="__main__":
        cfg = load_config_in_pipeline(cfg_path)
        
        print(f'pytorch version: {torch.__version__}')
        if not torch.cuda.is_available() : raise TypeError(f" cuda.is_available() is False ")
        
        
        
        
        # load_dataset
        with open(train_dataset_path, "r", encoding='utf-8') as f:
            train_dataset = json.load(f)
            
        with open(val_dataset_path, "r", encoding='utf-8') as f:
            val_dataset_path = json.load(f)
            
                           
        
  

        
        
    
  
download_dataset_op = create_component_from_func(func = download_dataset,
                                        base_image = pl_cfg.LOAD_DATA_IMAGE,
                                        output_component_file= pl_cfg.LOAD_DATA_COM_FILE)