from ast import Param
from re import M
from typing import NamedTuple
import kfp
from kfp.components import OutputPath, create_component_from_func

from configs.labelme_config import Labelme_Config as for_type
config = for_type
cfg_type = type(config)
            
            
def Labelme_Custom(cfg):
    train_dataset = dict(one = 1)
    val_dataset = dict(two = 2)
    
    return train_dataset, val_dataset

def labelme(args: dict,
           train_dataset_path: OutputPath("dict"),
           val_dataset_path: OutputPath("dict")) -> cfg_type:

    import os
    print("???")
    print(os.path.isfile('/usr/lib/python3.9/json/encoder.py'))
    print(os.listdir('/usr'))
    
    import json
    from configs.labelme_config import Labelme_Config
    from utils import NpEncoder
    cfg = Labelme_Config
    
    

    
    cfg.options['proportion_val'] = args['ratio_val']    
    
        
    train_dataset, val_dataset = Labelme_Custom(cfg)
    json.dump(train_dataset, open(train_dataset_path, "w"), indent = 4, cls = NpEncoder)
    json.dump(val_dataset, open(val_dataset_path, "w"), indent = 4, cls = NpEncoder)
    
    return cfg


labelme_op = create_component_from_func(func = labelme,
                                        base_image = 'hibernation4958/labelme:0.1',
                                        output_component_file="labelme.component.yaml")

