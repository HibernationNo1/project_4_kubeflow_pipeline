
import os
from pipeline_taeuk4958.configs.config import Config

def change_to_tuple(org_cfg, boolean_flag):
    """
    org_cfg : original config
    boolean_flag : key or list of idx that type was tuple at original config
    """
    if isinstance(boolean_flag, dict):
        assert isinstance(org_cfg, dict)
        
        for key in list(boolean_flag.keys()) :
            if key in list(org_cfg.keys()):
                org_cfg[key] = change_to_tuple(org_cfg[key], boolean_flag[key])
                
    elif isinstance(boolean_flag, list):
        assert isinstance(org_cfg, list)
            
        tmp_list = []
        for idx, ele in enumerate(boolean_flag):
            if isinstance(ele, dict):
                if len(list(ele.keys())) == 0: tmp_list.append(org_cfg[idx])
                else: tmp_list.append(change_to_tuple(org_cfg[idx], ele))
            elif isinstance(ele, int): tmp_list.append(tuple(org_cfg[ele]))
        return tmp_list
    
    elif boolean_flag :
        return tuple(org_cfg)
    
    return org_cfg




def load_config(cfg_dict):
    cfg = change_to_tuple(cfg_dict['cfg_dict'], cfg_dict['tuple_flag'])
    
    config_file_path = os.path.join(os.getcwd(), cfg_dict['cfg_name'])  
    with open(config_file_path, 'w') as f:
        f.write('\n')       # 빈 file 생성
    
    return Config.fromfile(config_file_path, cfg)       # 빈 file에 config를 그대로 옮겨쓴다.
    
    
    
    

def load_config_in_pipeline(cfg_path):
    if not cfg_path.endswith('.py'):
        cfg_pyformat_path = cfg_path + ".py"        # cfg_pyformat_path : {wiorkspace}/inputs/cfg/data.py
                                                    # can't command 'mv' 
    
    # change format to .py
    with open(cfg_path, "r") as f:
        data = f.read()
    with open(cfg_pyformat_path, "w") as f:
        f.write(data)       
    f.close()
    
    return Config.fromfile(cfg_pyformat_path)       # cfg_pyformat_path : must be .py format  