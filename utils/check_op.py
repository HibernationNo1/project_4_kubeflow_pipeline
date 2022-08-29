from genericpath import exists
from kfp.components import OutputPath, create_component_from_func
from pipeline_config import Pipeline_Config
pl_cfg = Pipeline_Config    


def status_check(input_mode : str, cfg_dict : dict) :
    
    
    import os    
    import pipeline_taeuk4958
    from pipeline_taeuk4958.configs.config import Config
    
    config_file_path = os.path.join(os.getcwd(), cfg_dict['cfg_name'])
    with open(config_file_path, 'w') as f:
        f.write('\n')
    
    cfg = Config.fromfile(config_file_path, cfg_dict['cfg_dict'])
    
    
    if input_mode == "train":
        import torch
        print(f"torch.__version__ : {torch.__version__}")
        assert torch.cuda.is_available(), f"\n cuda is unavailable!"
    
    
    assert pipeline_taeuk4958.__version__ == "1.3.2", f"\n package pipeline_taeuk4958 version invalid version! \n installed version:{pipeline_taeuk4958.__version__}"
    print(f"pipeline_taeuk4958.__version__ : {pipeline_taeuk4958.__version__}")
    
    assert isinstance(cfg, Config), f"\n config got wrong type: {type(cfg)}, expected pipeline_taeuk4958"
    print(f"config : \n{cfg.pretty_text}")
    
 
    
print(f"set_config base_image : {pl_cfg.CHECK_IMAGE}")
status_check_op = create_component_from_func(func = status_check,
                                        base_image = pl_cfg.CHECK_IMAGE,
                                        output_component_file= pl_cfg.CHECK_COM_FILE)

