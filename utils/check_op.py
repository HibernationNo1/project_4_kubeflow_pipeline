from genericpath import exists
from kfp.components import OutputPath, create_component_from_func
from pipeline_config import Pipeline_Config
pl_cfg = Pipeline_Config    


def status_check(input_mode : str, cfg_dict : dict) :
    
    
    import os    
    import pipeline_taeuk4958
    from pipeline_taeuk4958.configs.utils import load_config
    from pipeline_taeuk4958.configs.config import Config
    
    cfg = load_config(cfg_dict)
    
    print(f"pipeline_taeuk4958.__version__: : {pipeline_taeuk4958.__version__}")
    
    if input_mode == "train":
        import torch
        import mmcv
        import mmdet_taeuk4958
        print(f"\n torch.__version__ : {torch.__version__}")
        assert torch.cuda.is_available(), f"\n cuda is unavailable!"
        print(f" mmcv.__version__ : {mmcv.__version__}")
        print(f"mmdet_taeuk4958.__version__ : {mmdet_taeuk4958.__version__}")
    
    
    # assert pipeline_taeuk4958.__version__ == "1.3.3", f"\n package pipeline_taeuk4958 version invalid version! \n installed version:{pipeline_taeuk4958.__version__}"
    # print(f"pipeline_taeuk4958.__version__ : {pipeline_taeuk4958.__version__}")
    
    assert isinstance(cfg, Config), f"\n config got wrong type: {type(cfg)}, expected pipeline_taeuk4958"
    print(f"config : \n{cfg.pretty_text}")
    
    
    
print(f"set_config base_image : {pl_cfg.CHECK_IMAGE}")
status_check_op = create_component_from_func(func = status_check,
                                        base_image = pl_cfg.CHECK_IMAGE,
                                        output_component_file= pl_cfg.CHECK_COM_FILE)

