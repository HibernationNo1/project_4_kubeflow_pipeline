from kfp.components import create_component_from_func
from pipeline_config import Pipeline_Config
pl_cfg = Pipeline_Config  


# 1.6.0-cuda10.1-cudnn7-devel
# 1.6.0-cuda10.1-cudnn7-runtime

def test_container() :
    

    import os             
    import pipeline_taeuk4958
    print(f"\n pipeline_taeuk4958.__version__ : {pipeline_taeuk4958.__version__} \n")
        

    import torch
    print(f"torch.cuda.is_available() ; {torch.cuda.is_available()}")
    print(f"torch.__version__ : {torch.__version__}")

    
print(f"test_container base_image : {pl_cfg.test_image}")
test_container_op = create_component_from_func(func = test_container,
                                        base_image = pl_cfg.test_image,
                                        output_component_file= pl_cfg.comfile)

