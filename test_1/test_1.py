
import kfp
from kfp.components import InputPath, OutputPath, create_component_from_func


import argparse

def test_1(args : int, 
           save_path : OutputPath("dict")):
    
    import json
    
    print(f"args.test_1 : {args.tmp}")
    
    dict_tmp = {'1' : args}
    dict_tmp['2'] = 2

    with open(save_path, "w") as f:
        json.dump(dict_tmp, f)
        

    
    
    
    
    
    
    