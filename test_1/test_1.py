
import kfp
from kfp.components import InputPath, OutputPath, create_component_from_func


import argparse

@create_component_from_func(
        base_image = 'hibernation4958/test_1:0.1',
        output_component_file="test_1.component.yaml")
def test_1(args : int, 
           save_path : OutputPath("dict")):
    
    import json
    
    print(f"args.test_1 : {args.tmp}")
    
    dict_tmp = {'1' : args}
    dict_tmp['2'] = 2

    with open(save_path, "w") as f:
        json.dump(dict_tmp, f)
        

if __name__=="__main__":
    

    parser = argparse.ArgumentParser()    
    parser.add_argument("--tmp", help = "tmp") 
    
    args = parser.parse_args()
    test_1(args)
    
    
    
    
    
    
    