import kfp
from kfp.components import InputPath, OutputPath, create_component_from_func


import argparse

@create_component_from_func(
        base_image = 'hibernation4958/test_2:0.1',
        output_component_file="test_2.component.yaml")
def test_2(args : InputPath("dict"),
           save_path : OutputPath("dict")):  
    import json
    
    with open(args, "r") as f:
        data = json.load(f)
    
    print(f"args.test_2 : {data}")
    data['3'] = 3
    dict_tmp = data

    with open(save_path, "w") as f:
        json.dump(dict_tmp, f)
    
    return dict_tmp

