import kfp
from kfp.components import create_component_from_func

from test_1 import test_1
from test_2 import test_2

import argparse



test_1_op = create_component_from_func(finc = test_1, 
                                       base_image = 'hibernation4958/test_1:0.4',
                                       output_component_file="test_1.component.yaml")
test_2_op = create_component_from_func(finc = test_2,
                                       base_image = 'hibernation4958/test_2:0.4',
                                       output_component_file="test_2.component.yaml")


@kfp.dsl.pipeline(name="test Example")
def test_pipeline(tmp_num):
    _test_1_op = test_1_op(tmp_num)
    _test_2_op = test_2_op(_test_1_op.outputs["data_output"])
    
    print(f"_test_2_op : {_test_2_op}")

    
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser()    
    parser.add_argument("--tmp", help = "tmp") 
    
    args = parser.parse_args()
    
    kfp.compiler.Compiler().compile(test_pipeline(args.tmp), "./test_pipeline.yaml")