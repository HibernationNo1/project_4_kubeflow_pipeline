import kfp
from test_1 import test_1
from test_2 import test_2

import argparse


@kfp.dsl.pipeline(name="test Example")
def test_pipeline(tmp_num):
    test_1_op = test_1(tmp_num)
    result = test_2(test_1_op.outputs["data_output"])
    
    print(f"result : {result}")

    
    
if __name__=="__main__":
    
    parser = argparse.ArgumentParser()    
    parser.add_argument("--tmp", help = "tmp") 
    
    args = parser.parse_args()
    
    kfp.compiler.Compiler().compile(test_pipeline(args.tmp), "./test_pipeline.yaml")