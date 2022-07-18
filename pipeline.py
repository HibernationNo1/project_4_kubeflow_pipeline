import kfp
from kfp.components import InputPath, OutputPath, create_component_from_func

# from test_1 import test_1
# from test_2 import test_2


# test_1_op = create_component_from_func(func = test_1, 
#                                        base_image = 'hibernation4958/test_1:0.4',
#                                        output_component_file="test_1.component.yaml")
# test_2_op = create_component_from_func(func = test_2,
#                                        base_image = 'hibernation4958/test_2:0.4',
#                                        output_component_file="test_2.component.yaml")


# @kfp.dsl.pipeline(name="test Example")
# def test_pipeline(tmp_num):
#     _test_1_op = test_1_op(tmp_num)
#     _test_2_op = test_2_op(_test_1_op.outputs["data_output"])
    
#     print(f"_test_2_op : {_test_2_op}")

    
# @create_component_from_func(
#     base_image = 'hibernation4958/test_1:0.5',
#     output_component_file="test_1.component.yaml")
# def test_1(args : int, 
#            save_path : OutputPath("dict")):
    
#     import json
    
#     print(f"args.test_1 : {args.tmp}")
    
#     dict_tmp = {'1' : args}
#     dict_tmp['2'] = 2

#     with open(save_path, "w") as f:
#         json.dump(dict_tmp, f)


# @create_component_from_func(
#     base_image = 'hibernation4958/test_2:0.5',
#     output_component_file="test_2.component.yaml")
# def test_2(args : InputPath("dict"),
#            save_path : OutputPath("dict")):  
#     import json
    
#     with open(args, "r") as f:
#         data = json.load(f)
    
#     print(f"args.test_2 : {data}")
#     data['3'] = 3
#     dict_tmp = data

#     with open(save_path, "w") as f:
#         json.dump(dict_tmp, f)
    
#     return dict_tmp


# @kfp.dsl.pipeline(name="test Example")
# def test_pipeline(tmp_num):
#     _test_1_op = test_1(tmp_num)
#     _test_2_op = test_2(_test_1_op.outputs["data_output"])
    
#     print(f"_test_2_op : {_test_2_op}")

# if __name__=="__main__":    
#     kfp.compiler.Compiler().compile(test_pipeline(), "./test_pipeline.yaml")


import kfp
from kfp.dsl import pipeline
from kfp.components import create_component_from_func
                          
def add(value_1:int, value_2:int)->int:		
    ret = value_1 + value_2	
    return ret
                                                   
def subtract(value_1:int, value_2:int)->int:
    ret = value_1 - value_2
    return ret	   
                            
def multiply(value_1:int, value_2:int)->int:
    ret = value_1 * value_2
    return ret	
                            
add_op = create_component_from_func(add)
subtract_op = create_component_from_func(subtract)
multiply_op = create_component_from_func(multiply)


@pipeline(name="add_example")
def my_pipeline(value_1:int, value_2:int)->int:
    task_1 = add_op(value_1, value_2)
    task_2 = subtract_op(value_1, value_2)

    task_3 = multiply_op(task_1.output, task_2.output)  #  output -> input 으로 연결

if __name__=="__main__":
    kfp.compiler.Compiler().compile(my_pipeline, "./add_exam.yaml")