# import kfp
# from kfp.components import InputPath, OutputPath, create_component_from_func

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
from kfp.components import InputPath, OutputPath, create_component_from_func
       
@create_component_from_func                          
def test_1(input_test_1 : dict, output_test_1 : OutputPath("dict") = None):
    import json		
    print(f"input_test_1 : {input_test_1}")
    input_test_1["3"] = input_test_1["1"] + input_test_1["2"]
    
    with open(output_test_1, "w") as f:
        json.dump(input_test_1)
    
    print(f"output_test_1 : {output_test_1} \n")
    
             
@create_component_from_func                                      
def test_2(input_test_2 : InputPath("dict"), output_test_2 : OutputPath("dict")):
    print(f"\n input_test_2 : {input_test_2}")
    import json
    with open(input_test_2, "r") as f:
        data_test_2 = json.loads(f) 
        
    data_test_2["4"] = input_test_2["2"]  * input_test_2["3"]
    
    with open(output_test_2, "w") as f:
        json.dump(data_test_2)
    print(f"output_test_2 : {input_test_2} \n ")
    


                            


@pipeline(name="add_example 0.1")
def my_pipeline(value_1:int, value_2:int)->int:
    dict_tmp = {"1" : value_1, "2" : value_2} 
    _task_1 = test_1(dict_tmp)
    print(f"type(_task_1.outputs) : {_task_1.outputs}, _task_1.outputs.keys() : {_task_1.outputs.keys()}")
    _task_2 = test_2(_task_1.outputs["data_output"])
    print(f"type(_task_2.outputs) : {_task_2.outputs}, _task_2.outputs.keys() : {_task_2.outputs.keys()}")
    print(f'_task_2.outputs["data_output"] : {_task_2.outputs["data_output"]}')

    

if __name__=="__main__":
    kfp.compiler.Compiler().compile(my_pipeline, "./add_exam.yaml")