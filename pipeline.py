import kfp
import kfp.dsl as dsl
import requests
import os
import argparse

from labelme import labelme_op
from save_S3 import save_labelme_op

USERNAME = "user@example.com"
PASSWORD = "12341234"
NAMESPACE = "kubeflow-user-example-com"
HOST =  "https://b5b2-1-214-32-67.ngrok.io"     # "http://192.168.0.167:80"       
    
def connet_client():   
    session = requests.Session()
    response = session.get(HOST)

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {"login": USERNAME, "password": PASSWORD}
    session.post(response.url, headers=headers, data=data)                              
    session_cookie = session.cookies.get_dict()["authservice_session"]  

    # client에 접속
    client = kfp.Client(
        host=f"{HOST}/pipeline",
        namespace=f"{NAMESPACE}",
        cookies=f"authservice_session={session_cookie}",
    )
    return client

def parse_args():
    parser = argparse.ArgumentParser(description="Change structure from comments to custom dataset in json file format.")    
    
    parser.add_argument("--name", required = True, help="name of pipeline")
    
    parser.add_argument('--mode', required = True, choices=['labelme', 'train', 'test'])
    parser.add_argument("--cfg", required = True, help="name of config file")
    
    # mode : labelme 
    parser.add_argument("--ratio-val", type=float, default = 0.0, help = "split ratio from train_dataset to val_dataset for valditate during training") 
    parser.add_argument('--train', action = 'store_true', help = 'if True, go training after make custom dataset' ) # TODO
    

    
    args = parser.parse_args()
    
    return args



@dsl.pipeline(name="hibernation_project")
def project_pipeline(input_mode : str, input_dict : dict):
    
    with dsl.Condition(input_mode == "labelme") : 	
        _labelme_op = labelme_op(input_dict)
        _save_labelme_op = save_labelme_op(_labelme_op.outputs['Output'], _labelme_op.outputs['train_dataset'], _labelme_op.outputs['val_dataset'])
        
        
if __name__=="__main__":
    # python pipeline.py --name labelme_0.4 --mode labelme --cfg configs/labelme_config.py --ratio-val 0.01
    args = parse_args()
    args_dict = vars(args)

    pipeline_name = args.name
    pipeline_pac = "mmdet_project.yaml"
    pipeline_path = os.path.join(os.getcwd(), pipeline_pac)
    pipeline_description = "test"
    
    input_dict = args_dict
    input_mode = args.mode
    params_dict = dict(input_mode = 1, input_dict = 2)
    
    
    experiment_name = "test2"
    
    job_name = "test_run"
    
    print("\n comile pipeline")
    kfp.compiler.Compiler().compile(
        project_pipeline,
        f"./{pipeline_pac}"
        )

    print("\n connet_client")
    client = connet_client()

    print("\n upload_client")
    
    client.upload_pipeline(pipeline_name= pipeline_name,
                        description=pipeline_description,
                        pipeline_package_path=pipeline_path)


    pipeline_id = client.get_pipeline_id(pipeline_name)
    print(f"\n pipeline_id : {pipeline_id}")

    info_experiment = client.get_experiment(experiment_name= experiment_name, namespace= NAMESPACE)
    info_experiment_id = info_experiment.id
    print(f"experiment_id : {info_experiment_id}")

    print("\n run pipeline")
    exec_run = client.run_pipeline(
            experiment_id = info_experiment_id,
            job_name = job_name,
            pipeline_id = pipeline_id,
            params = params_dict
            )

    run_id = exec_run.id
    print(f"run_id : {run_id}")

    completed_run = client.wait_for_run_completion(run_id=run_id, timeout=600)
    print(f"run completed_run : {completed_run.run.status}")
    print(f"completed_run.run.error : {completed_run.run.error}")

