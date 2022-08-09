import re
import kfp
import kfp.dsl as dsl
import requests
import os 
import argparse
import json

from set_config import set_config_op
from record.record_dataset import record_op
from record.save_GS import save_dataset_op

from config import (USERNAME, PASSWORD, NAMESPACE, HOST,   
                    PIPELINE_PAC, PIPELINE_DISCRIPTION , EXPERIMENT_NAME, RUN_NAME, SECRETS_DICT)
    
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
    
    parser.add_argument("--pipeline_name", required = True, help="name of pipeline")    
    parser.add_argument("--pipeline_version", type = str, required = True, help="version of pipeline")    
    
    parser.add_argument('--mode', required = True, choices=['record', 'train', 'test'])
    parser.add_argument("--cfg", required = True, help="name of config file")
    
    # mode : record 
    parser.add_argument("--proportion-val", type=float, default = 0.0, help = "split proportion from train_dataset to val_dataset for valditate during training") 
    parser.add_argument('--dataset', help = 'annotations dir path')
    
    parser.add_argument('--client_secrets', default= 'client_secrets.json',
                        help = 'client_secrets file path (json format)')
    parser.add_argument('--ann_bk_name', help = 'bucket_name of annotation dataset stored in google cloud storage')
    parser.add_argument('--dataset_bk_name', help = 'bucket_name of annotation dataset stored in google cloud storage')
    parser.add_argument('--dataset_vers', type = str, help = 'version of recorded dataset to be store in google cloud storage.')
    
    parser.add_argument('--train', action = 'store_true', help = 'if True, go training after make custom dataset' ) # TODO
    

    args = parser.parse_args()
    
    return args

@dsl.pipeline(name="hibernation_project")
def project_pipeline(input_mode : str, input_dict : dict, gs_secret : dict):
    
    with dsl.Condition(input_mode == "record") : 	
        _set_config_op = set_config_op(input_dict)
        _record_op = record_op(gs_secret, _set_config_op.outputs['config'])
        _save_dataset_op = save_dataset_op(gs_secret, _set_config_op.outputs['config'], _record_op.outputs['train_dataset'], _record_op.outputs['val_dataset'])
        
        
if __name__=="__main__":      

    args = parse_args()
    args_dict = vars(args)
    
    input_dict = args_dict
    input_mode = args.mode
    # ann_path = os.path.join(os.getcwd(), "ann")
    
    
    if input_mode == "record":
        secrets_path = os.path.join(os.getcwd(), args.client_secrets)
        if not os.path.isfile(secrets_path): raise OSError(f"{secrets_path} is not exist!")
        with open(secrets_path, "r") as f:
            client_secrets_dict = json.load(f)
         
    params_dict = {'input_mode': input_mode, 'input_dict': input_dict, 'gs_secret' : client_secrets_dict}
   

    
    print("\n comile pipeline")
    kfp.compiler.Compiler().compile(
        project_pipeline,
        f"./{PIPELINE_PAC}"
        )

    print("\n connet_client")
    client = connet_client()

    print("\n upload_client")

    pipeline_path = os.path.join(os.getcwd(), PIPELINE_PAC)
    client.upload_pipeline(pipeline_name= args.pipeline_name,
                        description=PIPELINE_DISCRIPTION,
                        pipeline_package_path=pipeline_path)

    # client.upload_pipeline_version(pipeline_package_path = pipeline_path, 
    #                                pipeline_version_name = args.pipeline_version
    #                                pipeline_id = )



    pipeline_id = client.get_pipeline_id(args.pipeline_name)
    print(f"\n pipeline_id : {pipeline_id}")

    info_experiment = client.get_experiment(experiment_name= EXPERIMENT_NAME, namespace= NAMESPACE)
    info_experiment_id = info_experiment.id
    print(f"experiment_id : {info_experiment_id}")

    print("\n run pipeline")
    exec_run = client.run_pipeline(
            experiment_id = info_experiment_id,
            job_name = RUN_NAME,
            pipeline_id = pipeline_id,
            params = params_dict
            )
    list_pipelines = client.list_pipelines(page_size = 50) 
    
    run_id = exec_run.id
    print(f"run_id : {run_id}")

    completed_run = client.wait_for_run_completion(run_id=run_id, timeout=600)
    print(f"run completed_run : {completed_run.run.status}")
    print(f"completed_run.run.error : {completed_run.run.error}")

