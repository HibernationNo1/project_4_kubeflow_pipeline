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
from train.load_dataset import download_dataset_op

from config import pipeline_config

from kubernetes.client.models import V1EnvVar, V1EnvVarSource, V1SecretKeySelector

@dsl.pipeline(name="hibernation_project")
def project_pipeline(input_mode : str, input_dict : dict):
    client_sc_name = "client-secrets"
    ev_gs_type = V1EnvVar(name ='type', value_from= V1EnvVarSource( secret_key_ref=V1SecretKeySelector( name=client_sc_name, key = 'type')))
    ev_gs_project_id = V1EnvVar(name ='project_id', value_from= V1EnvVarSource( secret_key_ref=V1SecretKeySelector( name=client_sc_name, key = 'project_id')))
    ev_gs_private_key_id = V1EnvVar(name ='private_key_id', value_from= V1EnvVarSource( secret_key_ref=V1SecretKeySelector( name=client_sc_name, key = 'private_key_id')))
    ev_gs_private_key = V1EnvVar(name ='private_key', value_from= V1EnvVarSource( secret_key_ref=V1SecretKeySelector( name=client_sc_name, key = 'private_key')))
    ev_gs_client_email = V1EnvVar(name ='client_email', value_from= V1EnvVarSource( secret_key_ref=V1SecretKeySelector( name=client_sc_name, key = 'client_email')))
    ev_gs_client_id = V1EnvVar(name ='client_id', value_from= V1EnvVarSource( secret_key_ref=V1SecretKeySelector( name=client_sc_name, key = 'client_id')))
    ev_gs_auth_uri = V1EnvVar(name ='auth_uri', value_from= V1EnvVarSource( secret_key_ref=V1SecretKeySelector( name=client_sc_name, key = 'auth_uri')))
    ev_gs_token_uri = V1EnvVar(name ='token_uri', value_from= V1EnvVarSource( secret_key_ref=V1SecretKeySelector( name=client_sc_name, key = 'token_uri')))
    ev_gs_auth_provider_x509_cert_url = V1EnvVar(name ='auth_provider_x509_cert_url', value_from= V1EnvVarSource( secret_key_ref=V1SecretKeySelector( name=client_sc_name, key = 'auth_provider_x509_cert_url')))
    ev_gs_client_x509_cert_url = V1EnvVar(name ='client_x509_cert_url', value_from= V1EnvVarSource( secret_key_ref=V1SecretKeySelector( name=client_sc_name, key = 'client_x509_cert_url')))

    
    _set_config_op = set_config_op(input_dict) \
            .add_env_variable(ev_gs_type) \
            .add_env_variable(ev_gs_project_id) \
            .add_env_variable(ev_gs_private_key_id) \
            .add_env_variable(ev_gs_private_key) \
            .add_env_variable(ev_gs_client_email) \
            .add_env_variable(ev_gs_client_id) \
            .add_env_variable(ev_gs_auth_uri) \
            .add_env_variable(ev_gs_token_uri) \
            .add_env_variable(ev_gs_auth_provider_x509_cert_url) \
            .add_env_variable(ev_gs_client_x509_cert_url)     
                                                                                     
    with dsl.Condition(input_mode == "record") : 	
        _record_op = record_op(_set_config_op.outputs['config']) \
            .add_env_variable(ev_gs_type) \
            .add_env_variable(ev_gs_project_id) \
            .add_env_variable(ev_gs_private_key_id) \
            .add_env_variable(ev_gs_private_key) \
            .add_env_variable(ev_gs_client_email) \
            .add_env_variable(ev_gs_client_id) \
            .add_env_variable(ev_gs_auth_uri) \
            .add_env_variable(ev_gs_token_uri) \
            .add_env_variable(ev_gs_auth_provider_x509_cert_url) \
            .add_env_variable(ev_gs_client_x509_cert_url) 
            
        _save_dataset_op = save_dataset_op(_set_config_op.outputs['config'], _record_op.outputs['train_dataset'], _record_op.outputs['val_dataset']) \
            .add_env_variable(ev_gs_type) \
            .add_env_variable(ev_gs_project_id) \
            .add_env_variable(ev_gs_private_key_id) \
            .add_env_variable(ev_gs_private_key) \
            .add_env_variable(ev_gs_client_email) \
            .add_env_variable(ev_gs_client_id) \
            .add_env_variable(ev_gs_auth_uri) \
            .add_env_variable(ev_gs_token_uri) \
            .add_env_variable(ev_gs_auth_provider_x509_cert_url) \
            .add_env_variable(ev_gs_client_x509_cert_url) 

        
    with dsl.Condition(input_mode == "train") :
        _download_dataset_op = download_dataset_op(_set_config_op.outputs['config'])
        
        pass
         
         
            
def connet_client(pl_cfg):   
    session = requests.Session()
    response = session.get(pl_cfg.HOST)

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {"login": pl_cfg.USERNAME, "password": pl_cfg.PASSWORD}
    session.post(response.url, headers=headers, data=data)                              
    session_cookie = session.cookies.get_dict()["authservice_session"]  

    # client에 접속
    client = kfp.Client(
        host=f"{pl_cfg.HOST}/pipeline",
        namespace=f"{pl_cfg.NAMESPACE}",
        cookies=f"authservice_session={session_cookie}",
    )
    return client


def upload_pipeline(client, args, pl_cfg):
    pipeline_path = os.path.join(os.getcwd(), pl_cfg.PIPELINE_PAC)
    if not os.path.isfile(pipeline_path) : raise OSError(f" {pipeline_path} is not exist! ")
    
    if client.get_pipeline_id(pl_cfg.PIPELINE_NAME) == None:
        print(f"\n Upload initial version pipeline named {pl_cfg.PIPELINE_NAME}!",end = " " )
        client.upload_pipeline(pipeline_package_path= pipeline_path,
                            pipeline_name= pl_cfg.PIPELINE_NAME,
                            description= pl_cfg.PIPELINE_DISCRIPTION)
        print("seccess!")
        pipeline_id = client.get_pipeline_id(pl_cfg.PIPELINE_NAME)
    else: 
        pipeline_id = client.get_pipeline_id(pl_cfg.PIPELINE_NAME)
        pipelince_versions = client.list_pipeline_versions(pipeline_id = pipeline_id, page_size = 50)
        
        versions = []  
        for pipeline_index in range(pipelince_versions.total_size):
            versions.append(pipelince_versions.versions[pipeline_index].name) 
      
        
        if args.p_version in versions: raise TypeError(f" {args.p_version} version of {pl_cfg.PIPELINE_NAME} is exist! ")
                
        print(f"\n Upload pipeline {args.p_version} version named {pl_cfg.PIPELINE_NAME} : ", end = " ")
        client.upload_pipeline_version(pipeline_package_path= pipeline_path,
                                    pipeline_version_name = args.p_version,
                                    pipeline_id = pipeline_id,
                                    description = pl_cfg.PIPELINE_DISCRIPTION)    
        print("seccess!")  
        
        
    return pipeline_id  

def get_experiment(client, pl_cfg) : 
        list_experiments = client.list_experiments(page_size = 50)
        if list_experiments.total_size == None:
            experiment = client.create_experiment(name = pl_cfg.EXPERIMENT_NAME)
        else:
            experiment = client.get_experiment(experiment_name= pl_cfg.EXPERIMENT_NAME, namespace= pl_cfg.NAMESPACE)
        
        experiment_id = experiment.id
        return experiment_id
    

def run_pipeline(client, experiment_id, pipeline_id, params_dict, run_name):
    exec_run = client.run_pipeline(
            experiment_id = experiment_id,
            job_name = run_name,
            pipeline_id = pipeline_id,
            params = params_dict
            )
    print("\n run pipeline")

    run_id = exec_run.id


    completed_run = client.wait_for_run_completion(run_id=run_id, timeout=600)
    print(f"status of {run_name} : {completed_run.run.status}")
        
        
def get_params(args, input_dict):
            
    params_dict = {'input_mode': args.mode, 'input_dict': input_dict}
        
    return params_dict
   
    

def parse_args(pl_cfg):
    parser = argparse.ArgumentParser(description="Change structure from comments to custom dataset in json file format.")    
    
    parser.add_argument("--p_version", type = str, required = True, help="version of pipeline")    
    
    parser.add_argument('--mode', required = True, choices=['record', 'train', 'test'])
    parser.add_argument("--cfg", required = True, help="name of config file")
    
    # mode : record 
    parser.add_argument("--proportion-val", type=float, default = 0.0, help = "split proportion from train_dataset to val_dataset for valditate during training") 
    parser.add_argument('--dataset', help = 'annotations dir path')
    
    parser.add_argument('--client_secrets', default= 'client_secrets.json',
                        help = 'client_secrets file path (json format)')
    parser.add_argument('--ann_bk_name', help = 'bucket_name of annotation dataset stored in google cloud storage')
    parser.add_argument('--dataset_bk_name', help = 'bucket_name of annotation dataset stored in google cloud storage')
    parser.add_argument('--d_version', type = str, help = 'version of recorded dataset to be store in google cloud storage.')
    
    parser.add_argument('--train', action = 'store_true', help = 'if True, go training after make custom dataset' ) # TODO
    
    # mode : train
    parser.add_argument('--train_json', help= "name of train dataset file in .json format")
    parser.add_argument(
        '--validate',
        default=False,
        action="store_true",
        help='whether do evaluate the checkpoint during training')
    parser.add_argument('--val_json', help= "name of train dataset file in .json format")
    parser.add_argument(
        '--finetun',
        default=False,
        action="store_true",
        help='whether do fine tuning')
    
    parser.add_argument('--model_vers', 
                        type = str,
                        default= 0.1,
                        help= "directory name. version of model to be store in google cloud storage.")
    parser.add_argument('--d_version_t', type = str, help = 'version of recorded dataset in google cloud storage.')
    
    args = parser.parse_args()
    input_dict = vars(args)
    input_dict['p_name'] = pl_cfg.PIPELINE_NAME
    
    
    
    return args, input_dict
       
# python pipeline.py --mode record --cfg record_config.py --d_version 0.1 --p_version 0.1
# python pipeline.py --mode train --cfg train_config.py --d_version_t 0.1 --p_name train_test --p_version 0.13 
if __name__=="__main__":      

    pl_cfg = pipeline_config() 
    args, input_dict = parse_args(pl_cfg)       

    print("\n comile pipeline")
    kfp.compiler.Compiler().compile(
        project_pipeline,
        f"./{pl_cfg.PIPELINE_PAC}"
        )

    print("\n connet_client")
    client = connet_client(pl_cfg)
    
    experiment_id = get_experiment(client, pl_cfg)
   
    if pl_cfg.RUN_EXIST_PIPELINE:
        print(f" get pipeline: {pl_cfg.PIPELINE_NAME}.{args.p_version} id ")
        pipeline_id = client.get_pipeline_id(pl_cfg.PIPELINE_NAME)
    else:
        pipeline_id = upload_pipeline(client, args, pl_cfg)
    
    print("\n get experiment")
    
    
    
    
    
    
    params_dict = get_params(args, input_dict)
    run_pipeline(client, experiment_id, pipeline_id, params_dict, pl_cfg.RUN_NAME)
    
    
    
        

