import kfp
import kfp.dsl as dsl
 
import argparse

from config.set_config import set_config_op
from record.record_dataset import record_op
from record.save_GS import save_dataset_op
from train.load_dataset import download_dataset_op

from pipeline_config import Pipeline_Config
from pipeline_utils import (connet_client,get_experiment, upload_pipeline, get_params, run_pipeline, get_pipeline_id)

from kubernetes.client.models import V1EnvVar, V1EnvVarSource, V1SecretKeySelector

@dsl.pipeline(name="hibernation_project")
def project_pipeline(input_mode : str, input_args : dict):
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
    
        
    _set_config_op = set_config_op(input_args) 
                      
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
    input_args = vars(args)
    input_args['p_name'] = pl_cfg.PIPELINE_NAME
    
    
    return args, input_args
       
# python pipeline.py --mode record --cfg record_config.py --d_version 0.1 --p_version 0.2
# python pipeline.py --mode train --cfg train_config.py --d_version_t 0.1 --p_name train_test --p_version 0.13 
if __name__=="__main__":      

    pl_cfg = Pipeline_Config()              # get config
    args, input_args = parse_args(pl_cfg)   # get args, input_args    

    print("\n compile pipeline")             
    kfp.compiler.Compiler().compile(
        project_pipeline,
        f"./{pl_cfg.PIPELINE_PAC}"
        )

    print("\n connet_client")
    client = connet_client(pl_cfg)  
    
    print("\n get experiment")
    experiment_id = get_experiment(client, pl_cfg)          # get experiment id by create experiment or load experiment info
    
    pipeline_id = get_pipeline_id(pl_cfg, args, client)     # get experiment id by create pipeline or updata pipeline version
        
    params_dict = get_params(args, input_args)              # parameters for pipeline run
    run_pipeline(client, experiment_id, pipeline_id, params_dict, pl_cfg.RUN_NAME)
    
    
    
        

