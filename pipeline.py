
import kfp
import kfp.dsl as dsl
 
import argparse
import time

from utils.check_op import check_status_op, check_status
# from record.record_dataset_op import record_op
# from record.save_GS_op import save_dataset_op

# from train.train_op import train_op

from pipeline_utils import (set_config,
                            connet_client, get_experiment, run_pipeline, upload_pipeline, 
                            kfb_print,
                            CONFIGS)

from kubernetes.client.models import V1EnvVar, V1EnvVarSource, V1SecretKeySelector
from hibernation_no1.configs.utils import get_tuple_key

SECRETS = dict()

@dsl.pipeline(name="hibernation_project")
def project_pipeline(cfg_ai : dict, cfg_ai_flag: dict, 
                     cfg_recode : dict, cfg_recode_flag: dict
                     ):  # TODO: rename
     
    # ev_gs_type = V1EnvVar(name ='type', value_from= V1EnvVarSource( secret_key_ref=V1SecretKeySelector( name=client_sc_name, key = 'type')))
    # ev_gs_project_id = V1EnvVar(name ='project_id', value_from= V1EnvVarSource( secret_key_ref=V1SecretKeySelector( name=client_sc_name, key = 'project_id')))
    # ev_gs_private_key_id = V1EnvVar(name ='private_key_id', value_from= V1EnvVarSource( secret_key_ref=V1SecretKeySelector( name=client_sc_name, key = 'private_key_id')))
    # ev_gs_private_key = V1EnvVar(name ='private_key', value_from= V1EnvVarSource( secret_key_ref=V1SecretKeySelector( name=client_sc_name, key = 'private_key')))
    # ev_gs_client_email = V1EnvVar(name ='client_email', value_from= V1EnvVarSource( secret_key_ref=V1SecretKeySelector( name=client_sc_name, key = 'client_email')))
    # ev_gs_client_id = V1EnvVar(name ='client_id', value_from= V1EnvVarSource( secret_key_ref=V1SecretKeySelector( name=client_sc_name, key = 'client_id')))
    # ev_gs_auth_uri = V1EnvVar(name ='auth_uri', value_from= V1EnvVarSource( secret_key_ref=V1SecretKeySelector( name=client_sc_name, key = 'auth_uri')))
    # ev_gs_token_uri = V1EnvVar(name ='token_uri', value_from= V1EnvVarSource( secret_key_ref=V1SecretKeySelector( name=client_sc_name, key = 'token_uri')))
    # ev_gs_auth_provider_x509_cert_url = V1EnvVar(name ='auth_provider_x509_cert_url', value_from= V1EnvVarSource( secret_key_ref=V1SecretKeySelector( name=client_sc_name, key = 'auth_provider_x509_cert_url')))
    # ev_gs_client_x509_cert_url = V1EnvVar(name ='client_x509_cert_url', value_from= V1EnvVarSource( secret_key_ref=V1SecretKeySelector( name=client_sc_name, key = 'client_x509_cert_url')))
    

    client_sc_name = "client-secrets"
    for secrets_cate, secrets_cfg in SECRETS.items():
        for key in secrets_cfg:
            SECRETS[secrets_cate][key] = V1EnvVar(name=key, value_from=V1EnvVarSource(secret_key_ref=V1SecretKeySelector(name=client_sc_name, key=key)))
    
    with dsl.Condition(cfg_recode_flag != None) :   # must be `_flag`
        _check_status_op = check_status_op(cfg_recode, cfg_recode_flag)\
            .add_env_variable(SECRETS['gs']["type"]) \
            .add_env_variable(SECRETS['gs']["project_id"]) \
            .add_env_variable(SECRETS['gs']["private_key_id"]) \
            .add_env_variable(SECRETS['gs']["private_key"]) \
            .add_env_variable(SECRETS['gs']["client_email"]) \
            .add_env_variable(SECRETS['gs']["client_id"]) \
            .add_env_variable(SECRETS['gs']["auth_uri"]) \
            .add_env_variable(SECRETS['gs']["token_uri"]) \
            .add_env_variable(SECRETS['gs']["auth_provider_x509_cert_url"]) \
            .add_env_variable(SECRETS['gs']["client_x509_cert_url"]) \
            .add_env_variable(SECRETS['db']["password"]) \
            .add_env_variable(SECRETS['db']["host"]) \
            .add_env_variable(SECRETS['db']["port"]) \
        
  

         
def _parse_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument("--cfg_pipeline", required = True, help="name of config file which for pipeline")       
    parser.add_argument("--cfg_ai", help="name of config file which for ai")                           # TODO: rename
    parser.add_argument("--cfg_recode", help="name of config file which for ai") 
    
    kbf_parser = parser.add_argument_group('kubeflow')
    kbf_parser.add_argument("--pipeline_pw", type = str, required = True, help="password of kubeflow dashboard")        # required
    kbf_parser.add_argument("--pipeline_v", type = str, required = True, help="version of pipeline")                    # required
    kbf_parser.add_argument("--pipeline_n", type = str, help="name of pipeline")    
    kbf_parser.add_argument("--experiment_n", type = str, help="name of experiment") 
    kbf_parser.add_argument("--run_n", type = str, help="name of run") 
    
    gs_parser = parser.add_argument_group('google_storage')

    
    db_parser = parser.add_argument_group('database')
    db_parser.add_argument('--name_db', type = str, help = 'database name to connect to database')
    db_parser.add_argument('--user_db', type = str, help = 'user name to connect to database')
    
    
    
    parser.add_argument('--model_name', type = str, help= "name of model(.pth format)") 

    
    swin_parser = parser.add_argument_group('SwinTransformer')
    swin_parser.add_argument('--pm_dilation', type = int, help= "dilation of SwinTransformer.PatchMerging") 
    swin_parser.add_argument('--drop_rate', type = float, help= "drop_rate of SwinTransformer") 
    swin_parser.add_argument('--drop_path_rate', type = float, help= "drop_path_rate of SwinTransformer") 
    swin_parser.add_argument('--attn_drop_rate', type = float, help= "attn_drop_rate of SwinTransformer.SwinBlockSequence.ShiftWindowMSA.WindowMSA") 
    
    args = parser.parse_args()
    input_args = vars(args)     # TODO delete
    
    return args, input_args


def set_intput_papams():
    """ convert type from ConfigDict to Dict
    """
    params = dict()
    
    def convert(flag):
        for key, item in CONFIGS.items():
            if key == "cfg_pipeline": continue
            if item is None:
                if flag: params[f"{key}_flag"] = False
                else: params[f"{key}"] = False
                continue
            
            if flag: params[f"{key}_flag"] = get_tuple_key(item)
            else: params[f"{key}"] = dict(item)
        
    convert(True)
    convert(False)

    return params


if __name__=="__main__":      
    args, input_args = _parse_args()  
    set_config(args)

    SECRETS['gs'] = dict(CONFIGS.get('cfg_pipeline', None).secrets.gs)  
    SECRETS['db'] = dict(CONFIGS.get('cfg_pipeline', None).secrets.db)  
    

    cfg_pipeline = CONFIGS.get('cfg_pipeline', None)
    kfb_print("connet to kubeflow client")
    client = connet_client(cfg_pipeline.kbf.dashboard)  
    
    kfb_print("compile pipeline")             
    kfp.compiler.Compiler().compile(
        project_pipeline,
        f"./{cfg_pipeline.kbf.pipeline.pac}"
        )
    
    # get experiment id by create experiment or load experiment info
    kfb_print("get experiment")
    experiment_id = get_experiment(client, cfg_pipeline)          

    # get experiment id by create pipeline or updata pipeline version
    pipeline_id = upload_pipeline(client, cfg_pipeline.kbf.pipeline)     
     
    params = set_intput_papams() 
    run_pipeline(client, cfg_pipeline.kbf, experiment_id, pipeline_id, params)
    
    
    
        

