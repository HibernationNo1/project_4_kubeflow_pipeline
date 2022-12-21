
import kfp
import kfp.dsl as dsl
 
import argparse

from pipeline_base_image_cfg import BASE_IMG
from recode.recode_op import recode_op
from train.train_op import train_op
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
def project_pipeline(cfg_train : dict, train_using, 
                     cfg_recode : dict, recode_using
                     ):  # TODO: rename
     

    client_sc_name = "client-secrets"
    for secrets_cate, secrets_cfg in SECRETS.items():
        for key in secrets_cfg:
            SECRETS[secrets_cate][key] = V1EnvVar(name=key, value_from=V1EnvVarSource(secret_key_ref=V1SecretKeySelector(name=client_sc_name, key=key)))
    
    with dsl.Condition(recode_using != None) :   
        _check_status_op = recode_op(cfg_recode)\
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
            .add_env_variable(SECRETS['db']["port"]) 
    
    with dsl.Condition(train_using != None) :
        _train_op = train_op(cfg_train)\
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
            .add_env_variable(SECRETS['db']["port"]) 
  

         
def _parse_args():
    
    parser = argparse.ArgumentParser()    
    parser.add_argument("--cfg_pipeline", required = True, help="name of config file which for pipeline")       
    parser.add_argument("--cfg_train", help="name of config file which for training")                           # TODO: rename
    parser.add_argument("--cfg_recode", help="name of config file which for recode") 
    
    kbf_parser = parser.add_argument_group('kubeflow')
    kbf_parser.add_argument("--pipeline_pw", type = str, required = True, help="password of kubeflow dashboard")        # required
    kbf_parser.add_argument("--pipeline_v", type = str, required = True, help="version of pipeline")                    # required
    kbf_parser.add_argument("--pipeline_n", type = str, help="name of pipeline")    
    kbf_parser.add_argument("--experiment_n", type = str, help="name of experiment") 
    kbf_parser.add_argument("--run_n", type = str, help="name of run") 
    
    # gs_parser = parser.add_argument_group('google_storage')

    
    db_parser = parser.add_argument_group('database')
    db_parser.add_argument('--name_db', type = str, help = 'Database name to connect to database')
    db_parser.add_argument('--user_db', type = str, help = 'User name to connect to database')
    
    
    train_parser = parser.add_argument_group('train')
    train_parser.add_argument('--model_name', type = str, help= "Name of model(.pth format)") 
    train_parser.add_argument('--val_iter', type = str, help= "Divisor number of iter printing the training state log.") 
    
    
    swin_parser = parser.add_argument_group('SwinTransformer')
    swin_parser.add_argument('--pm_dilation', type = int, help= "dilation of SwinTransformer.PatchMerging") 
    swin_parser.add_argument('--drop_rate', type = float, help= "drop_rate of SwinTransformer") 
    swin_parser.add_argument('--drop_path_rate', type = float, help= "drop_path_rate of SwinTransformer") 
    swin_parser.add_argument('--attn_drop_rate', type = float, help= "attn_drop_rate of SwinTransformer.SwinBlockSequence.ShiftWindowMSA.WindowMSA") 
    
    args = parser.parse_args()
    input_args = vars(args)     # TODO delete
    
    return args, input_args


def set_intput_papams():
    """ Convert type from ConfigDict to Dict for input to pipeline.
        And sets flags That determine which components be include in pipeline  
    """
    params = dict()
    
    def convert(flag):
        for key, item in CONFIGS.items():
            if key == "pipeline": continue
            if item is None:                
                if flag: params[f"{key}_using"] = False
                else: params[f"cfg_{key}"] = False
                continue
            
            
            if flag: 
                kfb_print(f"{key}_op base_image : {BASE_IMG[key]}", nn=False)
                params[f"{key}_using"] = True
            else: 
                params[f"cfg_{key}"] = dict(item)
                params[f"cfg_{key}"]['flag'] = get_tuple_key(item)
        
    convert(True)
    convert(False)
    
    # params.keys(): 
    # ['train_using', 'recode_using', 'cfg_train', 'cfg_recode']
    # If 'recode' is not selected as a pipeline component, it has a value of `False`.
    
    return params


if __name__=="__main__":      
    args, input_args = _parse_args()  
    set_config(args)

    SECRETS['gs'] = dict(CONFIGS.get('pipeline', None).secrets.gs)  
    SECRETS['db'] = dict(CONFIGS.get('pipeline', None).secrets.db)  
    

    cfg_pipeline = CONFIGS.get('pipeline', None)
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
    
  
    # exit()
    
    run_pipeline(client, cfg_pipeline.kbf, experiment_id, pipeline_id, params)
    
    
    
        

