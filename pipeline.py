
import kfp
import kfp.dsl as dsl
 
import argparse
import time

# from utils.check_op import check_status_op
# from record.record_dataset_op import record_op
# from record.save_GS_op import save_dataset_op

# from train.train_op import train_op

from pipeline_utils import (connet_client,get_experiment, get_params, run_pipeline, get_pipeline_id)

from kubernetes.client.models import V1EnvVar, V1EnvVarSource, V1SecretKeySelector
from hibernation_no1.configs.config import Config

@dsl.pipeline(name="hibernation_project")
def project_pipeline(input_mode : str, cfg: dict):    
    
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
    
        
    _status_check_op = check_status_op(input_mode, cfg)        
                      
    with dsl.Condition(input_mode == "record") : 	    
        _record_op = record_op(cfg).after(_status_check_op) \
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
 
            
        _save_dataset_op = save_dataset_op(cfg, _record_op.outputs['train_dataset'], _record_op.outputs['val_dataset']) \
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
        _train_op = train_op(cfg).after(_status_check_op) \
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
        
        
        pass


def get_config(args):
    return Config.fromfile(args.cfg_pipeline), Config.fromfile(args.cfg_ai)
        
def set_pipeline_cfg(args, cfg):
    if args.pipeline_n is not None: cfg.kbf.pipeline.name = args.pipeline_n
    if args.experiment_n is not None: cfg.kbf.experiment.name = args.experiment_n
    if args.run_n is not None: cfg.kbf.run.name = args.run_n    
    cfg.kbf.dashboard.pw =  args.pipeline_pw 

    if args.client_secrets is not None: cfg.gs.client_secrets = args.client_secrets   
    
         


def set_config(args):
    cfg_pipeline, cfg_ai = get_config(args)
      
    set_pipeline_cfg(args, cfg_pipeline)

    # set_ai_cfg()
    
    if args.model_name is not None: cfg_ai.filename_tmpl = f"{args.model_name}"+"_{}.path"

    if args.pm_dilation is not None: cfg_ai.model.backbone.pm_dilation = args.pm_dilation
    if args.drop_rate is not None: cfg_ai.model.backbone.drop_rate = args.drop_rate
    if args.drop_path_rate is not None: cfg_ai.model.backbone.drop_path_rate = args.drop_path_rate
    if args.attn_drop_rate is not None: cfg_ai.model.backbone.attn_drop_rate = args.attn_drop_rate
    
    return cfg_pipeline, cfg_ai
         
def parse_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument("--cfg_pipeline", required = True, help="name of config file which for pipeline")       
    parser.add_argument("--cfg_ai", required = True, help="name of config file which for ai")                           # TODO: 이름 바꾸기
    
    kbf_parser = parser.add_argument_group('kubeflow')
    kbf_parser.add_argument("--pipeline_pw", type = str, required = True, help="password of kubeflow dashboard")        # required
    kbf_parser.add_argument("--pipeline_v", type = str, required = True, help="version of pipeline")                    # required
    kbf_parser.add_argument("--pipeline_n", type = str, help="name of pipeline")    
    kbf_parser.add_argument("--experiment_n", type = str, help="name of experiment") 
    kbf_parser.add_argument("--run_n", type = str, help="name of run") 
    
    gs_parser = parser.add_argument_group('google_strage')
    gs_parser.add_argument('--client_secrets', help = 'client_secrets file path (json format).')
    
    
    parser.add_argument('--model_name', type = str, help= "name of model(.pth format)") 

    
    swin_parser = parser.add_argument_group('SwinTransformer')
    swin_parser.add_argument('--pm_dilation', type = int, help= "dilation of SwinTransformer.PatchMerging") 
    swin_parser.add_argument('--drop_rate', type = float, help= "drop_rate of SwinTransformer") 
    swin_parser.add_argument('--drop_path_rate', type = float, help= "drop_path_rate of SwinTransformer") 
    swin_parser.add_argument('--attn_drop_rate', type = float, help= "attn_drop_rate of SwinTransformer.SwinBlockSequence.ShiftWindowMSA.WindowMSA") 
    
    
    args = parser.parse_args()
    cfg_pipeline, cfg_ai = set_config(args)
 
    input_args = vars(args)
    
    return args, input_args, cfg_pipeline, cfg_ai
       
# python pipeline.py --cfg_pipeline pipeline_config.py --cfg_ai config/swin_maskrcnn.py  --pipeline_v 0.1  --pipeline_pw 4958
if __name__=="__main__":      
    args, input_args, cfg_pipeline, cfg_ai = parse_args()  
             
    # print("\n connet_client")
    # client = connet_client(cfg_pipeline.kbf.dashboard)  
          
    print("\n compile pipeline")             
    kfp.compiler.Compiler().compile(
        project_pipeline,
        f"./{pl_cfg.PIPELINE_PAC}"
        )

    
    
    # print("\n get experiment")
    # experiment_id = get_experiment(client, pl_cfg)          # get experiment id by create experiment or load experiment info
    
    # pipeline_id = get_pipeline_id(pl_cfg, args, client)     # get experiment id by create pipeline or updata pipeline version
    
    # params_dict = get_params(args, args.cfg, cfg_dict, tuple_flag)              # parameters for pipeline run
    # run_pipeline(client, experiment_id, pipeline_id, params_dict, pl_cfg.RUN_NAME)
    
    
    
        

