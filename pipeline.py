
import kfp
import kfp.dsl as dsl
 
import argparse

from component.record.record_op import record_op
from component.train.train_op import train_op
from component.evaluate.evaluate_op import evaluate_op
from component.test.test_op import test_op


from pipeline_config import set_config, CONFIGS
from pipeline_utils import (connet_client, get_experiment, run_pipeline, upload_pipeline, set_intput_papams, 
                            kfb_print)


from kubernetes.client.models import V1EnvVar, V1EnvVarSource, V1SecretKeySelector
from kubernetes.client import V1Volume, V1EmptyDirVolumeSource

SECRETS = dict()

@dsl.pipeline(name="hibernation_project")
def project_pipeline(run_flag: dict, 
                     input_cfg: dict):           

    # persistance volume
    pvc_cfg = CONFIGS['pipeline'].kbf.volume.pvc
    pvc_volume = dsl.VolumeOp(name= pvc_cfg.name,
                       resource_name= pvc_cfg.resource_name,
                       modes= pvc_cfg.mode,
                       storage_class = pvc_cfg.storage_class,
                       size= pvc_cfg.size)
    
    
    # for allocate shared memory
    shm_volume_cfg = CONFIGS['pipeline'].kbf.volume.share_memory
    shm_volume = dsl.PipelineVolume(
        volume=V1Volume(
            name= shm_volume_cfg.name,
            empty_dir=V1EmptyDirVolumeSource(medium=shm_volume_cfg.medium))
        )  

    
    # set secrets
    client_sc_name = "client-secrets"
    for secrets_cate, secrets_cfg in SECRETS.items():
        for key in secrets_cfg:
            SECRETS[secrets_cate][key] = V1EnvVar(name=key, value_from=V1EnvVarSource(secret_key_ref=V1SecretKeySelector(name=client_sc_name, key=key)))
       
    _record_op = record_op(input_cfg, run_flag) \
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
        .add_pvolumes({pvc_cfg.mount_path: pvc_volume.volume})
    
    _train_op = train_op(input_cfg, _record_op.outputs['run_flag'])\
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
        .add_pvolumes({shm_volume_cfg.path: shm_volume})\
        .add_pvolumes({pvc_cfg.mount_path: pvc_volume.volume})

    _evaluate_op = evaluate_op(input_cfg, _train_op.outputs['run_flag'])\
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
        .add_pvolumes({shm_volume_cfg.path: shm_volume})\
        .add_pvolumes({pvc_cfg.mount_path: pvc_volume.volume})

    _test_op = test_op(input_cfg, _evaluate_op.outputs['run_flag'])\
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
        .add_pvolumes({shm_volume_cfg.path: shm_volume})\
        .add_pvolumes({pvc_cfg.mount_path: pvc_volume.volume})


def _parse_args():
    
    parser = argparse.ArgumentParser()    
    parser.add_argument("--cfg_pipeline", help="name of config file which for pipeline")       
    parser.add_argument("--cfg_train", help="name of config file which for training")                           # TODO: rename
    parser.add_argument("--cfg_record", help="name of config file which for record") 
    parser.add_argument("--cfg_test", help="name of config file which for test")  
    parser.add_argument("--cfg_eval", help="name of config file which for evaluate") 
    
    kbf_parser = parser.add_argument_group('kubeflow')
    kbf_parser.add_argument("--dashboard_pw", type = str , help="password of kubeflow dashboard")        
    kbf_parser.add_argument("--pipeline_v", type = str, help="version of pipeline")                    
    kbf_parser.add_argument("--pipeline_n", type = str, help="name of pipeline")    
    kbf_parser.add_argument("--experiment_n", type = str, help="name of experiment") 
    kbf_parser.add_argument("--run_n", type = str, help="name of run") 

    kbf_parser.add_argument("--katib", action = 'store_true',  help="If run for experiment") 
    # gs_parser = parser.add_argument_group('google_storage')


    db_parser = parser.add_argument_group('database')
    db_parser.add_argument('--name_db', type = str, help = 'Database name to connect to database')
    db_parser.add_argument('--user_db', type = str, help = 'User name to connect to database')
    
    
    train_parser = parser.add_argument_group('train')
    train_parser.add_argument("--model", type = str, choices = ['MaskRCNN'],
                              help="Name of the model to be trained") 
    train_parser.add_argument("--epoch", type = int, help="epoch for training") 
    train_parser.add_argument("--lr", type = str, choices = ['0.0001', '0.0005', '0.001', '0.00005', '0.00001', '0.000005'],
                              help="Regular learning rate")             # why str?: To get linear value with katib.
    train_parser.add_argument("--swin_drop_rate", type = str, choices = ['0.0', '0.1', '0.2', '0.3', '0.4'],
                               help="drop_rate of swin transformar")         # why str?: To get linear value with katib.
    train_parser.add_argument("--swin_window_size", type = str, choices = ['5', '7', '9', '11', '13'],
                              help="window_size of swin transformar")     # why str?: To get linear value with katib.
    train_parser.add_argument("--swin_mlp_ratio", type = str, choices = ['3', '4', '5'],
                              help="mlp_ratio of swin transformar")         # why str?: To get linear value with katib.
    
    
    
    test_parser = parser.add_argument_group('test')
    test_parser.add_argument("--model_path", type = str, help = "Path of trained model(.pth format)")
    
    
    swin_parser = parser.add_argument_group('SwinTransformer')
    swin_parser.add_argument('--pm_dilation', type = int, help= "dilation of SwinTransformer.PatchMerging") 
    swin_parser.add_argument('--drop_rate', type = float, help= "drop_rate of SwinTransformer") 
    swin_parser.add_argument('--drop_path_rate', type = float, help= "drop_path_rate of SwinTransformer") 
    swin_parser.add_argument('--attn_drop_rate', type = float, help= "attn_drop_rate of SwinTransformer.SwinBlockSequence.ShiftWindowMSA.WindowMSA") 
    
    args = parser.parse_args()
    input_args = vars(args)     # TODO delete
    
    return args, input_args




if __name__=="__main__":      
    args, input_args = _parse_args()  
    set_config(args)

    
    SECRETS['gs'] = dict(CONFIGS['pipeline'].secrets.gs)  
    SECRETS['db'] = dict(CONFIGS['pipeline'].secrets.db)  
    

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

    run_pipeline(client, cfg_pipeline.kbf, experiment_id, pipeline_id, params)
    
    
    
        

