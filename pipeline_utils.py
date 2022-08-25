import kfp
import os
import requests

from pipeline_taeuk4958.configs.config import Config

def get_params(args, cfg_name, cfg_dict):
            
    params_dict = {'input_mode': args.mode, 'cfg_name': cfg_name, 'cfg_dict' : cfg_dict}
        
    return params_dict


def set_config(args):
    
    config_py_path = os.path.join(os.getcwd(), 'config', args.cfg)

    cfg_dict, _ = Config._file2dict(config_py_path, None, True)        # local에 있는 config가져오기
    
    cfg_dict['pipeline'].pipeline_name = args.p_name
    if args.pipeline_v is not None : cfg_dict['pipeline'].pipeline_version = args.pipeline_v
    
    if args.mode == "record":
        if args.proportion_val is not None : cfg_dict['options']['proportion_val'] = args.proportion_val
        if args.dataset is not None : cfg_dict['dataset']['dataset_dir'] = args.dataset
        
        if args.client_secrets is not None : cfg_dict['gs']['client_secrets'] = args.client_secrets
        if args.ann_bk_name is not None : cfg_dict['gs']['ann_bucket_name'] = args.ann_bk_name
        if args.dataset_bk_name is not None : cfg_dict['gs']['recoded_dataset_bucket_name'] = args.dataset_bk_name
        if args.dataset_v is not None : cfg_dict['gs']['recoded_dataset_version'] = args.dataset_v
    
    elif args.mode == "train":

        if args.train_json is not None : cfg_dict['train_dataset_json'] = args.train_json
        if args.val_json is not None : cfg_dict['val_dataset_json'] = args.val_json
        if args.validate : cfg_dict['train.validate'] = True
        if args.finetun : cfg_dict['train']['finetun'] = True
        if args.model_v is not None : cfg_dict['train']['model_version'] = args.model_v
    
    return cfg_dict

def get_pipeline_id(pl_cfg, args, client):
    if pl_cfg.RUN_EXIST_PIPELINE:                                       # if you want version updata of pipeline (modified pipeline)
        print(f" \n get pipeline: {pl_cfg.PIPELINE_NAME}.{args.pipeline_v} id ")
        pipeline_id = client.get_pipeline_id(pl_cfg.PIPELINE_NAME)
        
        list_pipeline_name = []
        list_pipelines = client.list_pipelines(page_size = 50)  	
        
        for pipeline_index in range(list_pipelines.total_size):
            list_pipeline_name.append(list_pipelines.pipelines[pipeline_index].name) 
        

        if pl_cfg.PIPELINE_NAME not in list_pipeline_name:
            pipeline_id = upload_pipeline(client, args, pl_cfg)
            return pipeline_id
        
        pipelince_versions = client.list_pipeline_versions(pipeline_id = pipeline_id, page_size = 50)
        
        versions = []  
        for pipeline_index in range(pipelince_versions.total_size):
            versions.append(pipelince_versions.versions[pipeline_index].name) 
            
        
        
        if args.pipeline_v not in versions:                              
            print(f" RUN_EXIST_PIPELINE is True, but [version:{args.pipeline_v}] is not exist.")
            print(f" upload {pl_cfg.PIPELINE_NAME} new version : {args.pipeline_v}")
            pipeline_id = upload_pipeline(client, args, pl_cfg)
            return pipeline_id
            
    else: 
        pipeline_id = upload_pipeline(client, args, pl_cfg) 
    
    return pipeline_id          

    
    

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
        
        
def upload_pipeline(client, args, pl_cfg):
    pipeline_path = os.path.join(os.getcwd(), pl_cfg.PIPELINE_PAC)
    if not os.path.isfile(pipeline_path) : raise OSError(f" {pipeline_path} is not exist! ")
    
    if client.get_pipeline_id(pl_cfg.PIPELINE_NAME) == None:                
        print(f"\n Upload initial version pipeline named {pl_cfg.PIPELINE_NAME}!",end = " " )           
        client.upload_pipeline(pipeline_package_path= pipeline_path,            # upload new pipeline
                            pipeline_name= pl_cfg.PIPELINE_NAME,
                            description= pl_cfg.PIPELINE_DISCRIPTION)
        print("success!")
        pipeline_id = client.get_pipeline_id(pl_cfg.PIPELINE_NAME)
    else:                                                                         
        pipeline_id = client.get_pipeline_id(pl_cfg.PIPELINE_NAME)
        pipelince_versions = client.list_pipeline_versions(pipeline_id = pipeline_id, page_size = 50)
        
        versions = []  
        for pipeline_index in range(pipelince_versions.total_size):
            versions.append(pipelince_versions.versions[pipeline_index].name) 
        
        if args.pipeline_v in versions: raise TypeError(f" {args.pipeline_v} version of {pl_cfg.PIPELINE_NAME} is exist! ")
                
        print(f"\n Upload pipeline | version: {args.pipeline_v}, name: {pl_cfg.PIPELINE_NAME} : ", end = " ")
        client.upload_pipeline_version(pipeline_package_path= pipeline_path,            # pipeline version updata
                                    pipeline_version_name = args.pipeline_v,
                                    pipeline_id = pipeline_id,
                                    description = pl_cfg.PIPELINE_DISCRIPTION)    
        print("seccess!")  
        
        
    return pipeline_id  

def get_experiment(client, pl_cfg) : 
    list_experiments = client.list_experiments(page_size = 50)
    if list_experiments.total_size == None:
        print(f"There no experiment. create experiment | name: {pl_cfg.EXPERIMENT_NAME}")
        experiment = client.create_experiment(name = pl_cfg.EXPERIMENT_NAME)
    else:
        experiment = client.get_experiment(experiment_name= pl_cfg.EXPERIMENT_NAME, namespace= pl_cfg.NAMESPACE)
    
    experiment_id = experiment.id
    return experiment_id


def connet_client(pl_cfg):   
    session = requests.Session()
    response = session.get(pl_cfg.HOST)

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {"login": pl_cfg.USERNAME, "password": pl_cfg.PASSWORD}
    session.post(response.url, headers=headers, data=data)                              
    session_cookie = session.cookies.get_dict()["authservice_session"]  

    # access kubeflow dashboard
    client = kfp.Client(
        host=f"{pl_cfg.HOST}/pipeline",
        namespace=f"{pl_cfg.NAMESPACE}",
        cookies=f"authservice_session={session_cookie}",
    )
    return client