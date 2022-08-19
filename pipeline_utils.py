import kfp
import os
import requests

def get_params(args, input_dict):
            
    params_dict = {'input_mode': args.mode, 'input_args': input_dict}
        
    return params_dict


def get_pipeline_id(pl_cfg, args, client):
    if pl_cfg.RUN_EXIST_PIPELINE:                                       # if you want version updata of pipeline (modified pipeline)
        print(f" get pipeline: {pl_cfg.PIPELINE_NAME}.{args.p_version} id ")
        pipeline_id = client.get_pipeline_id(pl_cfg.PIPELINE_NAME)
        pipelince_versions = client.list_pipeline_versions(pipeline_id = pipeline_id, page_size = 50)
        
        versions = []  
        for pipeline_index in range(pipelince_versions.total_size):
            versions.append(pipelince_versions.versions[pipeline_index].name) 
            
        if args.p_version not in versions:                              
            print(f" RUN_EXIST_PIPELINE is True but [version:{args.p_version}] is not exist.")
            print(f" upload {pl_cfg.PIPELINE_NAME} new version : {args.p_version}")
            pipeline_id = upload_pipeline(client, args, pl_cfg)
            
    else: pipeline_id = upload_pipeline(client, args, pl_cfg)           

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
        
        if args.p_version in versions: raise TypeError(f" {args.p_version} version of {pl_cfg.PIPELINE_NAME} is exist! ")
                
        print(f"\n Upload pipeline {args.p_version} version named {pl_cfg.PIPELINE_NAME} : ", end = " ")
        client.upload_pipeline_version(pipeline_package_path= pipeline_path,            # pipeline version updata
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