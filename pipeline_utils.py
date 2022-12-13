import kfp
import os, os.path as osp
import requests
import warnings

from pipeline_taeuk4958.configs.config import Config

def kfb_print(string, nn = True): 
    if nn:  
        print(f"\nkubeflow>> {string}")
    else:
        print(f"kubeflow>> {string}")
    

def get_params(args, cfg_name, cfg_dict, tuple_flag):
    
    cfg = {'cfg_name': cfg_name, 'cfg_dict' : cfg_dict, 'tuple_flag' : tuple_flag}
    params_dict = {'input_mode': args.mode, 'cfg': cfg}
    return params_dict


def set_config(args, pl_cfg):
    config_py_path = os.path.join(os.getcwd(), 'config', args.cfg)

    cfg_dict, _ = Config._file2dict(config_py_path, None, True)        # local에 있는 config가져오기   
     
    # config를 componunt에 전달하기 위해서는 custom class type이 아닌 dict type으로 전달해야 한다.
    cfg_dict['pipeline']['pipeline_name'] = pl_cfg.PIPELINE_NAME = args.pipeline_n
    cfg_dict['mode'] = args.mode
    
    if args.pipeline_v is not None : cfg_dict['pipeline']['pipeline_version'] = args.pipeline_v
    
    if args.mode == "record":
        if args.proportion_val is not None : cfg_dict['options']['proportion_val'] = args.proportion_val
        if args.dataset is not None : cfg_dict['dataset']['dataset_dir'] = args.dataset
        
        if args.client_secrets is not None : cfg_dict['gs']['client_secrets'] = args.client_secrets
        if args.ann_bk_name is not None : cfg_dict['gs']['ann_bucket_name'] = args.ann_bk_name
        if args.dataset_bk_name is not None : cfg_dict['gs']['recoded_dataset_bucket_name'] = args.dataset_bk_name
        if args.dataset_v is not None : cfg_dict['gs']['recoded_dataset_version'] = args.dataset_v
        
    
    elif args.mode == "train":        
        if args.validate : cfg_dict['train.validate'] = True
        if args.finetun : cfg_dict['finetun'] = True
        if args.epo is not None: cfg_dict['runner']['max_epochs'] = args.epo        
        
        cfg_dict['model_version'] = args.model_v
        cfg_dict['seed'] = args.seed
        cfg_dict['deterministic'] = args.deterministic

        # only support single GPU mode in non-distributed training.
        cfg_dict['gpu_ids'] = [0]
        
    tuple_flag = get_tuple_key(cfg_dict)
    
 
    
    return cfg_dict, tuple_flag

def upload_pipeline(client, cfg):
    pipeline_path = os.path.join(os.getcwd(), cfg.pac)
    if not osp.isfile(pipeline_path) : raise OSError(f"upload file : `{pipeline_path}` is not exist! ")
    
    pipeline_id = client.get_pipeline_id(cfg.name)
    if pipeline_id is None:
        kfb_print(f"Upload initial version pipeline | pipeline name: `{cfg.name}` ")
        cfg.version = "0.0"
        client.upload_pipeline(pipeline_package_path= pipeline_path,            
                               pipeline_name= cfg.name,
                               description= cfg.discription)
        pipeline_id = client.get_pipeline_id(cfg.name)
        return pipeline_id        
   

    pipelince_versions = client.list_pipeline_versions(pipeline_id = pipeline_id, page_size = 50)
    versions = []  
    for pipeline_index in range(pipelince_versions.total_size):
        versions.append(pipelince_versions.versions[pipeline_index].name) 
    
    if cfg.version in versions:
        raise ValueError(f" `{cfg.version}` version of pipeline(name:`{cfg.name}`) is exist! ")
    else:
        kfb_print(f"Upload pipeline; version update | pipeline name: `{cfg.name}`, version: `{cfg.version}`")
        client.upload_pipeline_version(pipeline_package_path= pipeline_path,            # pipeline version updata
                                    pipeline_version_name = cfg.version,
                                    pipeline_id = pipeline_id,
                                    description = cfg.discription)  
        
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
        
        
def upload_pipeline_(client, cfg):
    pipeline_path = os.path.join(os.getcwd(), cfg.pac)
    if not os.path.isfile(pipeline_path) : raise OSError(f"upload file : `{pipeline_path}` is not exist! ")
    
    if client.get_pipeline_id(cfg.name) == None:                
        print(f"\n Upload initial version pipeline named [{cfg.PIPELINE_NAME}]!",end = " " )           
        client.upload_pipeline(pipeline_package_path= pipeline_path,            # upload new pipeline
                            pipeline_name= cfg.PIPELINE_NAME,
                            description= cfg.PIPELINE_DISCRIPTION)
        print("success!")
        pipeline_id = client.get_pipeline_id(cfg.PIPELINE_NAME)
    else:                                                                         
        pipeline_id = client.get_pipeline_id(cfg.PIPELINE_NAME)
        pipelince_versions = client.list_pipeline_versions(pipeline_id = pipeline_id, page_size = 50)
        
        versions = []  
        for pipeline_index in range(pipelince_versions.total_size):
            versions.append(pipelince_versions.versions[pipeline_index].name) 
        
        if cfg.version in versions: raise TypeError(f" {cfg.version} version of {cfg.PIPELINE_NAME} is exist! ")
                
        print(f"\n Upload pipeline | version: {cfg.version}, name: {cfg.PIPELINE_NAME} : ", end = " ")
        client.upload_pipeline_version(pipeline_package_path= pipeline_path,            # pipeline version updata
                                    pipeline_version_name = cfg.version,
                                    pipeline_id = pipeline_id,
                                    description = cfg.PIPELINE_DISCRIPTION)    
        print("seccess!")  
        
        
    return pipeline_id  

def get_experiment(client, cfg) : 
    list_experiments = client.list_experiments(page_size = 50)
    if list_experiments.total_size == None:
        print(f"There no experiment. create experiment | name: `{cfg.kbf.experiment.name}`")
        experiment = client.create_experiment(name = cfg.kbf.experiment.name)
    else:
        experiment = client.get_experiment(experiment_name= cfg.kbf.experiment.name, namespace= cfg.kbf.dashboard.name_space)
    
    experiment_id = experiment.id
    return experiment_id


def connet_client(cfg):   
    session = requests.Session()
    response = session.get(cfg.host)

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {"login": cfg.user_n, "password": cfg.pw}
    session.post(response.url, headers=headers, data=data)                              
    session_cookie = session.cookies.get_dict()["authservice_session"]  
    
   
    # access kubeflow dashboard
    client = kfp.Client(
        host=f"{cfg.host}/pipeline",
        namespace=f"{cfg.name_space}",
        cookies=f"authservice_session={session_cookie}",
    )
    return client


def get_tuple_key(cfg):     # config안에서 tuple type인 key 또는 idx를 반환
    if isinstance(cfg, dict):
        tmp_dict = {}
        for key in list(cfg.keys()):
            is_tuple = get_tuple_key(cfg[key]) 
            
            if is_tuple :
                tmp_dict[key] = is_tuple
            else: continue

        
        return tmp_dict     
    elif isinstance(cfg, tuple):
        return True
    
    elif isinstance(cfg, list):
        tmp_list = []
        for i, ele in enumerate(cfg):       # list에 tuple이 포함되어 있는 경우
            is_tuple = get_tuple_key(ele)
            if isinstance(is_tuple, dict):
                tmp_list.append(is_tuple)
            elif isinstance(is_tuple, bool) and is_tuple:
                tmp_list.append(i)      
            
            
            else: continue
            
        
        if len(tmp_list) == 0: return False
        return tmp_list
    
    else: return False