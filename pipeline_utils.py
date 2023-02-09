import kfp
import os, os.path as osp
import requests

from pipeline_base_config import BASE_IMG, Config
from pipeline_config import CONFIGS


def kfb_print(string, nn = True): 
    if nn:  
        print(f"\nkubeflow>> {string}")
    else:
        print(f"kubeflow>> {string}")
    


def upload_pipeline(client, cfg):
    pipeline_path = os.path.join(os.getcwd(), cfg.pac)
    
    pipeline_id = client.get_pipeline_id(cfg.name)
    if pipeline_id is None:             # Upload initial version pipeline
        if not osp.isfile(pipeline_path) : raise OSError(f"upload file : {pipeline_path} is not exist! ")
        
        kfb_print(f"Upload pipeline:: initial version | pipeline name: {cfg.name} ")
        # cfg.version = "0.0"
        client.upload_pipeline(pipeline_package_path= pipeline_path,            
                               pipeline_name= cfg.name,
                               description= cfg.discription)
        pipeline_id = client.get_pipeline_id(cfg.name)
        return pipeline_id        
    
    pipelince_versions = client.list_pipeline_versions(pipeline_id = pipeline_id, page_size = 50)
  
    version_idx = None
    for pipeline_index in range(pipelince_versions.total_size):        
        if cfg.version == pipelince_versions.versions[pipeline_index].name:
            version_idx = pipeline_index
    

    if version_idx is not None:            # if cfg.version uploaded before
        if cfg.run_uploaded_pipeline:      # get version id that was uploaded pipeline
            version_id = pipelince_versions.versions[version_idx].id    
            kfb_print(f"This pipeline:{cfg.name}(verson:{cfg.version}) uploaded before. get this pipeline version ID .")
            return dict(version_id = version_id)
        else: 
            raise ValueError(f"{cfg.version} version of pipeline(name:{cfg.name}) is exist! ")
    else:
        if not osp.isfile(pipeline_path) : raise OSError(f"upload file : {pipeline_path} is not exist! ")
        # pipeline version updata
        kfb_print(f"Upload pipeline:: version update | pipeline name: {cfg.name}, version: {cfg.version}")
        client.upload_pipeline_version(pipeline_package_path= pipeline_path,            
                                    pipeline_version_name = cfg.version,
                                    pipeline_id = pipeline_id,
                                    description = cfg.discription)  
   
    return pipeline_id          

    

def run_pipeline(client, cfg, experiment_id, pipeline_id, params):
    
    
    if isinstance(pipeline_id, dict):       # TODO: Unrecognized input parameter: 뜬다.
        kfb_print(f"Run pipeline:: this version uploaded before | name: {cfg.pipeline.name}, version: {cfg.pipeline.version}")
        exec_run = client.run_pipeline(
            experiment_id = experiment_id,
            job_name = cfg.run.name,
            version_id = pipeline_id["version_id"],
            params = params
            )
    else:
        kfb_print(f"Run pipeline:: this version is first upload | name: {cfg.pipeline.name}, version: {cfg.pipeline.version}")
        exec_run = client.run_pipeline(
                experiment_id = experiment_id,
                job_name = cfg.run.name,
                pipeline_id = pipeline_id,
                params = params
                )
    

    # run_id = exec_run.id
    # completed_run = client.wait_for_run_completion(run_id=run_id, timeout=600)
    # kfb_print(f"status of {cfg.run.name} : {completed_run.run.status}")
    


def get_experiment(client, cfg) : 
    list_experiments = client.list_experiments(page_size = 50)
    
    if list_experiments.total_size == None:
        kfb_print(f"There no experiment. create experiment | name: {cfg.kbf.experiment.name}", nn= False)
        experiment = client.create_experiment(name = cfg.kbf.experiment.name)
    else:
        for i, experiment_info in enumerate(list_experiments.experiments):
            if cfg.kbf.experiment.name == experiment_info.name:
                experiment = client.get_experiment(experiment_name= cfg.kbf.experiment.name, namespace= cfg.kbf.dashboard.name_space)
                break
            
            if i == list_experiments.total_size-1:
                kfb_print(f"There is no experiment named {cfg.kbf.experiment.name}. Create experiment.", nn = False)
                experiment = client.create_experiment(name = cfg.kbf.experiment.name)
        
    
    experiment_id = experiment.id
    return experiment_id



def connet_client(cfg, return_session = False):   
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
        cookies=f"authservice_session={session_cookie}")
    if return_session:
        return client, session
    else:
        return client 




def set_intput_papams(pipeline = True):
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
                if pipeline:
                    kfb_print(f"{key}_op base_image : {BASE_IMG[key]}", nn=False)
                params[f"{key}_using"] = True
            else:    
                
                if item.get('path', None) is not None:
                    if pipeline:
                        item.path.local_package = None
                    else:
                        item.path.volume = None
                        item.path.work_space = None
                        item.path.tensorboard = None
                        
                params[f"cfg_{key}"] = dict(item)
                params[f"cfg_{key}"]['flag'] = get_tuple_key(item)
    
    convert(True)
    convert(False)
    
    # params.keys(): 
    # ['train_using', 'recode_using', 'cfg_train', 'cfg_recode']
    # If 'recode' is not selected as a pipeline component, it has a value of `False`.
    
    key = 'test'    # 'validation'  TODO: run test, validation in pipeline
    if pipeline:
        params.pop(f'cfg_{key}')
        params.pop(f'{key}_using')
    
    return params


def get_tuple_key(cfg):    
    """ 
        return boolean flag equal to the dict map of input 'cfg'. 
        flag gets True or index where if type of key and type of value in list are tuple.

    Args:
        cfg (_type_): config dict

    Returns:
        _type_: config dict, all value are boolean.
    """

    if isinstance(cfg, dict) or isinstance(cfg, Config):
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