import kfp
import os, os.path as osp
import requests
import warnings

from hibernation_no1.configs.config import Config

CONFIGS = dict()     # parameters for pipeline run  

def set_cfg_recode(args, cfg):
    assert cfg.dvc.recode.train == cfg.recode.train_dataset
    assert cfg.dvc.recode.val == cfg.recode.val_dataset
   
def set_cfg_pipeline(args, cfg):
    if args.pipeline_n is not None: cfg.kbf.pipeline.name = args.pipeline_n
    cfg.kbf.pipeline.version =  args.pipeline_v
    if args.experiment_n is not None: cfg.kbf.experiment.name = args.experiment_n
    if args.run_n is not None: cfg.kbf.run.name = args.run_n    
    cfg.kbf.dashboard.pw =  args.pipeline_pw 
    
    
     
def set_cfg_train(args, cfg):
    if args.name_db is not None: cfg.db.db = args.name_db 
    if args.user_db is not None: cfg.db.user = args.user_db 
    
    if args.model_name is not None: cfg.filename_tmpl = f"{args.model_name}"+"_{}.path"
    if args.val_iter is not None: 
        for i, custom_hook in enumerate(cfg.custom_hook_config):
            if custom_hook.get('Validation_Hook', None) is not None:
                 cfg.custom_hook_config[i].val = args.val_iter
                 break
    
    # If get dataset with dvc, load the paths from the database.
    # And all paths were set by dvc config
    if cfg.get('dvc', None) is not None:
        cfg.data.train.data_root = cfg.data.val.data_root = osp.join(cfg.dvc.category, 
                                                                     cfg.dvc.recode.name, 
                                                                     cfg.dvc.recode.version)
        cfg.data.train.ann_file = cfg.dvc.recode.train
        cfg.data.val.ann_file = cfg.dvc.recode.val
    

    if args.pm_dilation is not None: cfg.model.backbone.pm_dilation = args.pm_dilation
    if args.drop_rate is not None: cfg.model.backbone.drop_rate = args.drop_rate
    if args.drop_path_rate is not None: cfg.model.backbone.drop_path_rate = args.drop_path_rate
    if args.attn_drop_rate is not None: cfg.model.backbone.attn_drop_rate = args.attn_drop_rate    

CONFIG_SET_FUNCTION = dict(
    pipeline = set_cfg_pipeline,
    recode = set_cfg_recode,
    train = set_cfg_train
)


def set_config(args):
    """ 
        cfg arguments determines which component be run.
        Components that matching cfg arguments which got `None` are excluded from the pipeline.
        cfg arguments: is chooses in [args.cfg_train, args.cfg_recode]
    Args:
        args : argparse
    """

    CONFIGS['pipeline'] = args.cfg_pipeline
    CONFIGS['train'] = args.cfg_train
    CONFIGS['recode'] = args.cfg_recode
    
    for key, func in CONFIG_SET_FUNCTION.items():
        if CONFIGS[key] is not None:
            # Assign config only included in args 
            config =  Config.fromfile(CONFIGS[key])
            func(args, config)
        else: config = None
        # CONFIGS[key] = False or Config
        # if False, components matching the key will be excluded from the pipeline.
        # >>    example
        # >>    CONFIGS[recode] = False
        # >>    `recode_op` component will be excluded from the pipeline.
        CONFIGS[key] = config

    


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
    

    run_id = exec_run.id
    
    completed_run = client.wait_for_run_completion(run_id=run_id, timeout=600)
    kfb_print(f"status of {cfg.run.name} : {completed_run.run.status}")
    

def get_experiment(client, cfg) : 
    list_experiments = client.list_experiments(page_size = 50)
    if list_experiments.total_size == None:
        print(f"There no experiment. create experiment | name: {cfg.kbf.experiment.name}")
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



    