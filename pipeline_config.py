
import os
import json


class Pipeline_Config():
    ### kubeflow dashboard
    USERNAME = "winter4958@gmail.com"                  # "user@example.com"
    PASSWORD = "4958"                          # "12341234"
    NAMESPACE = "project-pipeline"        # "kubeflow-user-example-com"
    HOST =  "http://localhost:2222"     # "http://192.168.0.167:80"       

    ### docker images, component file 
    RECORD_IMAGE = 'hibernation4958/record_dataset_1:0.2'
    RECORD_COM_FILE = None                      # "record.component.yaml"

    SAVE_GS_IMAGE = RECORD_IMAGE                # 'hibernation4958/for_save.0.1'
    SAVE_GS_COM_FILE = None                     # "save_dataset.component.yaml"

    LOAD_DATA_IMAGE = "hibernation4958/train_5:0.1"
    LOAD_DATA_COM_FILE = None                   # "load_dataset.component.yaml"
    
    TRAIN_IMAGES = LOAD_DATA_IMAGE
    TRAIN_COM_FILE = None                       # "train.component.yaml"
    
    CHECK_IMAGE = LOAD_DATA_IMAGE
    CHECK_COM_FILE = None                   # "set_config.component.yaml"


    ### pipeline
    # pipeline
    PIPELINE_NAME = "train"
    PIPELINE_PAC = "mmdet_project.yaml"
    PIPELINE_DISCRIPTION = "project"
    
    RUN_EXIST_PIPELINE = True   # 이미 upload된 pipeline으로 run을 할 시 True  // 새로운 pipeline을 compile하고 RUN할 계획이면 False
    
    # experiment
    EXPERIMENT_NAME = "test01"
    
    # run
    RUN_NAME = "project"
    

"""


docker build record -t hibernation4958/record_dataset_1:0.2
docker push hibernation4958/record_dataset_1:0.2

docker build train -t hibernation4958/train_5:0.1
docker push hibernation4958/train_5:0.1



docker pull registry:latest
docker run --name private-docker -d -p 5000:5000 registry
docker build -f record/Dockerfile -t localhost:5000/0809:0.3
docker push localhost:5000/0809:0.3

dataset_0.0.1

"""

# use_aws_secret