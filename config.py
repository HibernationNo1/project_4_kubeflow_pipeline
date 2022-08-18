
import os
import json


class pipeline_config():
    # kubeflow dashboard
    USERNAME = "winter4958@gmail.com"                  # "user@example.com"
    PASSWORD = "project_pipeline"                          # "12341234"
    NAMESPACE = "project-pipeline-1"        # "kubeflow-user-example-com"
    HOST =  "http://3e60-1-214-32-67.ngrok.io"     # "http://192.168.0.167:80"       

    # docker images
    SETCONFIG_IMAGE = 'hibernation4958/record:0.1'
    SETCONFIG_COM_FILE = None                   # "set_config.component.yaml"

    RECORD_IMAGE = 'hibernation4958/record:0.1'
    RECORD_COM_FILE = None                  # "record.component.yaml"

    SAVE_GS_IMAGE = RECORD_IMAGE            # 'hibernation4958/for_save.0.1'
    SAVE_GS_COM_FILE = None                 # "save_dataset.component.yaml"

    LOAD_DATA_IMAGE = "hibernation4958/train:0.2"
    LOAD_DATA_COM_FILE = None               # 


    # pipeline
    PIPELINE_NAME = "project_test"
    PIPELINE_PAC = "mmdet_project.yaml"
    PIPELINE_DISCRIPTION = "project"
    
    RUN_EXIST_PIPELINE = True   # 이미 upload된 pipeline으로 run을 할 시 True  // 새로운 pipeline을 compile하고 RUN할 계획이면 False
    
    EXPERIMENT_NAME = "test01"
    
    RUN_NAME = "project"


"""
docker build record -t hibernation4958/record:0.1
docker push hibernation4958/record:0.1

docker build train -t hibernation4958/train:0.2
docker push hibernation4958/train:0.2


docker pull registry:latest
docker run --name private-docker -d -p 5000:5000 registry
docker build -f record/Dockerfile -t localhost:5000/0809:0.3
docker push localhost:5000/0809:0.3

dataset_0.0.1

"""

# use_aws_secret