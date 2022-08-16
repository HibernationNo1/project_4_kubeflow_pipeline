
import os
import json

# kubeflow dashboard
USERNAME = "user@example.com"
PASSWORD = "12341234"
NAMESPACE = "kubeflow-user-example-com"
HOST =  "http://0b87-1-214-32-67.ngrok.io"     # "http://192.168.0.167:80"       

# docker images
SETUP_IMAGE = 'hibernation4958/record:0.4'
SETUP_COM_FILE = None                   # "set_config.component.yaml"

RECORD_IMAGE = 'hibernation4958/record:0.4'
RECORD_COM_FILE = None                  # "record.component.yaml"

SAVE_GS_IMAGE = RECORD_IMAGE            # 'hibernation4958/for_save.0.1'
SAVE_GS_COM_FILE = None                 # "save_dataset.component.yaml"

LOAD_DATA_IMAGE = "hibernation4958/0812:0.2"
LOAD_DATA_COM_FILE = None               # 


# pipeline
PIPELINE_PAC = "mmdet_project.yaml"
PIPELINE_DISCRIPTION = "test"
EXPERIMENT_NAME = "test2"
RUN_NAME = "test_run"


"""
docker build record -t hibernation4958/record:0.4
docker push hibernation4958/record:0.4

docker build train -t hibernation4958/0812:0.2
docker push hibernation4958/0812:0.2


docker pull registry:latest
docker run --name private-docker -d -p 5000:5000 registry
docker build -f record/Dockerfile -t localhost:5000/0809:0.3
docker push localhost:5000/0809:0.3

dataset_0.0.1

"""

# use_aws_secret