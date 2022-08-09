
import os
import json

# kubeflow dashboard
USERNAME = "user@example.com"
PASSWORD = "12341234"
NAMESPACE = "kubeflow-user-example-com"
HOST =  "http://ebbc-1-214-32-67.ngrok.io"     # "http://192.168.0.167:80"       

# docker images
SETUP_IMAGE = 'hibernation4958/0809:0.2'
SETUP_COM_FILE = "set_config.component.yaml"

RECORD_IMAGE = 'hibernation4958/0809:0.2'
RECORD_COM_FILE = "lebelme.component.yaml"

SAVE_GS_IMAGE = 'hibernation4958/0809:0.2'        # 'hibernation4958/for_save.0.1'
SAVE_GS_COM_FILE = "save_dataset.component.yaml"

# pipeline
PIPELINE_PAC = "mmdet_project.yaml"
PIPELINE_DISCRIPTION = "test"
EXPERIMENT_NAME = "test2"
RUN_NAME = "test_run"


# GS
BUCKET_NAME = "dataset_taeuk4958"
secrets_path = os.path.join(os.getcwd(), "client_secrets.json")
with open(secrets_path, "r", encoding='utf-8') as f:
        secrets_dict = json.load(f)
SECRETS_DICT = secrets_dict


"""
docker build record -t hibernation4958/0809:0.2
docker push hibernation4958/0809:0.2

docker pull registry:latest
docker run --name private-docker -d -p 5000:5000 registry
docker build -f record/Dockerfile -t localhost:5000/0809:0.2
docker push localhost:5000/0809:0.2

dataset_0.0.1

"""

# use_aws_secret