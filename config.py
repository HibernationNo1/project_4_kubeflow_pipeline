
# kubeflow dashboard
USERNAME = "user@example.com"
PASSWORD = "12341234"
NAMESPACE = "kubeflow-user-example-com"
HOST =  "http://98cc-1-214-32-67.ngrok.io"     # "http://192.168.0.167:80"       

# docker images
SETUP_IMAGE = 'hibernation4958/0803:0.10'
LABELME_IMAGE = 'hibernation4958/0803:0.10'
SAVES3_IMAGE = 'hibernation4958/0803:0.10'        # 'hibernation4958/for_save.0.1'

# pipeline
PIPELINE_PAC = "mmdet_project.yaml"
PIPELINE_DISCRIPTION = "test"
EXPERIMENT_NAME = "test2"
RUN_NAME = "test_run"


# s3
AWS_ACCESS_KEY_ID ="AKIAXZX44242SIJNTR5O"
AWS_SECRET_ACCESS_KEY = "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"
BUCKET_NAME = "hibernationproject"
    
"""
docker build -t 0803 .
docker tag 0803:latest hibernation4958/0803:0.10
docker push hibernation4958/0803:0.10

dataset_0.0.1
dvc remote add -d storage gdrive://1DjZBugJPcXKytqUaNge8F5P_fJmEvfBu
"""

# use_aws_secret