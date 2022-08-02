
# kubeflow dashboard
USERNAME = "user@example.com"
PASSWORD = "12341234"
NAMESPACE = "kubeflow-user-example-com"
HOST =  "http://6957-1-214-32-67.ngrok.io"     # "http://192.168.0.167:80"       

# docker images
SETUP_IMAGE = 'hibernation4958/0802:0.9'
LABELME_IMAGE = 'hibernation4958/0802:0.9'
SAVES3_IMAGE = 'hibernation4958/0802:0.9'        # 'hibernation4958/for_save.0.1'

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
docker build -t 0802 .
docker tag 0802:latest hibernation4958/0802:0.9
docker push hibernation4958/0802:0.9
"""

# use_aws_secret