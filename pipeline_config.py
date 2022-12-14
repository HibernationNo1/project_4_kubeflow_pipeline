
kbf = dict(
    dashboard = dict(
        user_n = "winter4958@gmail.com",
        name_space = "project-pipeline",
        host = "http://localhost:8080",
        pw = None
        ),
    pipeline = dict(
        name = "train",
        pac = "project.yaml",
        discription = "project",
        run_uploaded_pipeline = True,      
        version = None
        ),
    experiment = dict(
        name = "test01"
        ),
    run = dict(
        name = "project"
        )
    
    )

secrets = dict(
    gs = dict(
        type = None, 
        project_id = None,
        private_key_id = None,
        private_key = None,
        client_email = None,
        client_id = None,
        auth_uri = None,
        token_uri = None,
        auth_provider_x509_cert_url = None,
        client_x509_cert_url = None   
        ),
    db = dict(
        host = None,
        password = None,
        port = None
        )
    )


pipeline = dict(
    
)
        
"""
docker build utils -t hibernation4958/check:0.6
docker push hibernation4958/check:0.6

docker build record -t hibernation4958/record:0.1
docker push hibernation4958/record_dataset_1:0.1

docker build train -t hibernation4958/train_0906:0.7
docker push hibernation4958/train_0906:0.7



docker pull registry:latest
docker run --name private-docker -d -p 5000:5000 registry
docker build -f record/Dockerfile -t localhost:5000/0809:0.3
docker push localhost:5000/0809:0.3

dataset_0.0.1

"""

# use_aws_secret