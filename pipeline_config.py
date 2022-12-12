
kbf = dict(
    dashboard = dict(
        user_n = "winter4958@gmail.com",
        name_space = "project-pipeline",
        host = "http://122.44.146.126:4958"
        ),
    pipeline = dict(
        name = "train",
        pac = "project.yaml",
        discription = "project",
        run_exist_pipeline = True
        ),
    experiment = dict(
        name = "test01"
        ),
    run = dict(
        name = "project"
        )
    )

gs = dict(
    client_secrets = "client_secrets.json"
)


"""
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