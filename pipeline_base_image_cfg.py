
BASE_IMG = dict(
    recode = "localhost:5000/docker:0.1",
    train = "localhost:5000/docker:0.1"
    )


class Base_Image_Cfg():
    recode = BASE_IMG['recode']
    recode_cp = "recode/recode.component.yaml"     # TODO: rename for format
    
    train = BASE_IMG['train'] 
    train_cp = "train/train.component.yaml"
    
        
"""
docker run --rm -it localhost:5000/docker:0.2
docker build docker --no-cache -t localhost:5000/docker:0.1


docker build train --no-cache -t localhost:5000/train:0.5
docker build recode --no-cache -t localhost:5000/recode:0.1

docker push localhost:5000/train:0.1

docker build recode --no-cache -t external_ip:port/recode:0.2
docker build recode --no-cache -t hibernation4958/recode:0.2
docker push hibernation4958/recode:0.2

docker build train --no-cache -t hibernation4958/train:0.1
docker push hibernation4958/train:0.1


docker pull registry:latest
docker run --name private-docker -d -p 5000:5000 registry
docker build -f record/Dockerfile -t localhost:5000/0809:0.3
docker push localhost:5000/0809:0.3

dataset_0.0.1

"""