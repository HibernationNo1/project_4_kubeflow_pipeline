

class Base_Image_Cfg():
    recode = "hibernation4958/recode:0.2"
    recode_cp = "recode/recode.log"     
    
    train = "hibernation4958/train:0.2"
    train_cp = "train/train"
    
        
"""
docker build recode --no-cache -t hibernation4958/recode:0.2
docker push hibernation4958/recode:0.3

docker build train --no-cache -t hibernation4958/train:0.1
docker push hibernation4958/train:0.1


docker pull registry:latest
docker run --name private-docker -d -p 5000:5000 registry
docker build -f record/Dockerfile -t localhost:5000/0809:0.3
docker push localhost:5000/0809:0.3

dataset_0.0.1

"""