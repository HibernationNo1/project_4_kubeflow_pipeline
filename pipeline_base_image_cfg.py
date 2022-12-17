

class Base_Image_Cfg():
    recode = "hibernation4958/check:0.5"
    recode_cp = "recode"
    
        
"""
docker build utils -t hibernation4958/check:0.5
docker push hibernation4958/check:0.5

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