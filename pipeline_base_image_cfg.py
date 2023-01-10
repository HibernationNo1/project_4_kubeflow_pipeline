
BASE_IMG = dict(
    recode = "localhost:5000/pipeline:0.1",
    train = "localhost:5000/pipeline:0.1"
    )


class Base_Image_Cfg():
    recode = BASE_IMG['recode']
    recode_cp = "recode/recode.component.yaml"     # TODO: rename for format
    
    train = BASE_IMG['train'] 
    train_cp = "train/train.component.yaml"
    
        
"""
docker build docker --no-cache -t localhost:5000/pipeline:0.1
docker push localhost:5000/pipeline:0.1
docker run --rm -it localhost:5000/pipeline:0.1

"""