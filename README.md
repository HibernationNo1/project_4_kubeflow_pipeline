

# Project: Automatic License Plate Recognition

## Table of Contents



## Overview

해당 프로젝트는 kubernetes상에서 kubeflow를 활용하여 모델을 학습하는 pipeline을 구현한 프로젝트입니다.

전체적인 흐름도는 아래와 같습니다.



![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/overview_.png?raw=true)

1. **Collect dataset and labeling**

   dataset을 수집하고 라벨링을 진행합니다.

2. **Integrate `annotation dataset` into` coco data` set form**

   mmdetection model의 학습 데이터로 사용하기 위해 coco dataset과 같은 구조로 데이터셋을 구성합니다.

3. **Model training and evaluation**

   model을 학습하고 evaluation과 inference test를 진행합니다.

   model을 통해 기대하는 inference결과는 번호판의 각 text를 인식하고  **등록지역(두 자리 번호)**, **차종기호(A~E 중 하나)**, **일련번호(네 자리 번호)**를 각각 추출해내는 것입니다.

   ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/License%20plate%20desc.png?raw=true)

   ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/License%20plate%20desc_1.png?raw=true)

4. Model serving

   model을 배포합니다.

   > 구현중에 있습니다.





---



## Managing dataset 

- 자동차 번호판을 학습시키는 것을 trining의 목적으로 했습니다.

- dataset은 `labelme.exe`를 통해 직접 만들었으며, DVC를 이용해 version관리를 하였고 DB에서 data의 정보를 관리했습니다.
- 실제 자동차 번호판을 모아놓은 데이터 셋을 만들고자 했지만, 자동차 소유주들에게 자문을 구한 결과 '모르는 사람이 자신의 자동차의 번호판을 촬영해간다면 매우 불쾌할 수 있다.'는 의견을 듣고, python을 사용해 배경 이미지에 임의로 번호판을 그려 dataset을 구성하는 것으로 방향을 잡게 되었습니다. 



### Process Of creating Datasets

1. 핸드폰으로 사진을 찍습니다.

2. python code를 통해 사진 위 무작위 좌표에 간단한 번호판 그림을 그려 넣습니다.

   ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/%EB%A7%8C%EB%93%A0%20%EC%9D%B4%EB%AF%B8%EC%A7%80%201.png?raw=true)

   [drawing code](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/tmp_code/draw_board.py) : image위에 임의의 번호판을 그리는 code입니다.

   번호판의 다양한 color, type, size을 결정 후 랜덤하게 할당되도록 했습니다. 

3. `Labelme.exe`를 통해 번호판이 그려진 이미지에 라벨링 작업을 진행합니다.

   ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/labeling.png?raw=true)

4. 라벨링을 완료한 파일들은 [pipeline_dataset](https://github.com/HibernationNo1/pipeline_dataset.git) repository로 관리합니다.

   1. 라벨링 작업이 완료된 dataset을 DVC를 활용하여 version을 관리하며, dvc명령어에 의해 google cloud에 push합니다.
   2. [pipeline_dataset](https://github.com/HibernationNo1/pipeline_dataset.git)의 [main.py](https://github.com/HibernationNo1/pipeline_dataset/blob/master/main.py) code실행 시 pymysql를 통해 서버에 구축한 DB에 각 image의 path 및 infomation을 기록하여 관리합니다. 
   3. tag를 통해 **annotation dataset**과 training dataset의 commit을 구분합니다.

   해당 reposigoty의 설명은 [README.md](https://github.com/HibernationNo1/pipeline_dataset/blob/master/README.md)를 통해 확인하실 수 있습니다.

   ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/dataset%20init.png?raw=true)

   

---



## Pipeline Configuration With Kubeflow

### 1. Customizing mmdetection, mmcv

model의 학습 code는 [open-mmlab](https://github.com/open-mmlab)/[mmdetection](https://github.com/open-mmlab/mmdetection) 과 [open-mmlab](https://github.com/open-mmlab)/[mmcv](https://github.com/open-mmlab/mmcv)를 차용했으며, 몇 가지 수정을 통해 원하는 동작을 추가했습니다.

![](https://github.com/open-mmlab/mmdetection/raw/main/resources/mmdet-logo.png)

![](https://raw.githubusercontent.com/open-mmlab/mmcv/main/docs/en/mmcv-logo.png)

1. Model load시 추가적인 infomation을 요구할 것 없이 model내부에서 내용을 가져와 training 및 inference를 진행할 수 있게 했습니다.
2. Object Detction의 성능 평가 지표인 **mAP**를 한 단계 더욱 고차원적인 방법으로 계산하는 성능 평가 지표를 만들었습니다.
3. 제가 만든 dataset를 위한 model 평가 지표를 추가했습니다.

해당 내용에 관한 설명이 길기 때문에, 자세한 내용은 **[이곳](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/customizing%20mmdetection%2C%20mmcv.md)**에 정리해두었습니다. 링크를 클릭하시면 확인하실 수 있습니다.



### 2. kubeflow

#### Experiments(Katib-AutoML)

Hyper-prameter tuning을 진행하기 위해 Kubeflow의 구성 요소인 Katib system을 사용했습니다.

![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/Experiment.png?raw=true)

hyper parameter는 `learning rate`와 `backbone model`의 몇 가지 parameter만 조정했습니다.

해당 내용에 관한 설명이 길기 때문에, 자세한 내용은 **[이곳](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/Experiments(Katib-AutoML).md)**에 정리해두었습니다. 링크를 클릭하시면 확인하실 수 있습니다.





#### Volumes

persistance volume을 python SDK를 사용하여 생성합니다.

**사용 이유**:

- component에서 custom package를 import하기 위해 사용합니다.

  poersistent volume에 `git clone`한 후 python system path에 해당 volume의 path를 append합니다.

  ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/sys%20append.png?raw=true)

- component에 의해 생성된 data를 보호합니다.(model file, evaluation result 등)



#### Pipelines, Experiments(KFP), Runs

- `Pipeline`, `Experiment`, `Run` 의 create, pipeline version control 그리고 delete동작은 모두 python SDK를 사용하여 구현했습니다.

  pipeline의 경우 upload하는 version의 존재 유무에 따라 Run의 동작이 달라질 수 있도록 했습니다. 

  ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/pipeline%20upload.png?raw=true)

- pipeline의 component는 아래와 같이 간단하게 구현했습니다.

  ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/pipeline%20graph.png?raw=true)

  - **Record** : labelme.exe를 통해 만들어진 annotations dataset을 training을 위한 `train_dataset.json`으로 통합합니다.

    통합 후 images는 google cloud에, 각 image및 dataset의 관련 정보는 DB에 commit합니다.

    ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/Record.png?raw=true)

    1.  [pipeline_dataset](https://github.com/HibernationNo1/pipeline_dataset.git) repository를 `clone`하고 annotation dataset의 특정 version이 명시된 tag로 `checkout`합니다. 
  
    2. [pipeline_dataset](https://github.com/HibernationNo1/pipeline_dataset.git) repository의 dvc file을 통해 google storage로부터 annotation dataset을 download합니다.
  
       DB의 annotation dataset의 정보를 토대로 image list와 json list를 생성합니다.
  
    3. annotation dataset을 model에 input으로 사용할 수 있도록 coco dataset과 같은 구조로 통합하여 training dataset을 구성합니다.
  
    4. training dataset의 정보를 DB에 insert하고 `dvc add`, `add push`를 통해 google storage에 upload합니다.
  
  - **Train** :  model training을 진행합니다.
  
    training과정에서 저장한 model과 logs, tensorboard는 volume에 저장하고 google storage로 push합니다.
  
    ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/Training.png?raw=true)

    1. [pipeline_dataset](https://github.com/HibernationNo1/pipeline_dataset.git) repository를 `clone`하고 training dataset의 특정 version이 명시된 tag로 `checkout`합니다. 
  
    2. [pipeline_dataset](https://github.com/HibernationNo1/pipeline_dataset.git) repository의 dvc file을 통해 google storage로부터 training dataset을 download합니다.
  
       DB의 training dataset의 정보를 토대로 Dataloader를 build합니다.
  
    3. Model training과 validation을 진행합니다.
  
    4. **Google cloud SDK**를 통해 trained model과 log파일들을  google storage에 upload합니다.
  
  - **Evaluate** : trained model을 통해 evaluation을 진행합니다.
  
    ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/Evaluation.png?raw=true)
  
    >  1번의 과정은 Train component와 동일합니다.
  
    1. [pipeline_dataset](https://github.com/HibernationNo1/pipeline_dataset.git) repository를 `clone`하고 training dataset의 특정 version이 명시된 tag로 `checkout`합니다. 
    2. [pipeline_dataset](https://github.com/HibernationNo1/pipeline_dataset.git) repository의 dvc file을 통해 google storage로부터 validation dataset을 download합니다.
  
       DB의 training dataset의 정보를 토대로 Dataloader를 build합니다.
  
       **Google cloud SDK**를 통해 trained model을 download합니다.
  
    3. `mAp`, `dv_mAP`, `EIR` 등 사용자가 결정한 성능 평가 지표에 대한 결과값을 계산하고, 특정 지표값에 대해 성능이 좋은 model을 선별하여 save합니다. 
    4. **Google cloud SDK**를 통해 evaluation과정에서 선별된 model을 google storage에 upload합니다.
  
  - **Test**: trained model을 통해 inferene를 진행합니다. 
  
    ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/Test.png?raw=true)
  
    1. **Google cloud SDK**를 통해 Google storage의 **result bucket**으로부터 evaluation과정에서 선별된 model을 download합니다.
    
    2. download한 model로 test dataset에 대해 inference를 진행하고, object detection의 결과를 시각화합니다.
    
       ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/inference%20result.png?raw=true)



#### Tensorboard

training과정에서 특정 value를 실시간으로 확인할 수 있도록 persistance volume에 tensorboard event file을 저장하도록 했습니다.

**Add tensorboard**

![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/tensorboard.png?raw=true)



**Connection**

port forward

```
$ kubectl get pods -n pipeline | grep pipeline-project
NAME                                                              READY   STATUS      RESTARTS      AGE
pipeline-project-6cd9c947c6-kp8q8                                 2/2     Running     0             21m
```

```
$ kubectl port-forward -n pipeline pipeline-project-6cd9c947c6-lxr2j 6006:6006
Forwarding from 127.0.0.1:6006 -> 6006
Forwarding from [::1]:6006 -> 6006
Handling connection for 6006
```



**Result**

![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/tensorboard_viewer.png?raw=true)



#### Secrets

component의 code상에서 dataset을 관리할 때 DB의 passward나 google secrets key와 같이 중요한 정보를 사용하는 경우가 있습니다.

이를 위해 kubernetest의 resource중 하나인 secret를 사용하여 보안 문제를 해결하고자 했습니다.

secret은 file의 contents를(2개 이상의 값) secret value로 전달하는 **from-env-file** 방식을 사용했습니다.

file은 아래와 같은 형식으로 작성했습니다. (예시)

```
$ vi project_secrets.txt
```

```
type=service_account
project_id=adroit-xxxxx-xxxx
private_key_id=0b55dxxxxxxx30211daf249b0xxxxxxxx
private_key=
client_email=xxxxxxx8@adroit-xxxxxxx-xxxxxx.iam.xxxxxxxx.com
client_id=xxxxxxxxxxxxxxxxxxxx4
auth_uri=https://accounts.google.com/o/xxxx/xxxx
token_uri=https://xxxx.googleapis.com/xxxxx
auth_provider_x509_cert_url=https://www.googleapis.com/xxxxx/v1/xxxxx
client_x509_cert_url=https://www.googleapis.com/robot/v1/metadata/xxxx/xxxxxxxxx-xxxxxxxxxxr-xxxxx.iam.gserviceaccount.com
```

```
$ kubectl -n project-pipeline create secret generic project_secrets --from-env-file project_secrets.txt
```

이후 pipeline을 정의하는 python code에서 kubernetes python SDK를 사용해서 secret값을 특정 component의 환경변수로 할당했습니다.

해당 환경변수는 `os.environ['name']`을 통해 값을 사용했습니다.

해당 code: [pipeline.py](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/pipeline.py) line 24  `project_pipeline()`

- google의 `private_key`같은 경우 아래와 같이 `\n`이 포함된 값입니다.

  ```
  -----BEGIN PRIVATE KEY-----\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n-----END PRIVATE KEY-----\n
  ```

  해당 값을 code상에서 `os.environ`로 출력하면 `\n` 이 줄바꿈으로 변환 없이 그대로 출력되게 되는데, 이런 경우 잘못된 key값으로 인식되게 됩니다.

  이를 해결하기 위해 아래 code를 통해 key값을 변환시켜 주었습니다.

  ```
  flag = False
  private_key = ""
  for arp in list(os.environ['private_key']):
      if arp == '\\':
          private_key+=arp
          flag = True
          continue
  
      if arp == 'n' and flag == True:
          private_key = private_key[:-1]
          private_key +='\n'
          flag = False
          continue
  
       private_key+=arp
  print(private_key)
  ```

  아래는 변환된 key값입니다.

  ```
  -----BEGIN PRIVATE KEY-----
  xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  -----END PRIVATE KEY-----
  ```

  



---



## TODO List

- Model serving (with katib)
- Using other model (swin-transformer, Mask2Former)



## Installation Process

Ubuntu 20.04 환경에서 프로젝트를 진행했으며, 전체적인 tools 및 package의 설치과정은 **[여기](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/setting.md)**에서 확인하실 수 있습니다.



- **used**
  - **technology** : `docker`, `kubernetes`, `kubeflow`, `dvc`, `git`, `google cloud storage`, `mysql`, `PypI`
  - **package**: `torch`, `numpy`, `opencv`, `pymysql`, `pandas`, `kubernetes-sdk`, `kubeflow-sdk`, `gitpython`, `json`, `mmcv`



## 참고 문헌 및 강의

- **mmdet**: https://github.com/open-mmlab/mmdetection
- **mmvc**: https://github.com/open-mmlab/mmcv
- **Swin-Transformer**: https://arxiv.org/pdf/2103.14030.pdf
- **도서 - 컨테이너 인프라 환경 구축을 위한 쿠버네티스/도커**: http://www.yes24.com/Product/Goods/102099414
- **도서 - 쿠브플로우**: http://www.yes24.com/Product/Goods/89494414

- **Kubeflow 인프런 강의**: https://www.inflearn.com/course/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4-%EC%8B%A4%EB%AC%B4
- **Mlops 패스트 캠퍼스 강의**: https://fastcampus.co.kr/data_online_mlops
- **Kubeflow YouTube 강의**: https://www.youtube.com/@myoh0623

그외 각 kubernetes, kubeflow 공식 페이지 및 github...