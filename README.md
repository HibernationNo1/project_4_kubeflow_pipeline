

# Project: Automatic License Plate Recognition

## Table of Contents



## Overview

해당 프로젝트는 kubernetes상에서 kubeflow를 활용하여 모델을 학습하는 pipeline을 구현한 프로젝트입니다.

전체적인 흐름도는 아래와 같습니다.



![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/overview_.png?raw=true)

1. Collect dataset and labeling

   dataset을 수집하고 라벨링을 진행합니다.

2. Integrate `annotation dataset` into` coco data` set form

   model의 학습 데이터로 사용하기 위해 coco dataset과 같은 구조로 데이터셋을 합칩니다.

3. Model training and evaluation

   model을 학습하고 evaluation과 inference test를 진행합니다.

   model을 통해 기대하는 inference결과는 License Plate의 각 text를 인식하고  **등록지역(두 자리 번호)**, **차종기호(A~E 중 하나)**, **일련번호(네 자리 번호)**를 각각 추출해내는 것입니다.

4. Model serving

   model을 배포합니다.

   > 구현중에 있습니다.







## Managing dataset 

- 자동차 번호판을 학습시키는 것을 trining의 목적으로 했습니다.

- dataset은 `labelme.exe`를 통해 직접 만들었으며, DVC를 이용해 version관리를 하였고 DB에서 data를 관리했습니다.
- 실제 자동차 번호판을 모아놓은 데이터 셋을 만들고자 했지만, 자동차 소유주들에게 자문을 구한 결과 '모르는 사람이 자신의 자동차의 번호판을 촬영해간다면 매우 불쾌할 수 있다.'는 의견을 듣고, python을 사용해 배경 이미지에 임의로 번호판을 그려 dataset을 구성하는 것으로 방향을 잡게 되었습니다. 



### Process Of Acquiring Datasets

1. 핸드폰으로 사진을 찍습니다.

2. python code를 통해 사진 위 무작위 좌표에 간단하 번호판 그림을 그려 넣습니다.

   ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/%EB%A7%8C%EB%93%A0%20%EC%9D%B4%EB%AF%B8%EC%A7%80%201.png?raw=true)

   [drawing code](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/tmp_code/draw_board.py)

3. `Labelme.exe`를 통해 번호판이 그려진 이미지에 라벨링 작업을 진행합니다.

   ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/labeling.png?raw=true)

4. 라벨링 작업이 완료된 dataset을 DVC를 활용하여 version관리하고 google cloud에 저장하며, DB에 각 image의 path 및 infomation을 기록하여 관리합니다. 

   ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/dataset%20init.png?raw=true)

   dataset을 관리하는 code는 [pipeline_dataset](https://github.com/HibernationNo1/pipeline_dataset.git) repository로 관리하며, tag를 통해 annotation dataset과 training dataset의 commit을 구분합니다.





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

persistance volume을 python SDK를 사용하여 관리합니다.

custom package code를 volume에 clone한 후 component내에서 package import시 해당 volume의 path를 포함하여 탐색할 수 있도록 하여 자유롭게 import 및 git version관리를 할 수 있도록 했습니다.   

volume 대쉬보드 및 sdk로 생성하는 code 화면 캡쳐



#### Pipelines, Experiments(KFP), Runs

- create, version control 그리고 delete 와 같은 `Pipeline`, `Experiment`, `Run` 관리는 모두 python SDK를 사용하여 구현하였습니다.

  전체적인 dir구조 그림으로 첨부(hibernation package도 표현) (filezila화면 또는 vscode화면 캡쳐)

- pipeline의 component는 아래와 같이 간단하게 구현했습니다.

  pipeline 이미지 첨부

  - **Record** : labelme.exe를 통해 만들어진 annotations dataset을 training을 위한 `train_dataset.json`으로 통합합니다.

    통합 후 images는 google cloud에, 각 image및 dataset의 관련 정보는 DB에 commit합니다.

    자세한 내용은 여기(링크 남기기)

  - **Train** :  training을 진행합니다.

    학습을 통해 저장한 model과 tensorboard는 volume과 google storage에 저장합니다.

    자세한 내용은 여기(링크 남기기)

  - **Evaluate** : trained model을 통해 evaluation을 진행합니다.

    evaluation과정에서 계산된 각 class별 average plot과 mAP값은 volume에 저장하고, 추가로 mAP값은 kubeflow dashboard에 시각화합니다.

    mAP결과 시각화 된 것 이미지 

  - **Test**: trained model을 통해 inferene를 진행합니다. 

    inference결과 image와 text persing결과는 kubeflow dashboard에 시각화하고 volume에 저장합니다.

    inference결과 시각화 된 것 이미지, text persing 시각화 된 것 이미지



#### Tensorboard

training과정에서 특정 value를 실시간으로 확인할 수 있도록 persistance volume에 tensorboard event file을 저장하도록 했습니다.

tensorboard화면 캡쳐 이미지





## TODO List

- Model serving (with katib)
- Using other model (swin-transformer, Mask2Former)



## Installation Process

Ubuntu 20.04 환경에서 프로젝트를 진행했으며, 전체적인 tool의 설치과정은 **[여기](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/setting.md)**에서 확인하실 수 있습니다.



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