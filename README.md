

# Project: Automatic License Plate Recognition

## Table of Contents



## Overview

해당 프로젝트는 kubernetes상에서 kubeflow를 활용하여 모델을 학습하는 pipeline을 구성하는 것을 구현한 프로젝트입니다.

전체적인 process는 아래와 같습니다.



- collect dataset
- labeling
- Integrate `annotation dataset` into` coco data` set form
- model training
- model serving(구현중)



model을 통해 기대하는 inference결과는 License Plate의 각 text를 인식하고  **등록지역(두 자리 번호)**, **차종기호(A~E 중 하나)**, **일련번호(네 자리 번호)**를 각각 추출해내는 것입니다.



## Managing dataset 

- 자동차 번호판을 학습시키는 것을 trining의 목적으로 했습니다.

- dataset은 `labelme.exe`를 통해 직접 만들었으며, DVC를 이용해 version관리를 하였고 DB에서 data를 관리했습니다.
- 실제 자동차 번호판을 모아놓은 데이터 셋을 만들고자 했지만, 자동차 소유주들에게 자문을 구한 결과 '모르는 사람이 자신의 자동차의 번호판을 촬영해간다면 매우 불쾌할 수 있다.'는 의견을 듣고 배경 이미지에 임의로 번호판을 그리는 것으로 방향을 잡게 되었습니다. 



### Process Of Acquiring Datasets

1. 핸드폰으로 사진을 찍습니다.

   >  사진 촬영 이미지

2. python code를 통해 사진 위 무작위 좌표에 간단하 번호판 그림을 그려 넣습니다.

   > 코드를 통한 사진 변화 전, 후 이미지

   [해당 code]()

3. `Labelme.exe`를 통해 번호판이 그려진 이미지에 라벨링 작업을 진행합니다.

   >  라벨링 작업 전 후 이미지

4. 라벨링 작업이 완료된 dataset을 DVC를 활용하여 version관리하고 google cloud에 저장하며, DB에 각 image의 path 및 infomation을 기록하여 관리합니다. 

   [detail]()





## Pipeline Configuration With Kubeflow

### 1. mmdetection

model의 학습 code는 [open-mmlab](https://github.com/open-mmlab)/**[mmdetection](https://github.com/open-mmlab/mmdetection)**을 차용했으며, 몇 가지 수정을 통해 원하는 동작을 추가했습니다.

[detail]()

1. **Change config map**

   - kubeflow python SDK를 활용하여 pipeline을 구성할 때 용이하게 parameters를 전달하기 위해 변경했습니다.
   - kubeflow의 기능 중 하나인 **Experiments**를 사용하기 위해서는 pipeline이 아닌 local에서도(docker container상에서도) 간편하게 training및 test를 수행할 수 있어야 하기 때문에, 이를 위해 변경했습니다.
   - `Swin-Transformer`, `Mask rcnn`외 다른 backbone과 mask head model을 학습시킬 때 config file의 path를 명시하기 보단 model name만 명시할 수 있도록 하기 위해 변경했습니다.

   기존 - 변경 후 폴더 구조 이미지

2. **Add `config dict` and `instance name list` to the trained model file**

   - 기존 mmdet의 code는 학습 된 model을 이어서 학습할 때, 해당 model의 학습을 처음 시작 할 시점의 config와 이어서 학습을 진행 할 시점의 config가 다르면 학습 과정 중 code error가 발생하게 됩니다.

     >  ex) hook의 값 변경, 특정 hyper parameter변경

     이를 방지하기 위해 학습 당시 사용했던 config를 model file에 추가로 저장했습니다.

   - 기존 mmdet의 code는 training에 사용된 instance의 정보를 하드 코딩으로 저장하기 때문에 새 학습을 진행 할 때마다 training dataset에 new instance가 있거나 배열의 순서가 바뀌면 inference과정에서 object name에 대한 error 또는 wrong value가 출력될 수 있습니다.

     이런 번거로움을 없애기 위해 학습 당시 사용했던 training dataset의 instance list를 model file에 추가로 저장했습니다.

   이미지 만들어서 추가

3. **Add custom validation code**

   validation 및 evaluation을 진행할 때 object detection의 성능 평가 지표인 *mean average precision*(이하 mAP)을 계산하는 방법에 segmentation을 위한 한 가지 고차원적인 평가 기능을 추가했습니다.

   detail

   detail 안에 이미지 만들어서 divide_iou계산 설명

4. **Modified original hooks, and added custom hooks.**

   - log 출력 및 저장 시 제가 원하는 infomation까지 포함할 수 있도록 변경했습니다.
   - tensorboard를 저장하는 hook을 추가했습니다. 
   - validation을 수행하는 hook을 추가했습니다.

5. **Add text parsing algorithm**

   inference결과를 통해 등록지역(두 자리 번호)**, **차종기호(A~E 중 하나)**, **일련번호(네 자리 번호)를 추출하는 algorithm을 추가했습니다.

   detail (어떻게 각 번호를 추출하는지 그림으로 설명)



### 2. kubeflow

1. **Experiments(Katib-AutoML)**:

   Hyper-prameter tuning을 진행하기 위해 Kubeflow의 구성 요소인 Katib system을 사용했습니다.

   결과 이미지 추가

   각각 어떤 hyper parameter를 튜닝했고, 어떤 방식으로 output을 받았는지 설명

   자세한 설명은 [이곳]()에서 확인하실 수 있습니다. (yaml file에 대해서 설명)

2. **Volumes**:

   persistance volume을 python SDK를 사용하여 관리합니다.

   custom package code를 volume에 clone한 후 component내에서 package import시 해당 volume의 path를 포함하여 탐색할 수 있도록 하여 자유롭게 import 및 git version관리를 할 수 있도록 했습니다.   

   volume 대쉬보드 및 sdk로 생성하는 code 화면 캡쳐

3. **Pipelines, Experiments(KFP), Runs**

   - create, version control 그리고 delete 와 같은 `Pipeline`, `Experiment`, `Run` 관리는 모두 python SDK를 사용하여 구현하였습니다.

     전체적인 dir구조 그림으로 첨부(hibernation package도 표현) (filezila화면 또는 vscode화면 캡쳐)

   - pipeline의 component는 아래와 같이 간단하게 구현했습니다.

     pipeline 이미지 첨부

     - **recode** : labelme.exe를 통해 만들어진 annotations dataset을 training을 위한 `train_dataset.json`으로 통합합니다.

       통합 후 images는 google cloud에, 각 image및 dataset의 관련 정보는 DB에 commit합니다.

       자세한 내용은 여기(링크 남기기)

     - **train** :  training을 진행합니다.

       학습을 통해 저장한 model과 tensorboard는 volume과 google storage에 저장합니다.

       자세한 내용은 여기(링크 남기기)

     - **evaluation** : trained model을 통해 evaluation을 진행합니다.

       evaluation과정에서 계산된 각 class별 average plot과 mAP값은 volume에 저장하고, 추가로 mAP값은 kubeflow dashboard에 시각화합니다.

       mAP결과 시각화 된 것 이미지 

     - **test**: trained model을 통해 inferene를 진행합니다. 

       inference결과 image와 text persing결과는 kubeflow dashboard에 시각화하고 volume에 저장합니다.

       inference결과 시각화 된 것 이미지, text persing 시각화 된 것 이미지

4. **Tensorboard**

   training과정에서 특정 value를 실시간으로 확인할 수 있도록 persistance volume에 tensorboard event file을 저장하도록 했습니다.

   tensorboard화면 캡쳐 이미지





## TODO List

1. katib완성하기

2. 구글 스토리지 활성화 하기

3. DB 다시 만들고 학습 되는지 확인하기

4. validation component로 올리기

   - validation에서 infernce진행하여 추론 및 board text예측 결과 나오게 하고 dataset이랑 비교하는 code추가하기
   - 여러 model을 받은 단일 model을 하든 validation결과는 dict형태로 dashboard에 띄우기

5. test도 component올리기

   model 여러개를 list로 받아서 각 model에 따라 test이미지의 결과를 확인할 수 있도록 만들기(또는 1개)

   

   input으로 image 한 개 받고, 그 결과는 dashboard에 나오도록

   dashboard에 이미지 최대 몇개까지 올라가나 실험

   

   google storage에서 image여러개 다운받고 각 model별로 test결과 따로 저장하도록

6. recode - training - test 한 번에 하는거랑, 따로따로 하는걸 config하나로 가능하도록 하기

   





이어서 학습하기

모델 서빙

다른 모델 사용(swin-transformer, Mask2Former)



## Installation Process

우분투 20.04 환경에서 프로젝트를 진행했으며, 전체적인 tool의 설치는 `여기`에서 확인하실 수 있습니다.

여기: Prerequisites 포함하기





## 참고 문헌 및 강의

- **mmdet**: https://github.com/open-mmlab/mmdetection
- **Swin-Transformer**: https://arxiv.org/pdf/2103.14030.pdf
- **도서 - 컨테이너 인프라 환경 구축을 위한 쿠버네티스/도커**: http://www.yes24.com/Product/Goods/102099414
- **도서 - 쿠브플로우**: http://www.yes24.com/Product/Goods/89494414

- **Kubeflow 인프런 강의**: https://www.inflearn.com/course/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4-%EC%8B%A4%EB%AC%B4
- **Mlops 패스트 캠퍼스 강의**: https://fastcampus.co.kr/data_online_mlops
- **Kubeflow YouTube 강의**: https://www.youtube.com/@myoh0623

그외 각 kubernetes, kubeflow 공식 페이지 및 github...