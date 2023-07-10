# Project: Automatic License Plate Recognition

- **프로젝트 소개**: kubeflow를 활용한 임의 번호판 검출 모델 학습 Pipeline구축 및 InferenceService배포 

- **개발 인원**: 개인 프로젝트

- **개발 기간**: 2022.11.01 ~ (진행중)

- **진행 상황**
  - kubeflow - Experiments(Katib-AutoML) 을 통한 최적의 hyper-parameter 검색 (구현 완료)
  - Pipelines설계 (dataset통합, 학습, 검증 및 평가, 추론 테스트)  (구현 완료)
  - Endpoints (Kserve)를 활용한 model serving (테스트 완료)

- **사용 기술**
  - **technology** : `docker`, `kubernetes`, `kubeflow`, `katib`, `kserve`, `torchserve` , `DVC`, `GIT`, `Google Cloud Storage`, `MySQL`, `PypI`
  - **python - package, library**: `pytorch`, `numpy`, `opencv`, `pymysql`, `pandas`, `kubernetes-sdk`, `kubeflow-sdk`, `gitpython`, `json`, `mmcv`, `mmdet`

- **흐름도**

  ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/overview_.png?raw=true)

  해당 프로젝트는 크게 아래의 순서로 구현되었습니다.

  1. **Collect dataset and labeling**

     dataset을 수집하고 라벨링을 진행합니다.

  2. **Integrate `annotation dataset` into` coco data` set form**

     model의 학습 데이터로 사용하기 위해 coco dataset과 같은 구조로 데이터셋을 통합합니다.

  3. **Model training and evaluation**

     model을 학습하고 evaluation과 test를 진행합니다.
  
     1. `Experiments(Katib-AutoML)`을 활용해 hyper-parameter tuning을 진행합니다.
  
     2. `Pipelines`를 활용해 데이터 통합, 모델 학습 및 검증, 테스트를 진행합니다.
  
        - 번호판과 각 text를 감지하는 것이 model을 통해 기대하는 inference결과입니다.
  
        - Inference의 결과로 얻어진 데이터를 토대로 License plate의 정보를 추출하는 것이 Post Processing의 목적입니다.
  
          
  
          ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/License%20plate%20desc.png?raw=true)
  
          ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/License%20plate%20desc_1.png?raw=true)
  
  4. **Model serving**
  
     학습된 model을 serving합니다.

dataset의 구축 - model 학습 및 검증 - model 테스트 - 최적의 model 선별 - model serving의 동작을 구현했습니다. 



## Table of Contents

- [Managing dataset](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/README.md#managing-dataset)
- [Customizing mmdetection, mmcv](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/README.md#1-customizing-mmdetection-mmcv)
- [Pipeline Configuration With Kubeflow](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/README.md#pipeline-configuration-with-kubeflow)
  - [Experiments(Katib-AutoML)](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/README.md#experimentskatib-automl)
  - [Volumes](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/README.md#volumes)
  - [Pipelines, Experiments(KFP), Runs](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/README.md#pipelines-experimentskfp-runs)
  - [Tensorboard](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/README.md#tensorboard)
  - [Secrets](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/README.md#secrets)
- [Model Serving](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/README.md#model-serving)
- [About project](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/README.md#about-project)
  - [Project management](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/README.md#project-management)
  - [TODO List](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/README.md#todo-list)
  - [Installation Process](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/README.md#installation-process)
  - [참고 문헌 및 강의](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/README.md#%EC%B0%B8%EA%B3%A0-%EB%AC%B8%ED%97%8C-%EB%B0%8F-%EA%B0%95%EC%9D%98)





## Managing Dataset 

- **Dataset 소개**

  - dataset의 category는 **자동차 번호판**으로 구성하였습니다.

    실제 자동차 번호판을 모아놓은 데이터 셋을 만들고자 했지만 '모르는 사람이 자신의 자동차의 번호판을 촬영해간다면 매우 불쾌할 수 있다.'는 의견을 듣고, python을 사용해 배경 이미지에 임의로 번호판을 그려 dataset을 구성하는 것으로 방향을 잡게 되었습니다. 

  - dataset은 `labelme.exe`를 통해 직접 그려가며 만들었으며, [DVC](https://dvc.org/)와 GIT을 이용해 version을 관리했습니다.

  - DateBase에서 data의 정보를 version별로 구분하여 관리했습니다.

  - dataset images와 version관리 파일은 [pipeline_dataset](https://github.com/HibernationNo1/pipeline_dataset) repository로 관리합니다.

- **Dataset 라이프 사이클**

  dataset은 다음과 같은 두 개의 과정이 있습니다.

  - annotation dataset

    `labeme.exe`를 통해 각 image에 대하여 라벨링이 완료된 dataset

  - training datase

    training을 진행할 때 직접적으로 input으로 사용되기 위해 coco형태로 통합된 dataset

  각각 dataset의 새 version이 구성되는 경우, 관련 정보는 DataBase에 저장합니다.

- **Dataset 구축**

  1. 핸드폰 사진 촬영

  2. python code를 통해 사진 위 무작위 좌표에 간단한 임의 번호판 생성

     ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/%EB%A7%8C%EB%93%A0%20%EC%9D%B4%EB%AF%B8%EC%A7%80%201.png?raw=true)

     [drawing code](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/tmp_code/draw_board.py) : 이미지 위에 임의의 번호판을 그리는 code입니다.

     번호판의 color, type, size값을 다양하게 구성 후 랜덤하게 할당되도록 했습니다. 

- **Dataset 관리**

  - annotation dataset

    1. 라벨링 작업이 완료된 dataset은 dvc명령어를 통해 google cloud에 push합니다.

    2. [pipeline_dataset](https://github.com/HibernationNo1/pipeline_dataset) repository의 [main.py](https://github.com/HibernationNo1/pipeline_dataset/blob/master/main.py) code를 실행합니다.

       pymysql를 활용하여 서버에 구축한 DB에 각 이미지의 path, dataset version등 여러 정보를 기록하여 관리합니다. 

    >  위 동작에 관한 설명은 [pipeline_dataset](https://github.com/HibernationNo1/pipeline_dataset) repository의  [README.md](https://github.com/HibernationNo1/pipeline_dataset/blob/master/README.md)를 통해 확인하실 수 있습니다.

  - training datas

    [project_4_kubeflow_pipeline](https://github.com/HibernationNo1/project_4_kubeflow_pipeline) repository의 [component/record/record_op.py](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/component/record/record_op.py) code를 실행합니다.

    - google storage로부터 annotation dataset을 pull합니다.
    - 학습을 위해 coco형태로 통합합니다.
    - pymysql를 활용하여 DB에 통합 된 dataset의 path, purpose,version등 여러 정보를 기록하여 관리합니다. 

  각각의 dataset은 tag를 사용하여 annotation dataset과 training dataset의 commit을 구분합니다.





## Customizing mmdetection, mmcv

model의 학습 code는 [open-mmlab](https://github.com/open-mmlab)/[mmdetection](https://github.com/open-mmlab/mmdetection) 과 [open-mmlab](https://github.com/open-mmlab)/[mmcv](https://github.com/open-mmlab/mmcv)를 차용했으며, 몇 가지 수정을 통해 원하는 동작을 추가했습니다.

![](https://github.com/open-mmlab/mmdetection/raw/main/resources/mmdet-logo.png)

![](https://raw.githubusercontent.com/open-mmlab/mmcv/main/docs/en/mmcv-logo.png)

1. Model은 **Swin-Transformer**(Backbone)와 **Mask-RCNN**(Head)으로 구성했으며, mmdetection의 code를 사용했습니다.

2. Model load시 `config`와 같은 추가적인 파일을 load할 것 없이 model파일 내부에서 관련 내용을 가져와 training 및 inference를 진행할 수 있도록 했습니다.

   [detail](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/customizing%20mmdetection%2C%20mmcv.md#add-configuration-and-instance-name-list-to-the-model-file)

3. Inference의 결과로 얻어진 데이터를 토대로 License plate의 정보를 추출하는 post-processing을 구현했습니다.

   [detail](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/customizing%20mmdetection%2C%20mmcv.md#post-processing)

4. Object Detction의 성능 평가 지표인 **mAP**를 한 단계 더욱 고차원적인 방법으로 계산하는 성능 평가 지표를 만들고 구현했습니다.

   [detail](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/customizing%20mmdetection%2C%20mmcv.md#1-dv-map)

5. 제가 만든 dataset를 위한 model 평가 지표를 구현했습니다.

   [detail](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/customizing%20mmdetection%2C%20mmcv.md#2-exact-inference-rateeir)

6. Component, Kserve와 같은 resource를 위한 docke image를 구성하는 과정에서 mmcv, mmdet의 설치가 되지 않는 문제가 있어 의존성을 제거했습니다.

7. 해당 code는 [sub_module](https://github.com/HibernationNo1/sub_module)에 의해 관리하고 있습니다.

★☆★해당 파트에 관한 자세한 내용은 **[이곳](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/customizing%20mmdetection%2C%20mmcv.md)**에 정리해두었습니다. ★☆★



## Pipeline Configuration With Kubeflow

pipeline을 구성하는 과정에서 활용된 kubeflow의 resource는 아래와 같습니다.

- Experiments(Katib-AutoML)

  model의 가장 이상적인 학습을 위해 hyper parameter를 조정합니다.

- volume

  pipeline의 각 component에서 접근할 수 있는 공용 공간입니다.

- Pipelines, Experiments(KFP), Runs

  - `Runs`: 구축된 Pipelines이 정상적으로 동작하는지 확인합니다.

  - `Pipelines`: 여러 component를 활용하여 pipeline을 구축합니다.

  - `Experiments(KFP)`: pipeline을 목적에 따른 category로 관리합니다.

- Tensorboard

  학습 과정의 정보를 확인하기 위해 여러 data를 수집합니다. 

- Secrets

  보안상 노출해서는 안되는 값을 사용해야 되는 경우를 위해 사용합니다.



#### Experiments(Katib-AutoML)

resource중 Experiments에 관한 설명입니다.

Hyper-prameter tuning을 진행하기 위해 Kubeflow의 구성 요소인 Katib system을 사용했습니다.

![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/Experiment.png?raw=true)

- `learning rate`와 `backbone model`의 몇 가지 hyper parameter에 대해서 조정을 진행했습니다.

- 수집하고자 하는 metric은 `mAP`, `dv_mAP`, `EIR` 으로 하였으며, **goal**의 기준으로 할 metric은 `mAP`로 결정했습니다.

  [detail](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/Experiments(Katib-AutoML).md#Experiment)

- dockerfile은 ubuntu:20.04를 기반으로 한 `base ubuntu:20.04` image를 사용했습니다.

  [detail](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/Experiments(Katib-AutoML).md#base-ubuntu2004)

★☆★Experiments(Katib-AutoML) 구현에 관한 자세한 내용은 [이곳](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/Experiments(Katib-AutoML).md)에 정리해두었습니다. ★☆★





#### Volumes

resource중 Volumes에 관한 설명입니다.

persistance volume을 생성하여 사용합니다.

**사용 이유**:

- Component에서 custom package를 import하기 위해 사용합니다.

  > custom package란, 위에서 말씀드린`sub_module`과 같은 보조 library 또는 module을 의미합니다.
  >
  > 보안산 PypI에 업로드 할 수 없는 library인 경우 직접 import 할 수 있도록 하기 위함입니다.

  poersistent volume에 git으로 관리하는 code를  `git clone`한 후, python system path에 해당 volume의 path를 append하여 사용합니다.

  ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/sys%20append.png?raw=true)

- component에 의해 생성된 data를 저장합니다.(model file, evaluation result 등)



#### Pipelines, Experiments(KFP), Runs

resource중 Pipelines 구성에 관한 설명입니다.

`Pipeline`, `Experiment`, `Run` 의 create, pipeline version control 그리고 delete동작은 모두 python SDK를 사용하여 구현했습니다.

pipeline의 경우 upload하는 version의 존재 유무에 따라 Run의 동작이 달라질 수 있도록 했습니다. 

>  해당 code는 [pipeline_utils.py](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/pipeline_utils.py)의 `line:17`에서 에서 확인하실 수 있습니다.

![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/pipeline%20upload.png?raw=true)

pipeline의 component는 아래와 같이 간단하게 구현했습니다.

![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/pipeline%20graph.png?raw=true)

> 각각의 component는 Persistence volume을 통해 보조 code를 import합니다.
>
> 각각의 component는 필요한 데이터를 google storage에서 다운로드하고, 동작의 결과를 google storage에 업로드합니다.

- Recode : labelme.exe를 통해 만들어진 annotations dataset을 training을 위한 `train_dataset.json`으로 통합합니다.

  [detail](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/Pipeline.md#record)

- `Train` : model training을 진행합니다.

  [detail](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/Pipeline.md#train)

- `Evaluate` : trained model을 통해 evaluation을 진행합니다.

  [detail](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/Pipeline.md#evaluate)

- `Test`: trained model을 통해 inferene를 진행합니다. 

  [detail](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/Pipeline.md#test)

★☆★Pipelines구성에 관한 자세한 내용은 [이곳](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/Pipeline.md)에 정리해두었습니다. ★☆★





#### Tensorboard

resource중 Tensorboard에 관한 설명입니다.

training과정에서 특정 value를 실시간으로 확인할 수 있도록 persistance volume에 tensorboard event file을 저장하도록 했습니다.

**Add tensorboard**

![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/tensorboard.png?raw=true)



**Connection**

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

각 loss외에도 성능 평가 지표와 GPU memory사용량 등을 기록할 수 있도록 했습니다.

tensorboard에 관한 code는 [custom.py](https://github.com/HibernationNo1/sub_module/blob/ca9a56efb315352f57096edc73a9af81a79c34e8/mmdet/hooks/custom.py) `Line: 198`에서 확인하실 수 있습니다.

![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/tensorboard_viewer.png?raw=true)



#### Secrets

resource중 Secrets에 관한 설명입니다.

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

  



## Model Serving

Kserve의 InferenceService를 기반으로 한 torchserve를 배포하여 model serving을 진행했습니다.

**Service의 목표는 이미지 데이터를 입력받아 License plate를 감지하고 해당 정보를 출력하는 것입니다.**

**kserve.InferenceService**에 관한 code및 모든 file은 [project4_kserve](https://github.com/HibernationNo1/project4_kserve) repository로 관리하고 있습니다.

1. InferenceService를 생성했습니다.

   ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/kserve_6.png?raw=true)

2. serving된 model로 직접 inference를 진행했습니다.

   test를 위해 API로 보낼 이미지는 아래 3개를 사용했습니다.

   > 3번 이미지는 다른 형태의 plate가 한 장의 사진에 위치할 수 있도록 임의로 만든 이미지입니다.

   ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/kserve_1.png?raw=true)

   ```
   $ python request/request.py {image_name}.jpg --kserve
   ```

   

   아래는 response로 받은 결과입니다.

   ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/kserve_2.png?raw=true)

   response는 **plate의 모양**, **plate의 이미지 상 좌표 및 크기**, **등록지역(두 자리 번호)**, **차종기호(A~E 중 하나)**, **일련번호(네 자리 번호)**에 대한 정보를 담고 있습니다.

   해당 데이터를 토대로 사용자가 보기 쉽게(`각 번호를 위치에 맞게, plate의 모양 타입에 맞게`) 변환하여 보여줍니다.

★☆★Model serving과정에 관한 자세한 내용은 **[이곳](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/Model_Serving.md)**에 정리해두었습니다. ★☆★





## About project



### Trouble Shooting

- **version 문제 해결**

  - cluster를 구성하는 과정에서 여러 프로그램의 version의 호환을 맞춰야 하는 경우가 있었습니다.

    ```
    cri-dockerd >= 0.3.0		# 0.2.6이하의 version에서는 kubeadm init을 실행 시 error발생
    kubernetes >= 1.25.0 		
    kserve >= 0.8				# kubernetes version <= 1.24.0이하인 경우 kubeflow 1.6.0과 호환 문제 발생
    ```

  - torch와 cuda toolkit의 version을 docker image로 build가능한 것(`torch==1.12.1+cu113`)으로 맞춰야 했습니다.

    - `torch<=1.12.0` : docker container상에서 mmcv, mmdet의 의존성 제거 과정에서 포함한 cpython프로그래밍 함수를 import하지 못했습니다. 

    - `torch==1.12.0` : `'Adam' object has no attribute '_warned_capturable_if_run_uncaptured'`라는 error가 발생합니다.

    - `torch>=1.13.0` , `cuda11.6`: docker hub에서 제공되는 torch관련 image들은 torch의 version이 1.13을 넘어가는 순간부터 cuda toolkit의 최소 적용 version이 `11.6`이게 됩니다. 

      이는 mmcv, mmdet의 의존성 제거 과정에서 포함한 cpython프로그래밍 함수를 import하지 못하게 되었습니다.

  - 사용 가능한 Linux device가 1개 뿐이라 master node에서 프로젝트를 진행했습니다.

    kubernetes는 1.13부터 Daemonset이 기본적으로 master node에서 schedule되지 않게 했기 때문에, taint를 해제한 상태로 프로젝트를 진행하고 있습니다.

- **mmcv, mmdet의 의존성 제거**

  docker image를 build하는 과정에서, mmcv의 설치 명령어를 포함한 `RUN`명령어가 동작하지 않는 문제가 있었습니다. (`pip install -r requirements.txt`역시 마찬가지)

  때문에 mmcv, mmdet를 `pip install`을 하지 않고도 학습 및 추론이 이루어져야 했기 때문에 [sub_module](https://github.com/HibernationNo1/sub_module)에 해당 module로부터 필요한 모든 code를 옮겨 관리하도록 했습니다.

- **persistance volume 구성**

  pipeline의 component에서 사용자 정의 code를 사용할 경우 PyPI에 package를 등록하여 pip로 설치하는 것은 보안문제가 될 수 있다고 판단했습니다.

  때문에 필요한 code는 private repository로 관리한다는 가정 하에 persistance volume으로 공유하도록 했습니다.

  이를 위해 component의 환경이 되는 docker image는 build시 ssh인증 파일이 포함되도록 했습니다.

- **The input parameter length exceed maximum size of 10000.**

  pipeline구성 시 input parameter를 전달할 때 model의 configuration을 포함 한 여러 지정값을 전달 시 text가 10000자 이상으로 넘어갈 시 발생하는 error입니다. 

  해당 namespace에 configmap을 생성하여 maximum size를 늘리고자 했지만 동작하지 않았습니다.

  때문에 configuration file을 [sub_module](https://github.com/HibernationNo1/sub_module)에 포함했습니다.

  input parameter를 전달할 때는 모든 값이 boolean type인, configuration과 같은 구조의 dict를 전달하도록 했습니다.

  component시작 시 input parameter와 configuration file을 대조하도록 했으며, input parameter의 특정 값이 False이 아닌 경우만 configuration의 지정 값에 수정이 적용되도록 했습니다.

  이를 통해 새로운 version의 pipeline을 구성할 때마다 input parameter의 최대 사이즈에 도달하지 않고도 사용자가 원하는 flag및 configuration값을 전달하는것에 문제가 없도록 했습니다.

- **아마존 웹 서비스(AWS)상에서 개발**

  초기에는 AWS의 cloud에서 cluster를 구축하여 여러 node를 활용하여 개발을 진행하려 했습니다. 하지만 아래 두 가지 문제에 직면하여 포기하게 되었습니다.

  - 비용 문제

    GPU를 활용하는 cluster를 구축 시 사용될 예상 청구비용과, PC와 GPU구입 비용을 비교를 하며 고민을 했었습니다. 그리고 제가 내린 결론은 PC구입이였습니다.

  - 보안 사고

    자격 인증 key가 알 수 없는 경로로 유출되어 여러 유럽 서버에서 EC2가 생성되어 순식간에 수십만원의 비용이 청구되는 사고가 있었습니다. 

    이후 PC를 포맷하고 support팀의 안내에 따라 여러 보안 설정을 마치고 비용은 환불받을 수 있었지만, AWS cloud service는 추 후 더욱 깊히 공부를 한 후 진행을 해야겠다 생각하게 되었습니다.  

     



### Project management

- 해당 project는 총 4개의 repository에 의해 관리됩니다.

  - pipeline을 구성하는 [project_4_kubeflow_pipeline](https://github.com/HibernationNo1/project_4_kubeflow_pipeline)  (main repository)

    `sub_module`, `pipeline_dataset`과 같은 repository를 import함과 동시에, 별도의 git repository로 관리할 수 있도록 **`git-Submodules`**기능을 사용

  - dataset을 관리하는 [pipeline_dataset](https://github.com/HibernationNo1/pipeline_dataset.git)

  - component에서 사용하는 module을 관리하는 [sub_module](https://github.com/HibernationNo1/sub_module)

  - model serving에 관한 [project_4_kserve](https://github.com/HibernationNo1/project_4_kserve)  

- [project_4_kubeflow_pipeline](https://github.com/HibernationNo1/project_4_kubeflow_pipeline),  [project_4_kserve](https://github.com/HibernationNo1/project_4_kserve)와 [sub_module](https://github.com/HibernationNo1/sub_module)은 특정 목적을 가진 branch를 임의로 생성, 삭제하며 운영 및 관리했으며, master branch에 merge를 하는경우 약식으로 github의 **`Pull requests`**기능을 사용하여 협업 환경에서 프로젝트를 진행하는 것 처럼 시뮬레이션을 해 보았습니다.



### TODO List

앞으로 공부 및 진행 예정인 사항들입니다.

- **Monitoring with prometheus**

  prometheus를 활용한 resource 모니터링, Grafana 대시보드 구성

- **make docker image for service model inference**

  Inferenceservice를 상시 가동중인 server에 배포 및 해당 service에 대해 누구든 test를 진행할 수 있는 code구현

- **Create more dataset**

  dataset규모 증량을 통한 model성능 향상

- **Set private storage for Inferenceservice Storage**

  Inferenceservice resource생성 시 private 저장소 활용 (보안사고 예방)

- **Using CI/CD**

  GitHub Actions를 이용한 CI/CD 구축

- **Using other model (swin-transformer, Mask2Former)**

  GPU메모리 환경 개선 후 더욱 큰 model활용

  



### Installation Process

Ubuntu 20.04 환경에서 프로젝트를 진행했으며, 전체적인 tool 및 package의 설치과정은 **[여기](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/setting.md)**에서 확인하실 수 있습니다.





### 참고 문헌 및 강의

- **mmdet**: https://github.com/open-mmlab/mmdetection
- **mmvc**: https://github.com/open-mmlab/mmcv
- **Swin-Transformer**: https://arxiv.org/pdf/2103.14030.pdf
- **도서 - 컨테이너 인프라 환경 구축을 위한 쿠버네티스/도커**: http://www.yes24.com/Product/Goods/102099414
- **도서 - 쿠브플로우**: http://www.yes24.com/Product/Goods/89494414

- **Kubeflow 인프런 강의**: https://www.inflearn.com/course/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EC%97%94%EC%A7%80%EB%8B%88%EC%96%B4-%EC%8B%A4%EB%AC%B4
- **Mlops 패스트 캠퍼스 강의**: https://fastcampus.co.kr/data_online_mlops
- **Kubeflow YouTube 강의**: https://www.youtube.com/@myoh0623

그외 각 kubernetes, kubeflow 공식 페이지 및 github...