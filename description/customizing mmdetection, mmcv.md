# Customizing mmdetection, mmcv

##### Table of Contents

- [Add `configuration` and `instance name list` to the model file](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/customizing%20mmdetection%2C%20mmcv.md#add-configuration-and-instance-name-list-to-the-model-file)
- Post Processing
- [Add custom evaluation code](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/customizing%20mmdetection%2C%20mmcv.md#add-custom-evaluation-code)
  - [dv mAP](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/customizing%20mmdetection%2C%20mmcv.md#dv-map)
  - [exact Inference Rate(EIR)](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/customizing%20mmdetection%2C%20mmcv.md#exact-inference-rateeir)
- [Modify hooks](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/customizing%20mmdetection%2C%20mmcv.md#modify-hooks) 
  - [Modify Original hooks](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/customizing%20mmdetection%2C%20mmcv.md#modify-original-hooks) 
  - [Add custom hooks](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/customizing%20mmdetection%2C%20mmcv.md#add-custom-hooks)
    - [Validation_Hook](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/customizing%20mmdetection%2C%20mmcv.md#validation_hook)
    - [TensorBoard_Hook](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/master/docs/description/customizing%20mmdetection%2C%20mmcv.md#tensorboard_hook)
    - [Check_Hook](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/customizing%20mmdetection%2C%20mmcv.md#check_hook)
- [Remove dependencies](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/customizing%20mmdetection%2C%20mmcv.md#remove-dependencies)



model의 학습 code는 [open-mmlab](https://github.com/open-mmlab)/[mmdetection](https://github.com/open-mmlab/mmdetection)과 [open-mmlab](https://github.com/open-mmlab)/[mmcv](https://github.com/open-mmlab/mmcv)를 차용했으며, 몇 가지 수정을 통해 원하는 동작을 추가했습니다.

![](https://github.com/open-mmlab/mmdetection/raw/main/resources/mmdet-logo.png)

![](https://raw.githubusercontent.com/open-mmlab/mmcv/main/docs/en/mmcv-logo.png)

해당 문서는 project 진행 과정에서 위의 두 open source를 사용하며 수정 및 개선, 추가한 부분에 대한 설명을 적어놓았습니다.



`mmdetection`과 `mmcv`중 필요한 부분만 차용하여 하나의 폴더 아래 위치시킨 후 각각의 함수가 상호작용하여 호출할 수 있도록 전체적인 구성을 변경했습니다.

그 과정에서 여러 function과 class의 수많은 수정이 이루어졌으나, 사소한 부분이 상항히 많이 포함되어 있기에 전부 적어놓지는 않았습니다.

**해당 code는 persistent volume에 `git clone`으로 위치시킨 후 component내에서 자유롭게 import할 수 있도록 하기 위해 [sub_module](https://github.com/HibernationNo1/sub_module) repository로 관리했습니다.**

---



### Add `configuration` and `instance name list` to the model file

**Model load시 `config`와 같은 추가적인 파일을 load할 것 없이 model파일 내부에서 관련 내용을 가져와 training 및 inference를 진행할 수 있도록 했습니다.**

- **model을 load하는 과정의 필요조건을 줄였습니다.**

  기존 mmdet의 code는 학습 된 model로 **continue training**또는 **inference**를 진행할 때 configuration 파일을 필요로 합니다.

  이 때 load하는 model의 configuration과 학습을 처음 시작 할 시점의 configuration의 값이 다르면 code error가 발생하게 됩니다.

  >  ex) hook의 값 변경, 특정 hyper parameter변경

  이를 방지하기 위해 학습 당시 사용했던 configuration를 model 파일에 추가하여 저장하도록 하였으며, **inference** 및 **continue training**시 저장된 configuration을 기반으로 model을 build하도록 했습니다.

- **inference과정의 필요 조건을 줄였습니다.**

  기존 mmdet의 code는 training에 사용된 instance의 정보를 하드 코딩으로 저장했습니다.

  이는 학습을 진행 할 때마다 training dataset에 new instance가 있거나 배열의 순서가 바뀌면 inference과정에서 object name에 대한 error 또는 wrong value가 출력되는 문제가 발생하게 됩니다.
  
  이런 문제를 없애기 위해 학습 당시 사용했던 training dataset의 instance list를 model 파일에 추가로 저장했으며, 해당 model을 통해 inference를 진행 시 자동으로 object의 name을 label로 할당하도록 했습니다.
  
  > - 기존 dataset 구축 module [mmdet_CocoDataset](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/coco.py)
  > - 변경 후 dataset 구축 module [Custom_dataset](https://github.com/HibernationNo1/sub_module/blob/568cbe11b2a76c22d545200463845013030a1048/mmdet/data/dataset.py) : METAINFO 삭제
  
  ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/model_save.png?raw=true)



---



### Post Processing

Inference의 결과로 얻어진 데이터를 토대로 License plate의 정보를 추출합니다.

- Inference result 예시

  ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/License_plate_4.png?raw=true)

  위 사진에서 기대하는 post-ptocessing의 결과는 아래와 같습니다.

  ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/License_plate_3.png?raw=true)

  - `type`: License plate의 형태는 text가 두 줄로 이루어진 `r_board`와, text아 한 줄로 이루어진 `l_board`로 나뉩니다.

  - `board_center_p`: 검출된 License plate가 image의 어느 위치에 있는지 pixel단위로 나타냅니다.

  - `width`, `height `: 검출된 License plate의 width, height에 대해서 나타냅니다.

  - `sub_text`: 검출된 License plate의 text중 새 개로 이루어진 부분을 나타냅니다.

    마지막 text는 반드시 `a~e` 의 알파벳이 위치한 경우만 검출 성공으로 간주합니다.

  - `main_text`: 검출된 License plate의 text중 네 개로 이루어진 부분을 나타냅니다.



위의 정보를 얻기 위해 labeling단계부터 각 text외에도 `sub_text`, `main_text`영역까지 라벨링을 진행합니다.

아래는 target object에 대해 labeling을 완료 한 예시 사진입니다.

![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/License_plate_1.png?raw=true)



Post Processing은 아래의 순서로 이루어집니다.

1. 특정 board의 bounding box영역 내에 위치한 `sub_text`와 `main_text`가 모두 검출되었는지 확인.
2. 번호 및 알파벳의 중앙 위치를 각각의 text의 위치와 비교하여 몇 번째의 text인지 확인
3. `sub_text`와 `main_text`의 각 내부 text의 길이가 3, 4개가 맞는지 확인
4. `sub_text`내부의 text중 의 세 번째가 숫자가 아님(알파벳임)을 확인



### Add custom evaluation code

#### 1. dv mAP

**Object Detction의 성능 평가 지표인 mAP를 한 단계 더욱 고차원적인 방법으로 계산하는 성능 평가 지표를 만들었습니다.**

validation 및 evaluation을 진행할 때 object detection의 성능 평가 지표인 *mean average precision*(이하 mAP)을 계산하는 방법에 mask boundary point을 활용한 더욱 고차원적인 **custom** 평가 기능을 추가했습니다.

![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/DV%20mAP.png?raw=true)

- 위 1번은 서로 다른 label이지만 bounding box의 크기가 일치하기 때문에 **Intersection Of Union**(이하 IOU)가 높은 값으로 계산됩니다.

  이는 label을 잘못 추론하더라도 정답으로 계산될 수 있으며, model의 **mAP**의 계산 시 더 좋은 성능을 가진 것 처럼 보여질 수 있습니다.

- 하지만 위 2번 처럼 mask의 bounding points를 3분할(point 개수 기준)로 나눈 후 각각의 분할에 대해 bounding box를 구하고, IOU를 계산하여 평균을 내면 더욱 정확한 정확도를 얻을 수 있습니다.  

  이러한 방식은 label이 정답에 맞게 추론을 하더라도, mask를 더욱 정확히 추론했는지까지 확인할 수 있기 때문에 segmentation의 목적에 맞는 성능 평가 지표로 생각됩니다.

  > 위 2번은 y좌표를 기준으로 sort후 slicing하여 분할 한 것입니다. 실제 code에서는 x좌표를 기준으로 수행하는 것 까지 포함하였습니다. 

  해당 code: [eval.py](https://github.com/HibernationNo1/sub_module/blob/568cbe11b2a76c22d545200463845013030a1048/mmdet/eval.py) line-529 `get_num_pred_truth()`





#### 2. Exact Inference Rate(EIR)

**해당 proeject의 dataset를 위한 model 평가 지표를 추가했습니다.**

**Exact Inference Rate**는 model의 inference결과인 License Plate의 각 text를 인식하고  **등록지역(두 자리 번호)**, **차종기호(A~E 중 하나)**, **일련번호(네 자리 번호)**를 각각 추출해내는 것의 성공률, 즉 `실제_License Plate/추론_License Plate` 을 의미합니다.

labeling진행 시 plate의 sub number과 main number의 각 영역을 하나의 instance로 추가하여 진행하였고, 해당 영역에 대한 predicted mask를 통해 License Plate의 각 text를 인식하도록 했습니다. 

이러한 방식을 통해 계산하자면, 10개의 License Plate중 6개를 정확하게 추론했을 시 EIR는 0.6이 됩니다.

- 62개의 License Plate 중 정확하게 예측 한 plate의 개수가 28인 경우

   License_plate_2 사진



---



### Modify hooks 

#### Modify Original hooks 

**기존에 있던 여러 hook class에 제가 필요로 하는 기능을 추가했습니다.**

- code진행 시 GPU와 RAM memory를 확인하는 code를 추가했습니다.

- log 출력 및 저장 시 제가 원하는 infomation까지 포함할 수 있도록 변경했습니다.
- model저장시 path와 name의 format을 더욱 쉽게 변경할 수 있게 했으며, 원하는 구간에서 추가적인 model saving이 실시되도록 했습니다.
- 학습 진행 시간과, 앞으로 남은 시간을 day-hour-minute-sec으로 변환하여 출력되도록 했습니다.



#### Add custom hooks

**제가 필요로 하는 동작을 수행하는 hook class를 추가했습니다.**

해당 code: [costom.py](https://github.com/HibernationNo1/sub_module/blob/568cbe11b2a76c22d545200463845013030a1048/mmdet/hooks/custom.py)



##### 1. Validation_Hook

validation을 진행하는 code를 추가했습니다.

- 기존의 code는 accuracy를 계산하는 code가 있었지만, 이를 삭제하고 **mAP**와 **DV mAP**, **EIR**를 계산하는 함수를 호출하도록 했습니다. 
- validation과정에서 model의 성능평가지표 중 사용자가 지정한 특정 keyword의 성능이 가장 좋은 model을 확인했을 경우 즉시 model을 save하는 code를 추가했습니다. 



##### 2. TensorBoard_Hook

- local의 위치 또는 persistent volume(pipeline의 component로 학습이 진행될 때)의 특정 위치에 tensorboard의 event.out file을 저장하는 code를 추가했습니다.
- loss와 mAP등 원하는 값을 tensorboard에 기록하도록 하는 code를 추가했습니다.



##### 3. Check_Hook

- model을 구성하는 특정 module이 configuration에 명시된 module과 맞는지 확인하는 code를 추가했습니다.
- dataset의 classes의 개수와 head에 명시된 instance의 개수가 맞는지 확인하는 code를 추가했습니다. 

- code진행 시 GPU와 RAM memory를 확인하고, 메모리 사용율이 특정 값 이상으로 높아질 경우 진행중인 함수를 건너뛰거나 code를 멈추도록 했습니다.



 

---

### Remove dependencies

**pip를 통해 mmcv와 mmdet를 설치하지 않더라도 training, evaluation, inference를 진행할 수 있도록 했습니다.** 

mmcv와 mmdet는 서로 참조하는 클래스 및 함수가 상당히 많습니다.

하지만 Component, Kserve와 같은 resource의 환경이 되어주는 docker image를 구축하는 과정에서 해당 패키지가 설치되지 않는 문제가 있었습니다.

둘 중 하나라도 설치가 안되더라도 코드가 원활히 돌아가기 어려웠기 때문에, 독립적인 패키지로 통합하여 하나의 git repository로 관리하도록 했습니다.

해당 code는 [sub_module](https://github.com/HibernationNo1/sub_module)에 의해 관리하고 있습니다.

