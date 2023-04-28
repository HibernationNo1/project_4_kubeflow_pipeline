# Customizing mmdetection, mmcv

model의 학습 code는 [open-mmlab](https://github.com/open-mmlab)/**[mmdetection](https://github.com/open-mmlab/mmdetection)**과 [open-mmlab](https://github.com/open-mmlab)/**[mmcv](https://github.com/open-mmlab/mmcv)**를 차용했으며, 몇 가지 수정을 통해 원하는 동작을 추가했습니다.

![](https://github.com/open-mmlab/mmdetection/raw/main/resources/mmdet-logo.png)

![](https://raw.githubusercontent.com/open-mmlab/mmcv/main/docs/en/mmcv-logo.png)

해당 문서는 project 진행 과정에서 위의 두 open source를 사용하며 수정 및 개선, 추가한 부분에 대한 설명을 적어놓았습니다.



`mmdetection`과 `mmcv`중 필요한 부분만 차용하여 하나의 폴더 아래 위치시킨 후 각각의 함수가 상호작용하여 호출할 수 있도록 전체적인 구성을 변경했습니다.

그 과정에서 여러 function과 class의 수많은 수정이 이루어졌으나 사소한 부분도 많이 포함되어 있기에 전부 적어놓지는 않았습니다.

**해당 code는 persistent volume에 위치시킨 후 component내에서 자유롭게 import할 수 있도록 하기 위해 [sub_module](https://github.com/HibernationNo1/sub_module) repository로 관리했습니다.**



### Add `configuration` and `instance name list` to the trained model file

- 기존 mmdet의 code는 학습 된 model을 이어서 학습하거나, inference를 진행할 때 

  해당 model의 configuration과 학습을 처음 시작 할 시점의 configuration의 값이 다르면 code error가 발생하게 됩니다.

  >  ex) hook의 값 변경, 특정 hyper parameter변경

  이를 방지하기 위해 학습 당시 사용했던 configuration를 model file에 추가하여 저장하도록 하였으며, inference 및 continue training시 저장된 configuration을 기반으로 model을 build하도록 했습니다.

- 기존 mmdet의 code는 training에 사용된 instance의 정보를 하드 코딩으로 저장하기 때문에 새 학습을 진행 할 때마다 training dataset에 new instance가 있거나 배열의 순서가 바뀌면 inference과정에서 object name에 대한 error 또는 wrong value가 출력될 수 있습니다.

  이런 번거로움을 없애기 위해 학습 당시 사용했던 training dataset의 instance list를 model file에 추가로 저장했으며, 해당 model을 통해 inference를 진행 시 자동으로 object의 name을 label로 할당하도록 했습니다.

  > - 기존 dataset 구축 module [mmdet_CocoDataset](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/datasets/coco.py)
  > - 변경 후 dataset 구축 module [Custom_dataset](https://github.com/HibernationNo1/sub_module/blob/568cbe11b2a76c22d545200463845013030a1048/mmdet/data/dataset.py) : METAINFO 삭제

  ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/model_save.png?raw=true)



### Add custom evaluation code

#### dv mAP

validation 및 evaluation을 진행할 때 object detection의 성능 평가 지표인 *mean average precision*(이하 mAP)을 계산하는 방법에 mask boundary point을 활용한 더욱 고차원적인 **custom** 평가 기능을 추가했습니다.

![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/DV%20mAP.png?raw=true)

- 위 1번은 서로 다른 label이지만 bounding box의 크기가 일치하기 때문에 **Intersection Of Union**(이하 IOU)가 높은 값으로 계산됩니다.

  이는 label을 잘못 추론하더라도 정답으로 계산될 수 있으며, model의 **mAP**의 계산 시 더 좋은 성능을 가진 것 처럼 보여질 수 있습니다.

- 하지만 위 2번 처럼 mask의 bounding points를 (point 개수를 기준으로) 3분할로 나눈 후 각각의 분할에 대해 bounding box를 구하고, IOU를 계산하여 평균을 내면 더욱 정확한 정확도를 얻을 수 있습니다.  

  이러한 방식은 label이 정답에 맞게 추론을 하더라도, mask를 더욱 정확히 추론했는지까지 확인할 수 있기 때문에 segmentation의 목적에 맞는 성능 평가 지표로 생각됩니다.

  > 위 2번은 y좌표를 기준으로 sort후 slicing하여 분할 한 것입니다. 실제 code에서는 x좌표를 기준으로도 수행하였습니다. 

  해당 code: [eval.py](https://github.com/HibernationNo1/sub_module/blob/568cbe11b2a76c22d545200463845013030a1048/mmdet/eval.py) line-529 `get_num_pred_truth()`



#### exact Inference Rate(EIR)

model의 inference 목표인 License Plate의 각 text를 인식하고  **등록지역(두 자리 번호)**, **차종기호(A~E 중 하나)**, **일련번호(네 자리 번호)**를 각각 추출해내는 것의 성공률을 의미합니다.

labeling진행 시 plate의 sub number과 main number의 각 영역을 하나의 instance로 추가하여 진행하였고, 해당 영역에 대한 predicted mask를 통해 License Plate의 각 text를 인식하도록 했습니다. 이러한 방식을 통해 

10개의 License Plate중 6개를 정확하게 추론했을 시 EIR는 0.6이 됩니다.

![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/License%20plate.png?raw=true)



### Modified original hooks 

- code진행 시 GPU와 RAM memory를 확인하는 code를 추가했습니다.

- log 출력 및 저장 시 제가 원하는 infomation까지 포함할 수 있도록 변경했습니다.
- model저장시 path와 name의 format을 더욱 쉽게 변경할 수 있게 했으며, 원하는 구간에서 추가적인 model saving이 실시되도록 했습니다.
- 학습 진행 시간과, 앞으로 남은 시간을 day-hour-minute-sec으로 변환하여 출력되도록 했습니다.



### Add custom hooks

해당 code: [costom.py](https://github.com/HibernationNo1/sub_module/blob/568cbe11b2a76c22d545200463845013030a1048/mmdet/hooks/custom.py)

#### Validation_Hook

validation을 진행하는 code를 추가했습니다.

- 기존의 code는 accuracy를 계산하는 code가 있었지만, 이를 삭제하고 **mAP**와 **DV mAP**, **EIR**를 계산하는 함수를 호출하도록 했습니다. 
- validation과정에서 model의 성능평가지표 중 사용자가 지정한 특정 keyword의 성능이 가장 좋은 model을 확인했을 경우 즉시 model을 save하는 code를 추가했습니다. 



#### TensorBoard_Hook

- local의 위치 또는 persistent volume(pipeline의 component로 학습이 진행될 때)의 특정 위치에 tensorboard의 event.out file을 저장하는 code를 추가했습니다.
- loss와 mAP등 원하는 값을 tensorboard에 기록하도록 하는 code를 추가했습니다.



#### Check_Hook

- model을 구성하는 특정 module이 configuration에 명시된 module과 맞는지 확인하는 code를 추가했습니다.
- dataset의 classes의 개수와 head에 명시된 instance의 개수가 맞는지 확인하는 code를 추가했습니다. 

- code진행 시 GPU와 RAM memory를 확인하고, 메모리 사용율이 특정 값 이상으로 높아질 경우 진행중인 함수를 건너뛰거나 code를 멈추도록 했습니다.



 