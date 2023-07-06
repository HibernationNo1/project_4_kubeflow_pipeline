# Pipeline

pipeline의 component는 아래와 같이 간단하게 구현했습니다.

**dataset의 통합, 학습 및 model저장, 평가 및 우수 model분류, model test까지 전부 완료 되어야 해당 pipeline이 유효하다고 판단했습니다.**

![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/pipeline%20graph.png?raw=true)

- 각각의 component는 Persistence volume을 통해 보조 code를 import합니다.

  이를 통해 완성된 pipeline을 다시 upload하여 동작하는 경우에도 보조code의 수정사항을 바로 적용할 수 있도록 했습니다.

- 각각의 component는 필요한 데이터를 google storage에서 다운로드하고, 동작의 결과를 google storage에 업로드합니다.

  필요한 데이터를 google storage로부터 다운로드 하여 완성된 pipeline을 다시 upload하여 동작하더라도 수정된 dataset을 사용할 수 있도록 했습니다. 

**Recode** : labelme.exe를 통해 만들어진 annotations dataset을 training을 위한 `train_dataset.json`으로 통합합니다.

**Train** : model training을 진행합니다.

**Evaluate** : trained model을 통해 evaluation을 진행합니다.

**Test**: trained model을 통해 inferene를 진행합니다. 





##### Table of Contents

- [Record](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/Pipeline.md#record)
- [Train](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/Pipeline.md#train)
- [Evaluate](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/Pipeline.md#evaluate)
- [Test](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/Pipeline.md#test)







### Record

**labelme.exe를 통해 만들어진 annotations dataset을 training을 위한 `train_dataset.json`으로 통합합니다.**

통합 후 images는 google cloud에, 각 image및 dataset의 관련 정보는 DB에 commit합니다.

![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/Record.png?raw=true)

1. [pipeline_dataset](https://github.com/HibernationNo1/pipeline_dataset.git) repository를 `clone`하고 annotation dataset의 특정 version이 명시된 tag로 `checkout`합니다. 

2. [pipeline_dataset](https://github.com/HibernationNo1/pipeline_dataset.git) repository의 dvc file을 통해 google storage로부터 annotation dataset을 다운로드합니다.

   DB의 annotation dataset의 정보를 토대로 image list와 json list를 생성합니다.

3. annotation dataset을 model에 input으로 사용할 수 있도록 coco dataset과 같은 구조로 통합하여 training dataset을 구성합니다.

4. training dataset의 정보를 DB에 insert하고 `dvc add`, `add push`를 통해 google storage에 업로드합니다.

해당 code는 [record_op.py](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/component/record/record_op.py)에서 확인하실 수 있습니다.



### Train

**model training을 진행합니다.**

training과정에서 저장한 model과 logs, tensorboard는 volume에 저장하고 google storage로 push합니다.

![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/Training.png?raw=true)

1. [pipeline_dataset](https://github.com/HibernationNo1/pipeline_dataset.git) repository를 `clone`하고 training dataset의 특정 version이 명시된 tag로 `checkout`합니다. 

2. [pipeline_dataset](https://github.com/HibernationNo1/pipeline_dataset.git) repository의 dvc file을 통해 google storage로부터 training dataset을 다운로드합니다.

   DB의 training dataset의 정보를 토대로 Dataloader를 build합니다.

3. Model training과 validation을 진행합니다.

4. **Google cloud SDK**를 통해 trained model과 log파일들을  google storage에 업로드합니다.

해당 code는 [train_op.py](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/component/train/train_op.py)에서 확인하실 수 있습니다.



### Evaluate 

**trained model을 통해 evaluation을 진행합니다.**

![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/Evaluation.png?raw=true)

>  1번의 과정은 Train component와 동일합니다.

1. [pipeline_dataset](https://github.com/HibernationNo1/pipeline_dataset.git) repository를 `clone`하고 training dataset의 특정 version이 명시된 tag로 `checkout`합니다. 

2. [pipeline_dataset](https://github.com/HibernationNo1/pipeline_dataset.git) repository의 dvc file을 통해 google storage로부터 validation dataset을 다운로드합니다.

   DB의 training dataset의 정보를 토대로 Dataloader를 build합니다.

   **Google cloud SDK**를 통해 trained model을 다운로드합니다.

3. `mAp`, `dv_mAP`, `EIR` 등 사용자가 결정한 성능 평가 지표에 대한 결과값을 계산하고, 특정 지표값에 대해 성능이 좋은 model을 선별하여 저장합니다. 

4. **Google cloud SDK**를 통해 evaluation과정에서 선별된 model을 google storage에 업로드합니다.

해당 code는 [evaluate_op.py](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/component/evaluate/evaluate_op.py)에서 확인하실 수 있습니다.

### Test

**trained model을 통해 inferene를 진행합니다.** 

![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/Test.png?raw=true)

1. **Google cloud SDK**를 통해 Google storage의 **result bucket**으로부터 evaluation과정에서 선별된 model을 download합니다.

2. download한 model로 test dataset에 대해 inference를 진행하고, object detection의 결과를 시각화합니다.

   ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/images/inference%20result.png?raw=true)



해당 code는 [test_op.py](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/component/test/test_op.py)에서 확인하실 수 있습니다.





