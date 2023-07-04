# Model Serving

Kserve의 InferenceService를 기반으로 한 torchserve를 배포하여 model serving을 진행했습니다.

**Service의 목표는 이미지 데이터를 입력받아 License plate를 감지하고 해당 정보를 출력하는 것입니다.**

해당 내용에 관한 code및 모든 file은 [project4_kserve](https://github.com/HibernationNo1/project4_kserve) repository로 관리하고 있습니다.



#### Table of Contents

- Serving

  - Requirements
    - handler
    - docker image
    - storageUri
  - InferenceService

- Test-Result

  





## Serving

### Requirements

#### handler

torchserve를 통해 `.mar`파일을 만들기 위해 handler를 구성했습니다.

- `_load_model`을 통해 model을 build하는 과정에서 instance의 순서와 configuration를 **model.pth**으로부터 자동으로 load하도록 했습니다. 
- input data가 image이기 때문에  `image_processing`을 진행했습니다.

- `postprocess`를 통해 inference결과를  **license plate의 모양**, **plate의 이미지 상 좌표 및 크기**, **등록지역(두 자리 번호)**, **차종기호(A~E 중 하나)**, **일련번호(네 자리 번호)**로 변환하도록 했습니다.

해당 code는 [handler.py](https://github.com/HibernationNo1/project4_kserve/blob/master/archrive/handler.py)에서 확인하실 수 있습니다.





#### docker image

1. **dockerfile**

   layer 구성은 아래의 환경이 포함되도록 했습니다.

   - `Ubuntu20.04`
   - `torch1.12.1`, `cuda 11.3`
   - `python 3.8`
   - 그 외 requirements.txt에 포함된 python library

   해당 dockerfile은 [`docker/dockerfile`](https://github.com/HibernationNo1/project4_kserve/blob/master/docker/Dockerfile)에 있습니다.

2. **build and push**

   ```
   $ docker build docker --no-cache -t localhost:5000/kserve:0.1-gpu
   $ docker push localhost:5000/kserve:0.1-gpu
   ```

   - 해당 image를 torchserve가 동작하는 pod에서 pull하기 위해서는 아래 내용을 docker의 `deamon.json`에 추가했습니다.

     ```
     $ sudo vi /etc/docker/daemon.json
     ```

     ```
     {
     
      "insecure-registries": ["172.30.1.70:5000"]
     
     }
     ```

     ```
     $ systemctl restart docker
     ```





#### storageUri

1. **download model from google storage**

   사전에 google storage에 업로드 된 `.pth`파일을 내려받습니다.

   ```
   $ gsutil cp -r gs://model_storage_hibernation/models/pipeline.pth ./archrive/pipeline.pth 
   ```

2. **make archiver file**(`.mar`)

   `torch-model-archiver`명령어로 `.mar` 파일을 만들고 google storage에 업로드 할 derectory에 저장합니다. 

   ```
   $ torch-model-archiver \
   	--model-name pipeline \
   	--version 0.1 \
   	--serialized-file archrive/pipeline.pth \
   	--extra-files archrive/extra \
   	--handler archrive/handler.py \
   	--runtime python \
   	--export-path gs/model-store
   ```

   - `gs/model-store`

     해당 dir에는 `gs/config/config.properties` 와 `gs/model-store/pipeline.mar`이 위치하게 됩니다.

     **config.properties**

     ```
     inference_address=http://0.0.0.0:8095
     management_address=http://0.0.0.0:8091
     metrics_address=http://0.0.0.0:8092
     grpc_inference_port=7070
     grpc_management_port=7071
     enable_envvars_config=true	
     enable_metrics_api=true
     metrics_format=prometheus
     NUM_WORKERS=4
     number_of_netty_threads=4
     job_queue_size=10
     load_models=all
     number_of_gpu=1
     gpu_id=0
     install_py_dep_per_model=true
     model_store=/mnt/models/model-store	
     model_snapshot={"name": "startup.cfg", "modelCount": 1, "models": {"pipeline": {"1.0": {"defaultVersion": true, "marName": "pipeline.mar", "minWorkers": 1, "maxWorkers": 3, "batchSize": 4, "maxBatchDelay": 100, "responseTimeout": 120}}}}
     ```

     - torchserve동작 시 각각의 API통신을 위한 port를 지정했습니다.
     - pod의 resource에 대한 각각의 할당량을 지정했습니다. 
     - pod내부에서 사용자가 사용하게 될 python library를 추가로 설치할 수 있도록 했습니다.

     

3. upload `.mar` file to google storage

   torch archrive로 만들어진 `.mar` 파일은 google storage에 업로드하여 차 후 InferenceService의 pod에서 다운로드할 수 있도록 했습니다.

   pod에서 접근할 수 있도록, 해당 storage와 path는 각각 공개 access로 설정했습니다. 

   ```
   gsutil cp -r gs gs://model_storage_hibernation/torchserve
   ```

   

### InferenceService

1. create resource

   ```
   apiVersion: "serving.kserve.io/v1beta1"
   kind: "InferenceService"
   metadata:
     annotations:
       isdecar.istio.is/inject: "false"
     name: "kserve-torchserve"
   spec:
     predictor:
       pytorch:
         storageUri: "gs://model_storage_hibernation/torchserve/gs"
         resources:
           limits:
             nvidia.com/gpu: 1
             cpu: "12" 
             memory: "8Gi"
           requests:
             cpu: "6"
             memory: "2Gi" 
         image: "172.30.1.70:5000/kserve:0.1"
   ```

   ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/kserve_6.png?raw=true)

2. check resource

   - InferenceService

     ```
     $ kubectl get InferenceService kserve-torchserve -n pipeline
     ```

     ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/kserve_3.png?raw=true)

   - pod

     InferenceService가 정상적으로 배포되었고 동작하고 있다면, torchserve가 동작하는 pod가 생성됩니다.

     ```
     $ kubectl get pod -n pipeline -l serving.kserve.io/inferenceservice=kserve-torchserve
     ```

     ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/kserve_4.png?raw=true)

3. port-forward

   pod가 정상적으로 Running이라면, port-forward를 통해 request신호를 전달할 수 있도록 합니다.

   ```
   $ kubectl port-forward {pod_name} -n pipeline 8081:8080
   ```

   ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/kserve_5.png?raw=true)

---



## Test-Result

[request.py](https://github.com/HibernationNo1/project4_kserve/blob/master/request/request.py)를 통해 InferenceService에 신호를 보냅니다.

해당 code의 동작 순서는 아래와 같습니다.

1. image를 load한 후 Base64로 인코딩합니다.
2. Base64로 인코딩된 문자열을 UTF-8 문자열로 디코딩합니다.
3. UTF-8 문자열의 size가 6MB를 넘어갈 경우, image를 다시 load하고 resizing하여 1, 2번 과정을 거치더라도 6MB보다 작은 크기가 될 수 있도록 합니다.
4. 이미지를 디코딩 한 UTF-8 문자열을 포함한 post신호를 pod로 보냅니다.
5. 응답으로 받은 response를 사용자가 보기 쉽게 변환하여 보여줍니다.



test를 위해 API로 보낼 이미지는 아래 3개를 사용했습니다.

![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/kserve_1.png?raw=true)

```
$ python request/request.py {image_name}.jpg --kserve
```



아래는 response로 받은 결과입니다.

![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/kserve_2.png?raw=true)

response는 **plate의 모양**, **plate의 이미지 상 좌표 및 크기**, **등록지역(두 자리 번호)**, **차종기호(A~E 중 하나)**, **일련번호(네 자리 번호)**에 대한 정보를 담고 있습니다.

해당 데이터를 토대로 사용자가 보기 쉽게(`각 번호를 위치에 맞게, plate의 모양 타입에 맞게`) 변환하여 보여줍니다.



## TODO

1. dataset의 규모가 작아 아직 detection의 성공률이 낮습니다. 

   dataset을 늘려 정확도를 높히고자 합니다.

2. post를 보내는 python code를 담은 docker image를 만들고자 합니다.

   이미지만 있다면 어느 환경에서든 손 쉽게 해당 server에 대한 test를 시행할 수 있도록 하는 것이 목표입니다. 
