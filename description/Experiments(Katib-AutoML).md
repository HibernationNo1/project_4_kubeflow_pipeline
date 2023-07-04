# Experiments(Katib-AutoML)

##### Table of Contents

- [Resource](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/Experiments(Katib-AutoML).md#resource)
- [Dockerfile](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/Experiments(Katib-AutoML).md#dockerfile)
- [Status](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/Experiments(Katib-AutoML).md#status)
- [result](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/master/description/Experiments(Katib-AutoML).md#result)



Hyper-prameter tuning을 진행하기 위해 Kubeflow의 구성 요소인 Katib system을 사용했습니다.



## Resource

아래는 Experiment를 구성하는 과정에서 지정한 사항들입니다.

- 수집하고자 하는 metric은 `mAP`, `dv_mAP`, `EIR` 으로 하였으며, **goal**의 기준으로 할 metric은 `mAP`로 결정했습니다.

- parameters 결정 algorithm은 `random`으로 했습니다.

- 해당 프로젝트에서 사용되는 GPU의 memory가 두 개 이상의 trail을 실행하기에 충분치 않아 `parallelTrialCount`는 1로 결정했습니다.

- 각 parameter의 type을 `float`으로 설정 시 균일하지 못한 편차의 값이 연속적으로 할당되는 문제가 있어, `str` type으로 전달 후 code에서 각각 알맞은 type으로 변형되도록 했습니다.

- `metricsCollector`는 `StdOut`(출력)으로, format은 `name=value`형태로 결정했습니다.

- docker file은 local의 docker registry로부터 가져와 build하도록 했습니다.

  > docker image에 보안과 관련된 file이 포함될 경우 상정.



**`experiments.yaml`**

```yaml
apiVersion: kubeflow.org/v1beta1
kind: Experiment
metadata:
  namespace: pipeline
  name: katib-custom
spec:
  objective:
    type: maximize
    goal: 0.99
    objectiveMetricName: mAP
    additionalMetricNames:
      - dv_mAP
      - EIR
    metricStrategies:
    - name: mAP
      value: max
    - name: dv_mAP
      value: max
    - name: EIR
      value: max
  algorithm:
    algorithmName: random
  parallelTrialCount: 1
  maxTrialCount: 12
  maxFailedTrialCount: 5
  parameters:
    - name: lr
      parameterType: categorical
      feasibleSpace:
        list:
          - "0.0001"
          - "0.0005"
          - "0.001"
          - "0.00005"
          - "0.00001"
    - name: swin_drop_rate
      parameterType: categorical
      feasibleSpace:
        list:
          - "0.0"
          - "0.1"
          - "0.2"
          - "0.3"
          - "0.4"
    - name: swin_window_size
      parameterType: categorical
      feasibleSpace:
        list:
          - "3"
          - "5"
          - "7"
          - "9"
          - "11"
    - name: swin_mlp_ratio
      parameterType: categorical
      feasibleSpace:
        list:
          - "3"
          - "4"
          - "5"
  metricsCollectorSpec:
    collector:
      kind: StdOut
    source:
      filter:
        metricsFormat:
        - "([\\w|-]+)\\s*=\\s*((-?\\d+)(\\.\\d+)?)"
  trialTemplate:
    primaryContainerName: training-container
    trialParameters:
      - name: lr
        description: learning rate
        reference: lr
      - name: swin_drop_rate
        description: drop_rate of SwinTransformer
        reference: swin_drop_rate
      - name: swin_window_size
        description: window_size of SwinTransformer
        reference: swin_window_size
      - name: swin_mlp_ratio
        description: mlp_ratio of SwinTransformer
        reference: swin_mlp_ratio
    trialSpec:
      apiVersion: batch/v1
      kind: Job
      spec:
        template:
          metadata:
            annotations:
              sidecar.istio.io/inject: "false"
          spec:
            containers:
              - name: training-container
                image: localhost:5000/katib:0.1     # docker.io/hibernation4958/katib:0.1
                command:
                  - "python"
                  - "main.py"
                  - "--katib"
                  - "--cfg_train=config/train_cfg.py"
                  - "--model=MaskRCNN"
                  - "--epoch=20"
                  - "--lr=${trialParameters.lr}"
                  - "--swin_drop_rate=${trialParameters.swin_drop_rate}"
                  - "--swin_window_size=${trialParameters.swin_window_size}"
                  - "--swin_mlp_ratio=${trialParameters.swin_mlp_ratio}"
            restartPolicy: Never
```



create resource

```
$ kubectl apply -f experiments.yaml
```



----





## Dockerfile

dockerfile구성에 대한 내용과 설명입니다.

dockerfile은 [pytorch/pytorch](https://hub.docker.com/r/pytorch/pytorch)를 참조한 것과 Ubuntu20.04를 기반으로 만든 것 2가지가 있습니다.

아래는 Experiments를 진행시 build될 docker image의 dockerfile입니다.

### base pytorch

```dockerfile
ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"  

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel	

ENV TORCH_CUDA_ARCH_LIST="7.5"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"	

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN apt-get update

# for isntall opencv
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get -y install libglib2.0-0
RUN pip install opencv-python-headless

WORKDIR /home/workspace
RUN mkdir -p /home/workspace
COPY ./sub_module /home/workspace/sub_module
COPY ./test_dataset /home/workspace/test_dataset
COPY ./component /home/workspace/component
COPY ./config /home/workspace/config
COPY ./sh /home/workspace/sh
COPY ./pipeline_base_config.py ./pipeline_config.py ./pipeline_utils.py ./pipeline.py \
	 ./main.py ./requirements.txt /home/workspace/
RUN pip install -r requirements.txt

RUN pip install --no-cache-dir --upgrade pip wheel setuptools

RUN apt-get install -y git

RUN mkdir /root/.ssh
ADD id_rsa /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts


ENTRYPOINT ["sh", "sh/local_train.sh"]
  
```

- `TORCH_CUDA_ARCH_LIST`: 해당 device의 GPU compute capability를 설정합니다.

  > [여기](https://developer.nvidia.com/cuda-gpus#compute) 에서 CUDA-Enabled GeForce and TITAN Products를 누르면 각 CPU에 대한 compute capability를 확인할 수 있습니다.

- `TORCH_NVCC_FLAGS`: 추가 NVIDIA CUDA compiler driver flag를 의미합니다.

- `CMAKE_PREFIX_PATH` : dependencies에 대한 search path를 명시하며, conda사용을 위해 conda 위치를 명시했습니다.

- ```
  RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
  RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub
  RUN apt-get update
  ```

  `apt-get update`실행 시 GPG key error가 발생하는 것을 막기 위해 특정ubuntu version의 apt를 download합니다. 

  이후 `apt-get update`를 실행합니다.

- ````
  RUN apt-get -y install libgl1-mesa-glx
  RUN apt-get -y install libglib2.0-0
  RUN pip install opencv-python-headless
  ````

  해당 docker contianer에서 opencv package를 사용하기 위한 install입니다.

- ```
  RUN mkdir /root/.ssh
  ADD id_rsa /root/.ssh/id_rsa
  RUN chmod 600 /root/.ssh/id_rsa
  RUN touch /root/.ssh/known_hosts
  RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
  ```

  code상에서 `git clone`을 실행할 때 타겟 repository가 private인 경우를 위해 ssh key를 copy후 등록합니다.

- ```
  ENTRYPOINT ["python", "main.py"]
  ```

  Experiment실행 시 container의 동작을 명시합니다.



### base ubuntu:20.04

```dockerfile
FROM ubuntu:20.04
LABEL org.opencontainers.image.ref.name=ubuntu
LABEL org.opencontainers.image.version=20.04

ARG TARGETARCH=amd64
ENV DEBIAN_FRONTEND=noninteractive

# ADD file:d05d1c0936b046937bd5755876db2f8da3ed8ccbcf464bb56c312fbc7ed78589 in / 
CMD ["/bin/bash"]

# install package
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl vim sudo wget git


# set CUDA
ENV NVARCH=x86_64
ENV NVIDIA_REQUIRE_CUDA=cuda>=11.3 brand=nvidia,driver>=470,driver<471 
ENV NV_CUDA_COMPAT_PACKAGE=cuda-compat-11-3
ENV NV_CUDA_CUDART_PACKAGE=cuda-cudart-11-3
RUN apt-get update && \
    apt-get install -y --no-install-recommends gnupg2 ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH}/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
    rm -rf /var/lib/apt/lists/* 
RUN apt-get update && \
    apt-get install -y --no-install-recommends ${NV_CUDA_CUDART_PACKAGE}\
    ${NV_CUDA_COMPAT_PACKAGE} && \
    rm -rf /var/lib/apt/lists/* 
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf 
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# install python3, package-python, java
ENV PYTHONUNBUFFERED=TRUE
ARG PYTHON_VERSION=3.8
RUN apt-get update --fix-missing && apt-get upgrade -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt remove python-pip && \
	apt-get install -y python3-pip && \
    apt-get install --no-install-recommends -y \
    python$PYTHON_VERSION \
    python3-distutils \
    python$PYTHON_VERSION-dev \
    python$PYTHON_VERSION-venv \
    openjdk-17-jdk \
    build-essential \
    && rm -rf /var/lib/apt/lists/* 
# create symbolic link for use `python` command for `python3`
RUN ln -s /usr/bin/python3 /usr/local/bin/python

ENV PATH /home/venv/bin${PATH:+:${PATH}}

WORKDIR /home/workspace
RUN mkdir -p /home/workspace
COPY ./sub_module /home/workspace/sub_module
COPY ./test_dataset /home/workspace/test_dataset
COPY ./component /home/workspace/component
COPY ./config /home/workspace/config
COPY ./sh /home/workspace/sh
COPY ./pipeline_base_config.py ./pipeline_config.py ./pipeline_utils.py ./pipeline.py \
	 ./main.py ./requirements.txt /home/workspace/


# install requirements
RUN pip install -r requirements.txt
RUN pip install opencv-python-headless
RUN pip install torch==1.12.1+cu113  -f https://download.pytorch.org/whl/torch_stable.html

ENTRYPOINT ["sh", "sh/local_train.sh"]
```





**COPY**

- `COPY `

   **1. Experiments를 진행 시**

  **2. local에서 code실행 시**

  **3. component상의 container에서 code실행 시** 

  위 3가지의 상이한 실행 환경을 위해 모두 같은 code상에서 문제 없이 실행되도록 하기 위해 필요한 code를 dockerfile에 복사하여 실행되도록 했습니다.

  
  
  - 예시) `record_op.py`의 code중 일부 
  
  ```
  	WORKSPACE = dict(component_volume = cfg['path']['component_volume'],       # pvc volume path on component container
                       local_volume = cfg['path'].get('local_volume', None),     # pvc volume path on local
                       docker_volume = cfg['path']['docker_volume'],     # volume path on katib container
                       work = cfg['path']['work_space']
                       )    
  
      # set package path to 'import {custom_package}'
      if __name__=="component.record.record_op":        
          docker_volume = f"/{WORKSPACE['docker_volume']}"
          
          if WORKSPACE['local_volume'] is not None:
              local_module_path = osp.join('/opt/local-path-provisioner', WORKSPACE['local_volume']) 
          else:
              local_module_path = osp.join(os.getcwd(), cfg['git']['package_repo'])       
          
          if osp.isdir(local_module_path):
              PACKAGE_PATH = os.getcwd()
              print(f"    Run `record` locally")
              
          elif osp.isdir(docker_volume):
              PACKAGE_PATH = docker_volume
              print(f"    Run `record` in docker container")
              
          else:
              raise OSError(f"Paths '{docker_volume}' and '{local_module_path}' do not exist!")
  
      if __name__=="__main__":    
          workspace = osp.join(os.getcwd(), cfg['git']['dataset']['repo'])
          os.makedirs(workspace, exist_ok = True)
          os.chdir(osp.join(os.getcwd(), cfg['git']['dataset']['repo']))
  
          assert osp.isdir(WORKSPACE['work']), f"The path '{WORKSPACE['work']}' is not exist!"
          assert osp.isdir(WORKSPACE['component_volume']), f"The path '{WORKSPACE['component_volume']}' is not exist!"
          print(f"    Run `record` in component for pipeline")
          PACKAGE_PATH = WORKSPACE['component_volume']
          # for import sub_module
          package_repo_path = osp.join(WORKSPACE['component_volume'], cfg['git']['package_repo'])        
          if not osp.isdir(package_repo_path):
              print(f" git clone 'sub_module' to {package_repo_path}")
              
              Repo.clone_from(f"git@github.com:HibernationNo1/{cfg['git']['package_repo']}.git", package_repo_path)
       
      sys.path.append(PACKAGE_PATH) 
  ```
  
  component의 container로 code를 실행 시  `__name__=="__main__"`이고, local 또는 docker run으로 code를 실행 시 `__name__=="component.record.record_op"` 임을 활용했습니다.





---



## Status

1. 해당 resource가 제대로 생성되고,  Running상태인지 확인합니다.

   ```
   $ kubectl -n project-pipeline get experiment katib-custom
   ```

2. 해당 resource가 문제 없이 생성되었다면, experiment에 의해 정의된 pod와 `Trail`이 진행되는 pod의 상태를 확인합니다.

   ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/katib_log_2.png?raw=true)

   `katib-local-random-f87dd5767-bnxxw` :  experiment에 의해 정의된 pod입니다.

   `katib-local-4jr6bwfq-bn8vt`:  `Trial`을 진행하는 pod로, 실제 code에 대한 log를 확인하고자 할 때 해당 pod의 log를 확인합니다.

   - log확인

     ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/katib_log_1.png?raw=true)

     해당 pod의 log를 확인하고자 하면 `training-container`와 `metrics-logger-and-collector`의 두 가지 metric을 선택할 수 있습니다.

     - `training-container`: container를 load하거나 run하는 과정에서 발생한 error를 확인할 수 있습니다.

     - `metrics-logger-and-collector`: image에 포함된 code의 진행 상황을 log로 출력합니다.

       ![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/katib_log_3.png?raw=true)







## result

experiment의 hyperparameter조합 결과는 해당 experiment의 Type이 `Succeeded`인 경우에만 알 수 있습니다.

```
$ kubectl get experiment -n pipeline katib-custom
```

```
NAME              TYPE        STATUS   AGE
katib-custom   Succeeded   True     23m
```



이후 kubeflow central dashboard를 통해 확인한 결과입니다. 

![](https://github.com/HibernationNo1/project_4_kubeflow_pipeline/blob/docs/description/Experiment.png?raw=true)