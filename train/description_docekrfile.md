

```
ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"   

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel	

ENV TORCH_CUDA_ARCH_LIST="7.5"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"	

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/HibernationNo1/project_dataset.git
WORKDIR project_dataset/

RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
RUN pip install google-cloud-storage
RUN pip install pipeline_taeuk4958==1.2.1
RUN pip install opencv-python
RUN pip install Pillow
RUN pip install pycocotools
```

- `FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel	`

   docker hub에서 PYTORCH, CUDA, CUDNN 의 version이 각각 호환되는 version별로 tag된 image가 있는지 확인 후 작성

- `ENV TORCH_CUDA_ARCH_LIST="7.5"`

  사용하는 GPU의 compute capability 값 [여기](https://developer.nvidia.com/cuda-gpus#compute)에서 확인

- `ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"	`

  Cmake 환경 변수에 anaconda/bin 하위 path를 할당 

- ```
  RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
  RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
  ```

  GPG key등록 방법

  `GPG error: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64  InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY A4B469963BF863CC` 라는 error 발생 시 `A4B469963BF863CC`를 확인할 수 있다.

  [여기](http://pgp.mit.edu/)에서 Search String 에 `0x`를 앞단에 붙여 `0xA4B469963BF863CC` 를 검색하면

  ```
  pub  4096R/3BF863CC 2022-04-14 cudatools <cudatools@nvidia.com>
  ```

  라고 뜨는걸 볼 수 있다.

  위 문구 중 `3BF863C`부분을 붙여 

  ```
  RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
  ```

  라고 붙임

