ARG PYTORCH="1.9.0"
ARG CUDA="10.2"
ARG CUDNN="7"   

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

# Install MMCV
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html

# install custom package
RUN pip install pipeline_taeuk4958==1.3.3
RUN pip install mmdet_taeuk4958==1.0.1

RUN pip install dvc[gs]
RUN pip install google-cloud-storage
RUN pip install opencv-python-headless==4.6.0.66
RUN pip install Pillow

