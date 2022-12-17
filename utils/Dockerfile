ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"   

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel	

ENV TORCH_CUDA_ARCH_LIST="7.5"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"	

COPY ./requirements.txt ./requirements.txt
# install custom package
RUN pip install -r requirements.txt

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y git

# git repo version fixed in the docker image even the docker image tag version is a different.
# run this command in python code
# RUN git clone https://github.com/HibernationNo1/pipeline_ann_dataset.git



