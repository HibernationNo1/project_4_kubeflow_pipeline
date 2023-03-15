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
RUN apt-get update 

# for isntall opencv
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get -y install libglib2.0-0
RUN pip install opencv-python-headless

COPY ./ ./
RUN pip install -r requirements.txt

# install MMCV
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch${PYTORCH}/index.html
RUN pip install mmdet

RUN apt-get install -y git

  
# docker build . --no-cache -t localhost:5000/katib:0.5
