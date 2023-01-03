ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"   

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel	

ENV TORCH_CUDA_ARCH_LIST="7.5"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"	

# RUN pip install hibernation_no1==0.2.31

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update 

# for isntall opencv
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get -y install libglib2.0-0
RUN pip install opencv-python-headless

COPY ./requirements.txt ./requirements.txt
# install custom package
RUN pip install -r requirements.txt
RUN pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
RUN pip install mmdet



RUN apt-get install -y git

RUN mkdir /root/.ssh
ADD id_rsa /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts


# git repo version fixed in the docker image even the docker image tag version is a different.
# run this command in python code

WORKDIR pipeline_dataset
RUN git init
RUN git remote add origin git@github.com:HibernationNo1/pipeline_dataset.git
RUN git pull git@github.com:HibernationNo1/pipeline_dataset.git

COPY ./hibernation_no1  /tmp/docker/hibernation_no1
