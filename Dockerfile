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
# docker build . --no-cache -t localhost:5000/katib:0.1
# docker push localhost:5000/katib:0.1
# docker run --rm -it localhost:5000/katib:0.2