FROM python:3.8
ENV PYTHONUNBUFFERED 1

RUN git clone https://github.com/HibernationNo1/project_4_kubeflow_pipeline.git
WORKDIR project_4_kubeflow_pipeline/

RUN pip install dvc[s3]
RUN pip install pipeline_taeuk4958==1.0.13
RUN pip install Pillow
RUN pip install tqdm
RUN pip install boto3  
# RUN pip install pipeline_taeuk4958==1.0.7
# RUN pip install Pillow
# RUN pip install tqdm
# RUN pip install boto3 

