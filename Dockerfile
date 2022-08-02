FROM python:3.8
ENV PYTHONUNBUFFERED 1

RUN mkdir workplace

COPY . ./workplace

WORKDIR workplace/

RUN pip install dvc[s3]

# RUN pip install pipeline_taeuk4958==1.0.7
# RUN pip install Pillow
# RUN pip install tqdm
# RUN pip install boto3 

