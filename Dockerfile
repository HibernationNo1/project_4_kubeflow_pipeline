FROM python:3.8
ENV PYTHONUNBUFFERED 1

RUN mkdir workplace

COPY . ./workplace

WORKDIR workplace/

RUN pip install dvc[s3]
RUN dvc remote modify --local bikes access_key_id 'AKIAUA6XTFHLBNEKA5X2'
RUN dvc remote modify --local bikes secret_access_key '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'

# RUN pip install pipeline_taeuk4958==1.0.7
# RUN pip install Pillow
# RUN pip install tqdm
# RUN pip install boto3 

