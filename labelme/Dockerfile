FROM python:3.8
ENV PYTHONUNBUFFERED 1

RUN git clone https://github.com/HibernationNo1/project_dataset.git
WORKDIR project_dataset/

RUN pip install dvc[gs]
RUN pip install pipeline_taeuk4958==1.1.0
RUN pip install Pillow
RUN pip install tqdm

