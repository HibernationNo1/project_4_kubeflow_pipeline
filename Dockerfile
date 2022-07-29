FROM python:3.8
ENV PYTHONUNBUFFERED 1

COPY requirements.txt requirements.txt


RUN pip install -r requirements.txt

# docker build -t 0729 .
# docker tag 0729:latest hibernation4958/0729:0.1
# docker push hibernation4958/0729:0.1