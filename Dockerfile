FROM python:3.8
ENV PYTHONUNBUFFERED 1

COPY requirements.txt requirements.txt


RUN pip install -r requirements.txt

# docker build -t labelme .
# docker tag labelme:latest hibernation4958/labelme:0.3
# docker push hibernation4958/labelme:0.3