FROM python:3.8
ENV PYTHONUNBUFFERED 1

COPY ./ ./  # 소용없음

RUN pip install -r requirements.txt

# docker build -t labelme .
# docker tag labelme:latest hibernation4958/labelme:0.2
# docker push hibernation4958/labelme:0.2