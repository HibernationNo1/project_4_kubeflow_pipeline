FROM python:3
ENV PYTHONUNBUFFERED 1

RUN mkdir /code
WORKDIR /code
COPY . /code/

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirement.txt
ENTRYPOINT ["python", "labelme.py"]