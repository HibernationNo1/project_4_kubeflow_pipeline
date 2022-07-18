FROM python:3
ENV PYTHONUNBUFFERED 1

RUN /usr/local/bin/python -m pip install --upgrade pip

RUN mkdir /code
WORKDIR /code
COPY . /code/


WORKDIR /code
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "train.py"]