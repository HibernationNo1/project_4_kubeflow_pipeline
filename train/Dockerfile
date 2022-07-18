FROM python:3
ENV PYTHONUNBUFFERED 1

RUN python -m pip install requests

RUN mkdir /code
WORKDIR /code
COPY . /code/


WORKDIR /code
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "train.py"]