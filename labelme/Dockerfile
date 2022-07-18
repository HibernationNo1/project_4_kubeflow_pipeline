FROM python:3
ENV PYTHONUNBUFFERED 1

RUN adduser -D myuser
USER myuser

RUN pip install --upgrade pip

RUN mkdir /code
WORKDIR /code
COPY . /code/

RUN ls /code

RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python", "labelme.py"]