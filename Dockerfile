FROM python:3.8
ENV PYTHONUNBUFFERED 1

COPY requirements.txt requirements.txt


RUN pip install -r requirements.txt
CMD git clone https://github.com/HibernationNo1/test_repo.git