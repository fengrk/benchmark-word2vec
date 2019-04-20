FROM python:3.6
MAINTAINER frkhit "frkhit@gmail.com"
COPY requirements.txt benchmark.py sgns.sikuquanshu.word.bz2 ./
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["-u", "benchmark.py"]
