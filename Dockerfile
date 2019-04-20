FROM python:3.6
MAINTAINER frkhit "frkhit@gmail.com"
RUN apt update && apt install -y libopenblas-base libomp-dev && pip install numpy gensim pyxtools
RUN pip install paramiko
COPY benchmark.py sgns.sikuquanshu.word.bz2 ./
ENTRYPOINT ["python"]
CMD ["-u", "benchmark.py"]
