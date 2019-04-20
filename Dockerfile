FROM python:3.6
MAINTAINER frkhit "frkhit@gmail.com"
RUN apt update && apt install -y libopenblas-base libomp-dev && pip install numpy gensim pyxtools
RUN pip install paramiko
COPY sgns.sikuquanshu.word.bz2 ./
COPY benchmark.py ./
ENTRYPOINT ["python"]
CMD ["-u", "benchmark.py"]
