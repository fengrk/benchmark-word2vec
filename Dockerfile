FROM python:3.6
MAINTAINER frkhit "frkhit@gmail.com"
RUN pip install numpy gensim pyxtools
COPY benchmark.py sgns.sikuquanshu.word.bz2 ./
ENTRYPOINT ["python"]
CMD ["-u", "benchmark.py"]
