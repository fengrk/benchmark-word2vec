FROM python:3.6
MAINTAINER frkhit "frkhit@gmail.com"
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["-u", "benchmark.py"]