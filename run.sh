#!/bin/bash

# download images
docker pull frkhit/benchmark-word2vec:latest

# run
docker logs -f $(docker run -d frkhit/benchmark-word2vec:latest)
