FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime


RUN apt-get update && apt-get install -y \
    wget \
    git


COPY . /private_investigator

WORKDIR /private_investigator

RUN pip install transformers==4.51.3 opacus==1.4.0

RUN pip install -e .



