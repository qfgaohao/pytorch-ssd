FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
MAINTAINER Michael Guarino <michael.guarino1@nbcuni.com>
COPY . /home/
WORKDIR /home/

RUN apt-get -qq update \
    && apt-get -qq -y install curl bzip2 \
    && apt-get install -yq libgl1-mesa-glx \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3 \
    && pip install awscli --upgrade \
    && apt-get install -y python-opencv \
    && pip install opencv-python \
    && pip install pandas \
    && pip install boto3 \
    && chmod +x *.sh

ENV PATH /opt/conda/bin:$PATH
