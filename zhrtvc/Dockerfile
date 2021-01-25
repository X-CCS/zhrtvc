# Docker for zhrtvc
FROM ubuntu:18.04

MAINTAINER kuangdd <kuangdd@foxmail.com>

RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list && \
    apt-get update && yes|apt-get upgrade && \
    apt-get install -y wget bzip2 && \
    # wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2019.10-Linux-x86_64.sh && \
    wget https://repo.continuum.io/archive/Anaconda3-2019.10-Linux-x86_64.sh && \
    bash Anaconda3-2019.10-Linux-x86_64.sh -b && \
    rm Anaconda3-2019.10-Linux-x86_64.sh && \
    apt-get install -y --no-install-recommends gcc && \
    apt-get install -y ffmpeg && \
    apt-get install -y vim && \
    apt-get install -y libsndfile1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH /root/anaconda3/bin:$PATH

RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --set show_channel_urls yes && \
    conda update conda

RUN pip install -U pip && \
    pip config set global.index-url http://mirrors.aliyun.com/pypi/simple && \
    pip config set install.trusted-host mirrors.aliyun.com

RUN pip install sounddevice && \
    pip install --no-dependencies umap_learn && \
    pip install -U phkit && \
    pip install tqdm && \
    pip install -U aukit && \
    pip install librosa==0.6.0 && \
    pip install pydub && \
    pip install visdom && \
    pip install SIP && \
    pip install PyQt5 && \
    pip install inflect && \
    pip install Unidecode && \
    pip install music21 && \
    pip install setproctitle && \
    pip install tensorboardX==2.1 && \
    pip install xmltodict && \
    pip install webrtcvad-wheels && \
    pip install torch==1.6.0 && \
    pip install tensorflow==1.15.2
