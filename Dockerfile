# Docker for zhrtvc
FROM pytorch/pytorch:nightly-devel-cuda10.0-cudnn7
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}

MAINTAINER kuangdd <kuangdd@foxmail.com>

ARG HOME="/home"

RUN apt-get update -y --fix-missing

RUN pip install -U pip && \
    pip config set global.index-url http://mirrors.aliyun.com/pypi/simple && \
    pip config set install.trusted-host mirrors.aliyun.com

# RUN pip install numpy scipy matplotlib librosa==0.6.0 tensorflow-gpu==1.15.2 tensorboardX inflect==0.2.5 Unidecode==1.0.22 pillow jupyter aukit phkit umap-learn tqdm numba==0.48 librosa pydub webrtcvad-wheels pyyaml setproctitle

RUN pip install tensorflow-gpu==1.15.2

RUN pip install sounddevice && \
    pip install -U aukit && \
    pip install -U phkit && \
    pip install tensorboardX==2.1 && \
    pip install numpy && \
    pip install numba==0.48 && \
    pip install librosa==0.6.0 && \
    pip install pydub && \
    pip install visdom && \
    pip install PyYAML && \
    pip install SIP && \
    pip install PyQt5 && \
    pip install inflect==0.2.5 && \
    pip install Unidecode && \
    pip install music21 && \
    pip install setproctitle && \
    pip install xmltodict && \
    pip install webrtcvad-wheels && \
    pip install scipy && \
    pip install matplotlib && \
    pip install --no-dependencies umap_learn

RUN cd $HOME && mkdir zhrtvc && cd zhrtvc
WORKDIR $HOME/zhrtvc

COPY ./ ./

CMD ["ls"]
