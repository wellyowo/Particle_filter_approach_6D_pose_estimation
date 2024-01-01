# Use an official CUDA base image
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
###################################### user #####################################

ENV SHELL=/bin/bash \
    USER=wellyuwe \
    UID=1000 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

ENV HOME=/home/${USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${UID} \
    ${USER} 

RUN echo "root:root" | chpasswd
RUN echo "${USER}:111111" | chpasswd
##################################### apt #####################################
# Install essential tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
        libjpeg-dev \
        libpng-dev && \
    rm -rf /var/lib/apt/lists/*

# Install CMake 3.17
RUN mkdir ~/temp && \
    cd ~/temp && \
    curl -LO https://cmake.org/files/v3.17/cmake-3.17.0-Linux-x86_64.sh && \
    sh cmake-3.17.0-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir && \
    rm -rf ~/temp

# Install Python 3.6
RUN apt-get update && apt-get install -y python3 python3-dev python3-pip && \
    ln -s /usr/bin/python3 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3 /usr/local/bin/python && \
    pip3 install --upgrade pip

# Install PyTorch (Adjust the version as needed)
# RUN pip3 install torch==1.7.0 torchvision==0.8.1
RUN pip3 install torch
RUN apt-get -o Acquire::ForceIPv4=true update && apt-get -yq dist-upgrade \
    && apt-get -o Acquire::ForceIPv4=true install -yq --no-install-recommends \ 
    locales \
    cmake \
    unzip \
    make \
    git \
    vim \
    gedit \
    wget \
    sudo \
    lsb-release \
    build-essential \
    net-tools \
    dirmngr \
    gnupg2 \
    mercurial \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    libpcap-dev \
    python3-pip \
    python3-setuptools \
    python3-opencv \
    python3-numpy \
    python3-empy  \
    python3-dev \ 
    python3-tk \ 
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install \
    scipy \
    transforms3d \
    open3d \
    scikit-image \
    mayavi \
    plyfile \
    dipy \
    torchvision
# Copy the current directory contents into the container at /workspace


##################################### setting #####################################

RUN chown -R ${USER}:${USER} ${HOME}/
RUN echo "${USER} ALL=(ALL)  NOPASSWD: ALL" > /etc/sudoers
RUN echo "export PATH=/home/${USER}/.codon/bin:$PATH" > /home/${USER}/.bashrc

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

USER ${USER}
WORKDIR ${HOME}
CMD ["/bin/bash"]