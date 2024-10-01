ARG USE_CUDA=1
ARG CUDA_VERSION=11.2.2
ARG CUDNN_VERSION=8
ARG LINUX_DISTRO=ubuntu
ARG DISTRO_VERSION=20.04
ARG TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ARG PYTORCH_REPO=https://github.com/mengwanguc/pytorch-gpemu.git
ARG TORCHVISION_REPO=https://github.com/mengwanguc/vision-gpemu.git

ARG BUILD_IMAGE=nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-${LINUX_DISTRO}${DISTRO_VERSION}

FROM ${BUILD_IMAGE} AS build-base

# Redeclare ARG variables in the build-base stage
ARG TORCH_CUDA_ARCH_LIST
ARG PYTORCH_REPO
ARG TORCHVISION_REPO

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Anaconda
ARG ANACONDA_VERSION=2021.11
RUN wget -q https://repo.anaconda.com/archive/Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh
RUN bash Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh -b
RUN rm -f Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh

ENV PATH="/root/anaconda3/bin:${PATH}"

# Install Python dependencies
RUN conda install -y astunparse numpy ninja pyyaml mkl mkl-include setuptools \
    cmake cffi typing_extensions future six requests dataclasses && \
    conda install -y -c pytorch magma-cuda112

# Clone repositories
WORKDIR /opt
RUN git clone ${PYTORCH_REPO} pytorch-gpemu && \
    git clone ${TORCHVISION_REPO} vision-gpemu

FROM build-base AS build-torch

# Redeclare ARG variables in the build-torch stage
ARG TORCH_CUDA_ARCH_LIST
ARG PYTORCH_VERSION_TAG=original-pytorch-v1.7

WORKDIR /opt/pytorch-gpemu
RUN git checkout ${PYTORCH_VERSION_TAG} && \
    git submodule update --init --recursive && \
    python setup.py install

ARG TORCHVISION_VERSION_TAG=original-torchvision-v0.8.2

WORKDIR /opt/vision-gpemu
RUN git checkout ${TORCHVISION_VERSION_TAG}
RUN python setup.py install
