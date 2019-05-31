# ==================================================================
# module list
# ------------------------------------------------------------------
# Ubuntu           16.04
# CUDA             9.2
# CuDNN            7-dev
# arrayfire        3.6.4    (git, CUDA backend)
# OpenMPI          latest   (apt)
# ==================================================================

FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        git \
        vim \
        emacs \
        nano \
        htop \
        g++ \
        # ssh for OpenMPI
        openssh-server openssh-client \
        # OpenMPI
        libopenmpi-dev libomp-dev \
        # nccl: for flashlight
        libnccl2 libnccl-dev \
        libglfw3-dev && \
# ==================================================================
# arrayfire https://github.com/arrayfire/arrayfire/wiki/
# ------------------------------------------------------------------
    cd /tmp && git clone --recursive https://github.com/arrayfire/arrayfire.git && \
    cd arrayfire && git checkout v3.6.4 && git submodule update --init --recursive && \
    mkdir build && cd build && \
    CXXFLAGS=-DOS_LNX cmake .. -DCMAKE_BUILD_TYPE=Release -DAF_BUILD_CPU=OFF -DAF_BUILD_OPENCL=OFF -DAF_BUILD_EXAMPLES=OFF && \
    make -j8 && \
    make install && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/* && \

    # If the driver is not found (during docker build) the cuda driver api need to be linked against the
    # libcuda.so stub located in the lib[64]/stubs directory
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1
