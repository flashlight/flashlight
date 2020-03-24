# ==================================================================
# module list
# ------------------------------------------------------------------
# Ubuntu           16.04
# OpenMPI          latest     (apt)
# cmake            3.10       (git)
# MKL              2018.4-057 (apt)
# arrayfire        3.6.4      (git, CPU backend)
# MKLDNN           4bdffc2    (git)
# Gloo             b7e0906    (git)
# ==================================================================

FROM ubuntu:16.04

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        wget \
        git \
        vim \
        emacs \
        nano \
        htop \
        g++ \
        # for MKL
        apt-transport-https \
        # for arrayfire CPU backend
        # OpenBLAS
        libopenblas-dev libfftw3-dev liblapacke-dev \
        # ATLAS
        libatlas3-base libatlas-dev libfftw3-dev liblapacke-dev \
        # ssh for OpenMPI
        openssh-server openssh-client \
        # OpenMPI
        libopenmpi-dev libomp-dev && \
# ==================================================================
# cmake 3.10 (for MKLDNN)
# ------------------------------------------------------------------
    apt-get purge -y cmake && \
    # for cmake
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL zlib1g-dev libcurl4-openssl-dev && \
    cd /tmp && wget https://cmake.org/files/v3.10/cmake-3.10.3.tar.gz  && \
    tar -xzvf cmake-3.10.3.tar.gz  && cd cmake-3.10.3  && \
    ./bootstrap --system-curl && \
    make -j8 &&  make install && cmake --version && \
# ==================================================================
# MKL https://software.intel.com/en-us/mkl
# ------------------------------------------------------------------
    cd /tmp && wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && \
    apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && \
    wget https://apt.repos.intel.com/setup/intelproducts.list -O /etc/apt/sources.list.d/intelproducts.list && \
    sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list' && \
    apt-get update && DEBIAN_FRONTEND=noninteractive $APT_INSTALL intel-mkl-64bit-2018.4-057 && \
    export MKLROOT=/opt/intel/mkl && \
# ==================================================================
# arrayfire with CPU backend https://github.com/arrayfire/arrayfire/wiki/
# ------------------------------------------------------------------
    cd /tmp && git clone --recursive https://github.com/arrayfire/arrayfire.git && \
    cd arrayfire && git checkout v3.6.4  && git submodule update --init --recursive && \
    mkdir build && cd build && \
    CXXFLAGS=-DOS_LNX cmake .. -DCMAKE_BUILD_TYPE=Release -DAF_BUILD_CUDA=OFF -DAF_BUILD_OPENCL=OFF -DAF_BUILD_EXAMPLES=OFF && \
    make -j8 && make install && \
# ==================================================================
# MKLDNN https://github.com/intel/mkl-dnn/
# ------------------------------------------------------------------
    cd /tmp && git clone https://github.com/intel/mkl-dnn.git && \
    cd mkl-dnn && git checkout 4bdffc2 && mkdir -p build && cd build && \
    cmake .. -DMKLDNN_USE_MKL=FULL && \
    make -j8 && make install && \
# ==================================================================
# Gloo https://github.com/facebookincubator/gloo.git
# ------------------------------------------------------------------
    cd /tmp && git clone https://github.com/facebookincubator/gloo.git && \
    cd gloo && git checkout b7e0906 && mkdir build && cd build && \
    cmake .. -DUSE_MPI=ON && \
    make -j8 && make install && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*
