#!/bin/bash

# Install dependencies from apt
sudo apt-get install -y libfftw3-dev libsndfile1-dev libgoogle-glog-dev libopenmpi-dev libboost-all-dev
# Install Kenlm
cd /tmp && git clone https://github.com/kpu/kenlm && cd kenlm && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make install -j$(nproc)
# Download and unpack ArrayFire v3.7.1
cd /tmp && wget https://arrayfire.s3.amazonaws.com/3.7.1/ArrayFire-v3.7.1-1_Linux_x86_64.sh
mkdir -p /opt/arrayfire
bash /tmp/ArrayFire-v3.7.1-1_Linux_x86_64.sh --skip-license --prefix=/opt/arrayfire
rm /tmp/ArrayFire-v3.7.1-1_Linux_x86_64.sh
# Remove some downloaded libs from the ArrayFire installer to avoid double linkeage
rm /opt/arrayfire/lib64/libnvrtc* /opt/arrayfire/lib64/libcu* /opt/arrayfire/lib64/libiomp*
# Install Intel MKL 2020
cd /tmp && wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB && \
    apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list' && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends intel-mkl-64bit-2020.0-088
# Remove existing MKL libs to avoid double linkeage
rm -rf /usr/local/lib/libmkl*
# Grab CMake 3.10.2
cd /opt  && wget https://github.com/Kitware/CMake/releases/download/v3.10.2/cmake-3.10.2-Linux-x86_64.tar.gz && \
    tar -xzf cmake-3.10.2-Linux-x86_64.tar.gz && rm /usr/local/bin/cmake && ln -s /opt/cmake-3.10.2-Linux-x86_64/bin/cmake /usr/local/bin/cmake

# CUDA-backend specific
# Use CUDA 10.0 - symlink to /usr/local/cuda
rm /usr/local/cuda && ln -s /usr/local/cuda-10.0 /usr/local/cuda

# CPU-backend specific
mkdir -p /opt/dnnl && cd /opt/dnnl && \
    wget https://github.com/oneapi-src/oneDNN/releases/download/v2.0/dnnl_lnx_2.0.0_cpu_iomp.tgz && \
    tar -xf dnnl_lnx_2.0.0_cpu_iomp.tgz
# Download and install Gloo
cd /tmp && git clone https://github.com/facebookincubator/gloo.git && cd gloo && \
    mkdir -p build && cd build && cmake .. -DUSE_MPI=ON && make install -j$(nproc)

# Environment variables need to be set outside of this script:
# %env MKLROOT=/opt/intel/mkl
# %env ArrayFire_DIR=/opt/arrayfire/share/ArrayFire/cmake
# %env DNNL_DIR=/opt/dnnl/dnnl_lnx_2.0.0_cpu_iomp/lib/cmake/dnnl
