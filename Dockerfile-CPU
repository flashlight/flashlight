# ==================================================================
# module list
# ------------------------------------------------------------------
# flashlight       master     (git, CPU backend)
# ==================================================================

FROM flml/flashlight:cpu-base-latest

# ==================================================================
# flashlight with CPU backend
# ------------------------------------------------------------------
# Setup and build flashlight
RUN mkdir /root/flashlight

COPY . /root/flashlight

RUN export MKLROOT=/opt/intel/mkl && cd /root/flashlight && mkdir -p build && \
    cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DFLASHLIGHT_BACKEND=CPU && \
    make -j8 && make install
