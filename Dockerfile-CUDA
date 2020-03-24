# ==================================================================
# module list
# ------------------------------------------------------------------
# flashlight       master   (git, CUDA backend)
# ==================================================================

FROM flml/flashlight:cuda-base-latest

# ==================================================================
# flashlight with GPU backend
# ------------------------------------------------------------------
# Setup and build flashlight
RUN mkdir /root/flashlight

COPY . /root/flashlight

RUN cd /root/flashlight && mkdir -p build && \
    cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DFLASHLIGHT_BACKEND=CUDA && \
    make -j8 && make install
