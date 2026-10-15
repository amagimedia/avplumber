# Ubuntu 22.04 / CUDA 11.7 mixer image, for a host whose driver is older than
# R615. Same layout as Dockerfile.fedora44: the Rust avplumber binary listens
# on 127.0.0.1:22422 and the entrypoint then runs the Python controller.
# libavcodec 62 selects ffmpeg8 and 63 selects ffmpeg9.
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AVPLUMBER_REVISION=workspace
ARG FFMPEG_TAG=n8.1
ARG NV_CODEC_HEADERS_TAG=n13.0.19.0
ARG BUILD_JOBS=4

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libboost-all-dev \
        libcurl4-openssl-dev \
        libdrm-dev \
        libegl-dev \
        libgles-dev \
        libglvnd-dev \
        libgnutls28-dev \
        libssl-dev \
        libtool \
        make \
        nasm \
        pkg-config \
        python3 \
        python3-dev \
        python3-numpy \
        python3-pil \
        python3-pip \
        xxd \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --quiet --branch "${NV_CODEC_HEADERS_TAG}" --depth 1 \
        https://github.com/FFmpeg/nv-codec-headers.git /tmp/nv-codec-headers \
    && make -C /tmp/nv-codec-headers install PREFIX=/usr/local \
    && rm -rf /tmp/nv-codec-headers

COPY deps/ffmpeg /build/deps/ffmpeg

RUN git clone --quiet --branch "${FFMPEG_TAG}" --depth 1 \
        https://github.com/FFmpeg/FFmpeg.git /tmp/ffmpeg \
    && git -C /tmp/ffmpeg config user.name "mixer-demo builder" \
    && git -C /tmp/ffmpeg config user.email "mixer-demo@local" \
    && bash /build/deps/ffmpeg/apply.sh /tmp/ffmpeg

RUN cd /tmp/ffmpeg \
    && ./configure \
        --prefix=/usr/local \
        --libdir=/usr/local/lib \
        --incdir=/usr/local/include \
        --pkgconfigdir=/usr/local/lib/pkgconfig \
        --enable-gpl \
        --enable-nonfree \
        --enable-shared \
        --disable-static \
        --enable-cuda \
        --enable-cuda-nvcc \
        --enable-cuvid \
        --enable-ffnvcodec \
        --enable-libdrm \
        --enable-nvdec \
        --enable-nvenc \
        --disable-libnpp \
        --disable-doc \
        --disable-htmlpages \
        --disable-manpages \
        --disable-podpages \
        --disable-txtpages \
        --extra-cflags=-I/usr/local/cuda/include \
        --extra-ldflags=-L/usr/local/cuda/lib64 \
        --extra-ldexeflags=-Wl,--disable-new-dtags,-rpath,/usr/local/lib \
        --nvcc=/usr/local/cuda/bin/nvcc \
        --nvccflags='-gencode arch=compute_70,code=compute_70 -O2' \
    && make -j"${BUILD_JOBS}" \
    && make install \
    && ldconfig \
    && rm -rf /tmp/ffmpeg

RUN for f in scale_cuda tonemap_cuda; do /usr/local/bin/ffmpeg -hide_banner -filters | grep -q " $f " || exit 1; done \
    && /usr/local/bin/ffmpeg -hide_banner -h filter=transition_cuda \
       | awk '$1 == "mode" && $3 ~ /T/ { m = 1 } $1 == "dip" { d = 1 } $1 == "color" && $3 ~ /T/ { c = 1 } \
              END { exit !(m && d && c) }'

# clang's libclang is what rusty_ffmpeg's bindgen links. Drop the toolchain
# after the binary is installed; the controller does not need rustc.
RUN apt-get update \
    && apt-get install -y --no-install-recommends clang libclang-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable

WORKDIR /build

COPY Cargo.toml Cargo.lock /build/
COPY build_support /build/build_support
COPY avplumber_f7k /build/avplumber_f7k
COPY avplumber_nodes /build/avplumber_nodes
COPY src /build/src
COPY deps/avcpp /build/deps/avcpp
COPY deps/rusty_ffmpeg /build/deps/rusty_ffmpeg
COPY deps/rsmpeg /build/deps/rsmpeg
COPY deps/include /build/deps/include
COPY deps/json /build/deps/json
COPY deps/cuda_loader /build/deps/cuda_loader
COPY pyplumber /build/pyplumber
COPY demos/mixer /build/demos/mixer
COPY docker-compose/mixer/deploy/l4/seed.py /build/docker-compose/mixer/deploy/l4/seed.py
COPY docker-compose/mixer/build-avplumber.sh /tmp/build-avplumber.sh
COPY docker-compose/mixer/avplumber-mixer.sh /usr/local/bin/avplumber-mixer

RUN . /root/.cargo/env \
    && BUILD_JOBS="${BUILD_JOBS}" bash /tmp/build-avplumber.sh \
    && rm -rf /root/.cargo /root/.rustup

RUN python3 -m pip install --no-cache-dir -r /build/pyplumber/mixer/gui/requirements.txt \
    && chmod 755 /usr/local/bin/avplumber-mixer

ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64
ENV AVPLUMBER_REVISION=${AVPLUMBER_REVISION}
ENV AVPLUMBER_MIXER_COMMAND=/usr/local/bin/avplumber-mixer
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
ENV NVIDIA_VISIBLE_DEVICES=all
ENV PYTHONPATH=/build
ENV PYTHONUNBUFFERED=1

EXPOSE 7777/tcp 22422/tcp

ENTRYPOINT ["/usr/local/bin/avplumber-mixer"]
