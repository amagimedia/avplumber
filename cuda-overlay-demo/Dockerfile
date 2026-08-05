FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AVPLUMBER_REVISION=workspace
ARG FFMPEG_TAG=n7.1.5
ARG NV_CODEC_HEADERS_TAG=n12.1.14.0

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
        xxd \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --quiet --branch "${NV_CODEC_HEADERS_TAG}" --depth 1 \
        https://github.com/FFmpeg/nv-codec-headers.git /tmp/nv-codec-headers \
    && make -C /tmp/nv-codec-headers install PREFIX=/usr/local \
    && rm -rf /tmp/nv-codec-headers

COPY deps/ffmpeg-patches /build/deps/ffmpeg-patches

RUN git clone --quiet --branch "${FFMPEG_TAG}" --depth 1 \
        https://github.com/FFmpeg/FFmpeg.git /tmp/ffmpeg \
    && git -C /tmp/ffmpeg config user.name "cuda-overlay-demo builder" \
    && git -C /tmp/ffmpeg config user.email "cuda-overlay-demo@local" \
    && git -C /tmp/ffmpeg am /build/deps/ffmpeg-patches/*.patch

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
    && make -j"$(nproc)" \
    && make install \
    && ldconfig \
    && rm -rf /tmp/ffmpeg

RUN /usr/local/cuda/bin/nvcc --version | grep -q 'release 11.7' \
    && /usr/local/bin/ffmpeg -hide_banner -filters | grep -q ' overlay_many_cuda '

WORKDIR /build

COPY deps /build/deps
COPY Makefile generate_node_list /build/
COPY src /build/src
COPY pyplumber /build/pyplumber

RUN git init --quiet /build \
    && git -C /build config user.name "cuda-overlay-demo builder" \
    && git -C /build config user.email "cuda-overlay-demo@local" \
    && git -C /build commit --quiet --allow-empty -m "workspace ${AVPLUMBER_REVISION}"

RUN make -j"$(nproc)" python_module \
        BUILD_TYPE=Release \
        HAVE_CUDA=1 \
        HAVE_DRM=0 \
        HAVE_GL=0 \
        HAVE_KAFKA=0 \
        HAVE_NVOF_FRUC=0 \
        HAVE_NVCC=1 \
        HAVE_NVJPEG=0 \
        HAVE_SCTE35=0 \
        HAVE_TENSORRT=0 \
        NEURAL_NET_COMMON=0 \
        NEURAL_NET_SPECIFIC=0 \
        CUDA_MODULE_IMAGE_FORMAT=ptx \
        NVCC=/usr/local/cuda/bin/nvcc \
        CUDA_ROOT=/usr/local/cuda \
        PKG_CONFIG_PATH=/usr/local/lib/pkgconfig \
        CXXFLAGS+=' -I/usr/local/include -I/usr/local/cuda/include' \
        LFLAGS+=' -Wl,--disable-new-dtags -L/usr/local/lib -Wl,-rpath,/usr/local/lib -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64'

RUN test -f /build/_avplumber"$(python3-config --extension-suffix)" \
    && ldd /build/_avplumber"$(python3-config --extension-suffix)" | grep -q '/usr/local/lib/libavfilter' \
    && LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64 \
       PYTHONPATH=/build \
       python3 -c 'import pyplumber'

COPY cuda-overlay-demo /build/cuda-overlay-demo

ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64
ENV AVPLUMBER_REVISION=${AVPLUMBER_REVISION}
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
ENV NVIDIA_VISIBLE_DEVICES=all
ENV PYPLUMBER_PATH=/build
ENV PYTHONPATH=/build

ENTRYPOINT ["python3", "/build/cuda-overlay-demo/tools/run_matrix.py"]
