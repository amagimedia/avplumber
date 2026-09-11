FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AVPLUMBER_REVISION=workspace
ARG FFMPEG_TAG=n7.1.5
ARG NV_CODEC_HEADERS_TAG=n12.1.14.0
ARG MXL_TAG=v1.1.0-beta-1

# gcc-13 is required by the MXL SDK (C++20 features + newer libstdc++).
# Ubuntu 22.04 defaults to gcc-11, so pull gcc-13 from the ubuntu
# toolchain PPA and keep it alongside; FFmpeg and avplumber continue
# to use gcc-11.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        bison \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        flex \
        g++-13 \
        gcc-13 \
        git \
        libboost-all-dev \
        libcurl4-openssl-dev \
        libgnutls28-dev \
        libgstreamer1.0-dev \
        libgstreamer-plugins-base1.0-dev \
        libpcap-dev \
        libssl-dev \
        libtool \
        libx11-dev \
        make \
        nasm \
        ninja-build \
        pkg-config \
        python3 \
        python3-dev \
        python3-numpy \
        python3-pil \
        python3-pip \
        tar \
        unzip \
        xxd \
        zip \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 22.04 ships cmake 3.22 but the MXL SDK requires >= 3.24.
# Upgrade via pip so a modern cmake is on PATH before FFmpeg configure.
RUN python3 -m pip install --no-cache-dir --upgrade 'cmake>=3.28'

RUN git clone --quiet --branch "${NV_CODEC_HEADERS_TAG}" --depth 1 \
        https://github.com/FFmpeg/nv-codec-headers.git /tmp/nv-codec-headers \
    && make -C /tmp/nv-codec-headers install PREFIX=/usr/local \
    && rm -rf /tmp/nv-codec-headers

# --- Rust (MXL's ada-url dep pulled through vcpkg needs a rustc) --------
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo
ENV PATH=/opt/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --default-toolchain 1.88.0 --profile minimal \
    && rustc --version

# --- vcpkg (used by the MXL SDK's CMake toolchain) ----------------------
ENV VCPKG_ROOT=/opt/vcpkg
ENV VCPKG_FORCE_SYSTEM_BINARIES=1
RUN git clone --quiet https://github.com/microsoft/vcpkg.git "$VCPKG_ROOT" \
    && "$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics

# --- MXL SDK ------------------------------------------------------------
# Built with gcc-13 (required by the SDK). FFmpeg + avplumber below keep
# using the distro gcc-11 that the CUDA composition suite is validated
# with. libmxl statically links spdlog, so its pkg-config file is
# scrubbed of the private Requires before FFmpeg configure looks at it.
RUN git clone --quiet --branch "${MXL_TAG}" --depth 1 \
        https://github.com/dmf-mxl/mxl.git /tmp/mxl \
    && CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 \
       cmake -S /tmp/mxl -B /tmp/mxl/build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" \
        -DVCPKG_MANIFEST_MODE=ON \
        -DBUILD_DOCS=OFF \
        -DBUILD_TESTS=OFF \
    && cmake --build /tmp/mxl/build --parallel "$(nproc)" \
    && cmake --install /tmp/mxl/build \
    && ldconfig \
    && sed -i '/^Requires.private: spdlog$/d' /usr/local/lib/pkgconfig/libmxl.pc \
    && rm -rf /tmp/mxl

COPY deps/ffmpeg-patches /build/deps/ffmpeg-patches

RUN git clone --quiet --branch "${FFMPEG_TAG}" --depth 1 \
        https://github.com/FFmpeg/FFmpeg.git /tmp/ffmpeg \
    && git -C /tmp/ffmpeg config user.name "mixer-demo builder" \
    && git -C /tmp/ffmpeg config user.email "mixer-demo@local" \
    && git -C /tmp/ffmpeg am /build/deps/ffmpeg-patches/*.patch \
    # 0008 (MXL) was cherry-picked with -X theirs and pulled in a
    # handful of test-scaffolding lines from the fork that reference
    # files not present in n7.1.5. Strip them so `make` succeeds.
    && sed -i '/tests\/fate\/ogg-\(opus\|flac\|vorbis\)\.mak/d' /tmp/ffmpeg/tests/Makefile \
    && sed -i '/^# uri test if demuxer is enabled$/,+1d' /tmp/ffmpeg/tests/fate/mxl.mak

RUN cd /tmp/ffmpeg \
    && PKG_CONFIG_PATH=/usr/local/lib/pkgconfig \
       ./configure \
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
        --enable-libmxl \
        --enable-demuxer=mxl \
        --enable-muxer=mxl \
        --enable-protocol=mxl \
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

RUN /usr/local/bin/ffmpeg -hide_banner -filters | grep -q ' overlay_many_cuda ' \
    && /usr/local/bin/ffmpeg -hide_banner -filters \
       | awk '$2 == "transition_cuda" && $1 ~ /C/ { found = 1 } END { exit !found }' \
    && /usr/local/bin/ffmpeg -hide_banner -demuxers 2>/dev/null \
       | awk '$1 ~ /D/ && $2 == "mxl" { found = 1 } END { exit !found }' \
    && /usr/local/bin/ffmpeg -hide_banner -muxers 2>/dev/null \
       | awk '$1 ~ /E/ && $2 == "mxl" { found = 1 } END { exit !found }'

WORKDIR /build

COPY deps /build/deps
COPY Makefile generate_node_list /build/
COPY src /build/src
COPY pyplumber /build/pyplumber
COPY avpmixer /build/avpmixer

RUN git init --quiet /build \
    && git -C /build config user.name "mixer-demo builder" \
    && git -C /build config user.email "mixer-demo@local" \
    && git -C /build commit --quiet --allow-empty -m "workspace ${AVPLUMBER_REVISION}"

RUN make -j"$(nproc)" python_module \
        BUILD_TYPE=Release \
        HAVE_CUDA=1 \
        HAVE_DRM=0 \
        HAVE_GL=0 \
        HAVE_NVOF_FRUC=0 \
        HAVE_NVCC=1 \
        HAVE_NVJPEG=0 \
        HAVE_TENSORRT=0 \
        NEURAL_NET=0 \
        CUDA_MODULE_IMAGE_FORMAT=ptx \
        NVCC=/usr/local/cuda/bin/nvcc \
        CUDA_ROOT=/usr/local/cuda \
        PKG_CONFIG_PATH=/usr/local/lib/pkgconfig \
        CXXFLAGS+=' -I/usr/local/include -I/usr/local/cuda/include' \
        LFLAGS+=' -Wl,--disable-new-dtags -L/usr/local/lib -Wl,-rpath,/usr/local/lib -L/usr/local/cuda/lib64 -Wl,-rpath,/usr/local/cuda/lib64'

RUN test -f /build/_avplumber"$(python3-config --extension-suffix)" \
    && ldd /build/_avplumber"$(python3-config --extension-suffix)" \
       | grep -q '/usr/local/lib/libavfilter' \
    && LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64 \
       PYTHONPATH=/build \
       python3 -c 'import pyplumber'

COPY demos/mixer /build/demos/mixer
COPY demos/mxl /build/demos/mxl

RUN python3 -m pip install --no-cache-dir -r /build/demos/mixer/requirements.txt \
    && if [ -s /build/demos/mxl/requirements.txt ]; then \
        python3 -m pip install --no-cache-dir -r /build/demos/mxl/requirements.txt; \
    fi

ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64
ENV AVPLUMBER_REVISION=${AVPLUMBER_REVISION}
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
ENV NVIDIA_VISIBLE_DEVICES=all
ENV PYTHONPATH=/build
ENV PYTHONUNBUFFERED=1

EXPOSE 7777/tcp

ENTRYPOINT ["python3", "/build/demos/mixer/mixer.py"]
