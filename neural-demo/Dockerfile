# syntax=docker/dockerfile:1.7

FROM fedora:42 AS base

ARG FFMPEG_TAG=n7.1.3
ARG CUDA_REPO_URL=https://developer.download.nvidia.com/compute/cuda/repos/fedora42/x86_64/cuda-fedora42.repo
ARG TENSORRT_ARCHIVE

RUN dnf -y install --setopt=install_weak_deps=False \
        ca-certificates \
        curl \
        dnf-plugins-core \
        findutils \
        git \
        tar \
    && dnf clean all

RUN dnf config-manager addrepo --from-repofile="${CUDA_REPO_URL}" \
    && dnf -y install --setopt=install_weak_deps=False \
        autoconf \
        automake \
        bash \
        boost-devel \
        cmake \
        diffutils \
        eigen3-devel \
        gcc \
        gcc-c++ \
        gnutls-devel \
        libcurl-devel \
        libtool \
        libwebp-devel \
        make \
        nasm \
        nv-codec-headers \
        openssl-devel \
        patch \
        perl \
        pkgconf-pkg-config \
        srt-devel \
        which \
        xxd \
        cuda-nvcc-13-0 \
        cuda-cudart-devel-13-0 \
        cuda-cudart-13-0 \
        cuda-crt-13-0 \
        cuda-driver-devel-13-0 \
        cuda-cccl-13-0 \
        cuda-culibos-devel-13-0 \
        libnpp-devel-13-0 \
        libnvjpeg-devel-13-0 \
        libnvptxcompiler-13-0 \
        libnvvm-13-0 \
    && dnf clean all

RUN mkdir -p /opt/tensorrt/bin /opt/tensorrt/include /opt/tensorrt/lib
COPY ${TENSORRT_ARCHIVE} /tmp/tensorrt.tar.gz
RUN tar -xzf /tmp/tensorrt.tar.gz -C /opt/tensorrt/bin --strip-components=2 --wildcards 'TensorRT-*/bin/trtexec' \
    && tar -xzf /tmp/tensorrt.tar.gz -C /opt/tensorrt/include --strip-components=2 --wildcards 'TensorRT-*/include/*' \
    && tar -xzf /tmp/tensorrt.tar.gz -C /opt/tensorrt/lib --strip-components=2 --wildcards \
        'TensorRT-*/lib/libnvinfer.so*' \
        'TensorRT-*/lib/libnvinfer_plugin.so*' \
        'TensorRT-*/lib/libnvonnxparser.so*' \
        'TensorRT-*/lib/libnvinfer_builder_resource*.so*' \
    && rm -f /tmp/tensorrt.tar.gz

WORKDIR /build

FROM base AS ffmpeg-builder

COPY deps/ffmpeg-patches /build/deps/ffmpeg-patches

RUN git clone --branch "${FFMPEG_TAG}" --depth 1 https://github.com/FFmpeg/FFmpeg.git /tmp/ffmpeg
RUN cd /tmp/ffmpeg \
    && git config user.name "avplumber neural demo builder" \
    && git config user.email "neural-demo@avplumber.local" \
    && git am /build/deps/ffmpeg-patches/*.patch

RUN cd /tmp/ffmpeg \
    && PKG_CONFIG_PATH="/usr/local/lib/pkgconfig" ./configure \
        --prefix=/usr/local \
        --libdir=/usr/local/lib \
        --incdir=/usr/local/include \
        --pkgconfigdir=/usr/local/lib/pkgconfig \
        --enable-gpl \
        --enable-nonfree \
        --enable-libnpp \
        --extra-ldflags=-lm \
        --extra-ldflags=-ldl \
        --enable-pthreads \
        --extra-libs=-lpthread \
        --enable-encoders \
        --enable-decoders \
        --enable-avfilter \
        --enable-muxers \
        --enable-demuxers \
        --enable-parsers \
        --enable-x86asm \
        --disable-debug \
        --disable-ffplay \
        --enable-ffprobe \
        --disable-libx264 \
        --disable-libx265 \
        --disable-libaom \
        --disable-libvpx \
        --extra-cflags=-I/usr/local/cuda-13.0/targets/x86_64-linux/include \
        --extra-ldflags=-L/usr/local/cuda-13.0/targets/x86_64-linux/lib \
        --enable-ffnvcodec \
        --enable-cuda-nvcc \
        --enable-cuda \
        --disable-libvpl \
        --enable-cuvid \
        --enable-nvdec \
        --enable-nvenc \
        --enable-shared \
        --disable-doc \
        --disable-htmlpages \
        --disable-manpages \
        --disable-podpages \
        --disable-txtpages \
        --enable-libwebp \
        --enable-gnutls \
        --enable-libsrt \
        --enable-bsf=scte35ptsadjust \
        --enable-bsf=dump_extra \
        --enable-bsf=h264_mp4toannexb \
        --enable-bsf=h264_redundant_pps \
        --enable-bsf=extract_extradata \
        --enable-bsf=aac_adtstoasc \
        --enable-protocol=file \
        --enable-protocol=udp \
        --enable-protocol=pipe \
        --enable-protocol=tls \
        --enable-protocol=http \
        --enable-protocol=https \
        --enable-protocol=rtmp \
        --enable-protocol=rtmps \
        --enable-protocol=tcp \
        --enable-protocol=fd \
        --enable-protocol=crypto \
        --enable-protocol=tee \
        --enable-protocol=concat \
        --enable-protocol=hls \
        --enable-protocol=rtp \
        --enable-protocol=rtsp \
        --enable-protocol=srt \
        --enable-muxer=flv \
        --enable-muxer=rtp \
        --enable-muxer=rtp_mpegts \
        --enable-muxer=mpegts \
        --enable-muxer=mp4 \
        --enable-muxer=matroska \
        --enable-demuxer=mpegts \
        --enable-demuxer=mov \
        --enable-demuxer=matroska \
        --enable-demuxer=flv \
        --enable-demuxer=aac \
        --enable-filter=aformat \
        --enable-filter=aresample \
        --enable-filter=scale_npp \
        --enable-encoder=aac \
        --enable-encoder=ac3 \
        --enable-encoder=mp2 \
        --enable-encoder=mp3 \
        --enable-encoder=opus \
        --enable-decoder=aac \
        --enable-decoder=h264 \
        --enable-decoder=h265 \
        --enable-decoder=ac3 \
        --enable-decoder=mp2 \
        --enable-decoder=mp3 \
        --enable-decoder=opus \
        --enable-parser=aac \
        --enable-parser=h264 \
        --enable-parser=hevc \
        --enable-parser=h265 \
        --enable-parser=ac3 \
        --enable-parser=mp3 \
        --enable-parser=mp2 \
        --enable-parser=opus \
        --extra-ldexeflags=-Wl,--disable-new-dtags,-rpath,/usr/local/lib \
        --nvcc=/usr/local/cuda-13.0/bin/nvcc \
        --nvccflags="-gencode arch=compute_120,code=sm_120 -gencode arch=compute_120,code=compute_120 -O2" \
    || { cat /tmp/ffmpeg/ffbuild/config.log; exit 1; } \
    && cd /tmp/ffmpeg \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig

RUN /usr/local/bin/ffmpeg -hide_banner -filters | grep -q ' overlay_many_cuda ' \
    && /usr/local/bin/ffmpeg -hide_banner -filters | grep -q ' scale_npp ' \
    && /usr/local/bin/ffmpeg -hide_banner -protocols | grep -q '^  srt$' \
    && /usr/local/bin/ffmpeg -hide_banner -muxers | grep -q ' rtp ' \
    && /usr/local/bin/ffmpeg -hide_banner -bsfs | grep -q '^dump_extra$' \
    && ldd /usr/local/bin/ffmpeg | grep -q '/usr/local/lib/libavfilter' \
    && ldd /usr/local/bin/ffmpeg | grep -q 'libnpp'

FROM base AS plan-builder

COPY neural-demo/build-models.sh /usr/local/bin/avp-neural-demo-build-models
RUN chmod +x /usr/local/bin/avp-neural-demo-build-models

ENTRYPOINT ["/usr/local/bin/avp-neural-demo-build-models"]

FROM ffmpeg-builder AS deps-builder

COPY deps /build/deps
COPY Makefile /build/Makefile

RUN make -C /build -j"$(nproc)" \
        deps/cpr/build/lib/libcpr.a deps/avcpp/build/src/libavcpp.a deps/cuda_loader/cuda_drvapi_dynlink.o \
        PKG_CONFIG_PATH=/usr/local/lib/pkgconfig \
        CXXFLAGS+=' -I/usr/local/include -I/usr/local/cuda-13.0/targets/x86_64-linux/include' \
        LFLAGS+=' -Wl,--disable-new-dtags -L/usr/local/lib -Wl,-rpath,/usr/local/lib -L/usr/local/cuda-13.0/targets/x86_64-linux/lib -Wl,-rpath,/usr/local/cuda-13.0/targets/x86_64-linux/lib'

FROM deps-builder AS builder

RUN dnf -y install --setopt=install_weak_deps=False ccache \
    && dnf clean all

COPY src /build/src
COPY generate_node_list /build/generate_node_list
COPY .git /build/.git

RUN --mount=type=cache,id=avplumber-neural-demo-avp-objs-f42-cuda13,target=/build/objs,sharing=locked \
    --mount=type=cache,id=avplumber-neural-demo-avp-ccache-f42-cuda13,target=/build/.ccache,sharing=locked \
    set -eux; \
    cache_sig="$( \
        printf '%s\n' \
            'HAVE_CUDA=1' \
            'HAVE_NVCC=1' \
            'HAVE_TENSORRT=1' \
            'NEURAL_NET_COMMON=1' \
            'NEURAL_NET_SPECIFIC=1' \
            'HAVE_NVOF_FRUC=0' \
            'NVCC=/usr/local/cuda-13.0/bin/nvcc' \
            'TENSORRT_ROOT=/opt/tensorrt' \
            'CXX=ccache g++' \
            'CXXFLAGS_EXTRA=-I/usr/local/cuda-13.0/targets/x86_64-linux/include' \
            'LFLAGS_EXTRA=-Wl,--disable-new-dtags -L/usr/local/lib -Wl,-rpath,/usr/local/lib -L/usr/local/cuda-13.0/targets/x86_64-linux/lib -Wl,-rpath,/usr/local/cuda-13.0/targets/x86_64-linux/lib -L/opt/tensorrt/lib -Wl,-rpath,/opt/tensorrt/lib'; \
        sha256sum /build/Makefile; \
        g++ --version | head -n 1; \
        /usr/local/cuda-13.0/bin/nvcc --version | tail -n 1 \
    )"; \
    if [ ! -f /build/objs/.avp-build-cache-signature ] || [ "$cache_sig" != "$(cat /build/objs/.avp-build-cache-signature)" ]; then \
        find /build/objs -mindepth 1 -maxdepth 1 -exec rm -rf {} +; \
    fi; \
    printf '%s\n' "$cache_sig" > /build/objs/.avp-build-cache-signature; \
    CCACHE_DIR=/build/.ccache \
    CCACHE_COMPRESS=1 \
    make -C /build -j"$(nproc)" \
        HAVE_CUDA=1 \
        HAVE_NVCC=1 \
        HAVE_TENSORRT=1 \
        NEURAL_NET_COMMON=1 \
        NEURAL_NET_SPECIFIC=1 \
        HAVE_NVOF_FRUC=0 \
        CXX='ccache g++' \
        NVCC=/usr/local/cuda-13.0/bin/nvcc \
        TENSORRT_ROOT=/opt/tensorrt \
        PKG_CONFIG_PATH=/usr/local/lib/pkgconfig \
        CXXFLAGS+=' -I/usr/local/cuda-13.0/targets/x86_64-linux/include' \
        LFLAGS+=' -Wl,--disable-new-dtags -L/usr/local/lib -Wl,-rpath,/usr/local/lib -L/usr/local/cuda-13.0/targets/x86_64-linux/lib -Wl,-rpath,/usr/local/cuda-13.0/targets/x86_64-linux/lib -L/opt/tensorrt/lib -Wl,-rpath,/opt/tensorrt/lib'

RUN test -x /build/avplumber \
    && ldd /build/avplumber | grep -q '/usr/local/lib/libavfilter' \
    && ldd /build/avplumber | grep -q '/opt/tensorrt/lib/libnvinfer'


FROM fedora:42 AS runtime

ARG PREBUILT_MODELS_DIR=neural-demo/.generated-models

RUN dnf -y install --setopt=install_weak_deps=False \
        bash \
        boost-system \
        boost-thread \
        ca-certificates \
        curl \
        gnutls \
        libcurl \
        libwebp \
        openssl-libs \
        python3 \
        srt \
    && dnf clean all

COPY --from=builder /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=builder /usr/local/bin/ffprobe /usr/local/bin/ffprobe
COPY --from=builder /usr/local/lib/libavcodec.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/libavdevice.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/libavfilter.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/libavformat.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/libavutil.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/libpostproc.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/libswresample.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/libswscale.so* /usr/local/lib/
COPY --from=builder /usr/local/cuda-13.0/targets/x86_64-linux/lib/libcudart.so* /usr/local/cuda-13.0/targets/x86_64-linux/lib/
COPY --from=builder /usr/local/cuda-13.0/targets/x86_64-linux/lib/libnpp*.so* /usr/local/cuda-13.0/targets/x86_64-linux/lib/
COPY --from=builder /usr/local/cuda-13.0/targets/x86_64-linux/lib/libnvjpeg.so* /usr/local/cuda-13.0/targets/x86_64-linux/lib/
COPY --from=builder /opt/tensorrt/lib/libnvinfer.so* /opt/tensorrt/lib/
COPY --from=builder /opt/tensorrt/lib/libnvinfer_plugin.so* /opt/tensorrt/lib/
COPY --from=builder /build/avplumber /usr/local/bin/avplumber
COPY ${PREBUILT_MODELS_DIR}/ /home/tensorrt/

COPY neural-demo/entrypoint.sh /usr/local/bin/avp-neural-demo-entrypoint
COPY neural-demo/janus_rtcp_feedback.py /usr/local/bin/avp-neural-demo-rtcp-feedback
COPY neural-demo/templates /opt/avp-neural-demo/templates

RUN chmod +x /usr/local/bin/avp-neural-demo-entrypoint /usr/local/bin/avp-neural-demo-rtcp-feedback \
    && mkdir -p /home/tensorrt /run/avp/input /run/avp/output /tmp/avp-sidecars /tmp/avp-rendered \
    && printf '%s\n' /usr/local/lib /usr/local/cuda-13.0/targets/x86_64-linux/lib /opt/tensorrt/lib > /etc/ld.so.conf.d/avp-local.conf \
    && ldconfig

ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda-13.0/targets/x86_64-linux/lib:/opt/tensorrt/lib
ENV PATH=/usr/local/bin:${PATH}

ENTRYPOINT ["/usr/local/bin/avp-neural-demo-entrypoint"]
