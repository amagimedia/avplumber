# Minimal DMA-BUF consumer for the public demo: avplumber built with DRM only
# (no CUDA / GL / TensorRT / neural). Consumes the dma-browser DMA-BUF fdpass
# socket (ipc_dmabuf_source, needs only HAVE_DRM=1) and streams H.264 (libx264,
# system ffmpeg) to a Janus RTP mountpoint.
#
# Build context = the avplumber repo root:
#   docker build -t avplumber-dmabuf-consumer -f dmabuf-demo/consumer/Dockerfile .
ARG AVP_FLAGS="HAVE_CUDA=0 HAVE_NVCC=0 HAVE_NVJPEG=0 HAVE_TENSORRT=0 NEURAL_NET_COMMON=0 NEURAL_NET_SPECIFIC=0 HAVE_NVOF=0 HAVE_OPENCV=0 HAVE_SCTE35=0 HAVE_GL=0 HAVE_DRM=1"

FROM fedora:42 AS builder
ARG AVP_FLAGS
RUN dnf -y install \
        "https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm" \
        "https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm" \
    && dnf -y install --setopt=install_weak_deps=False \
        gcc gcc-c++ binutils which make cmake git pkgconf-pkg-config \
        ffmpeg-devel \
        libdrm-devel \
        python3-devel \
        boost-devel libcurl-devel gnutls-devel openssl-devel zlib-devel \
    && dnf clean all
WORKDIR /build
COPY deps /build/deps
COPY Makefile /build/Makefile
# The copied deps/*/build dirs may carry a stale CMake cache with an empty
# CMAKE_AR (configured before binutils was present / on a different host), which
# makes static-lib archiving fail with `"" qc ...`. Verify the archiver exists
# and wipe the stale caches so cmake reconfigures fresh here.
RUN command -v ar \
    && find /build/deps -type f -name CMakeCache.txt -delete 2>/dev/null \
    ; find /build/deps -depth -type d -name CMakeFiles -exec rm -rf {} + 2>/dev/null; true
RUN make -j"$(nproc)" deps/cpr/build/lib/libcpr.a deps/avcpp/build/src/libavcpp.a ${AVP_FLAGS}
COPY src /build/src
COPY pyplumber /build/pyplumber
COPY generate_node_list /build/generate_node_list
COPY .git /build/.git
# Build against SYSTEM ffmpeg. Fedora's ffmpeg-devel puts headers under
# /usr/include/ffmpeg, but the Makefile hardcodes /usr/local/include for a custom
# ffmpeg build; add the system include path via pkg-config. This demo needs no
# ffmpeg patches (just libx264 / DRM hwdownload / rtp, all in system ffmpeg).
# Build the pyplumber python module only. (Building the `avplumber` binary in the
# same invocation fails to link: the generated node list then includes the Python
# node factories, which the plain binary doesn't compile in.)
RUN make -j"$(nproc)" python_module BUILD_TYPE=Release ${AVP_FLAGS} \
      CXXFLAGS+=" $(pkg-config --cflags libavformat libavcodec libavutil libavfilter libavdevice libswscale libswresample)"
RUN sofile="$(ls /build/_avplumber*.so)" && strings "$sofile" | grep -q ipc_dmabuf_source \
    && echo "OK: ipc_dmabuf_source registered in $sofile"

FROM fedora:42
RUN dnf -y install \
        "https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm" \
        "https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm" \
    && dnf -y install --setopt=install_weak_deps=False \
        bash ffmpeg libdrm boost-system boost-thread libcurl gnutls python3 \
    && dnf clean all
COPY --from=builder /build/_avplumber*.so /opt/avplumber/
COPY --from=builder /build/pyplumber /opt/avplumber/pyplumber
ENV PYTHONPATH=/opt/avplumber
WORKDIR /opt/avplumber
