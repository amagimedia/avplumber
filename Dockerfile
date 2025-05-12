FROM fedora:41 AS common

RUN dnf -y install \
		https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm

RUN dnf -y install \
		https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm

FROM common AS builder

RUN dnf -y --allowerasing install \
    alsa-lib-devel \
    asio-devel \
    cmake \
    ffmpeg-devel \
    fontconfig-devel \
    freetype-devel \
    gcc \
    gcc-c++ \
    gcc-objc \
    which \
    glib2-devel \
    jansson-devel \
    json-devel \
    librist-devel \
    libcurl-devel \
    curl \
    libdatachannel-devel \
    libdrm-devel \
    libglvnd-devel \
    libqrcodegencpp-devel \
    libuuid-devel \
    libva-devel \
    libv4l-devel \
    libX11-devel \
    libXcomposite-devel \
    libXdamage \
    libXinerama-devel \
    libxkbcommon-devel \
    luajit-devel \
    make \
    mbedtls-devel \
    nv-codec-headers \
    oneVPL-devel \
    pciutils-devel \
    pipewire-devel \
    pulseaudio-libs-devel \
    rnnoise-devel \
    speexdsp-devel \
    srt-devel \
    swig \
    systemd-devel \
    uthash-devel \
    websocketpp-devel \
    x264-devel \
    pipewire-jack-audio-connection-kit-devel \
    openssl-devel libxcrypt-compat \
    automake libtool awscli

COPY deps /build/deps
COPY Makefile /build/
    
RUN make -C /build -j `nproc` deps/cpr/build/lib/libcpr.a deps/avcpp/build/src/libavcpp.a deps/cuda_loader/cuda_drvapi_dynlink.o
    
COPY src /build/src
COPY generate_node_list /build/
COPY .git /build/.git
    
RUN make -C /build -j `nproc`

FROM common

RUN dnf -y --allowerasing install \
    alsa-lib \
    ffmpeg \
    fontconfig\
    freetype \
    glib2 \
    jansson \
    json-static \
    librist \
    libcurl \
    libdatachannel \
    libdrm \
    libglvnd \
    libqrcodegencpp \
    libuuid \
    libva \
    libv4l \
    libX11 \
    libXcomposite \
    libXdamage \
    libXinerama \
    libxkbcommon \
    luajit \
    mbedtls \
    oneVPL \
    pciutils \
    pipewire \
    pulseaudio-libs \
    rnnoise \
    speexdsp \
    srt \
    systemd \
    uthash \
    vlc \
    libwayland-client libwayland-cursor libwayland-egl libwayland-server libxml2 glibc expat \
    x264 \
    mesa-va-drivers-freeworld \
    pipewire-jack-audio-connection-kit \
    openssl libxcrypt-compat rsync \
    boost-thread boost-system boost-iostreams boost-stacktrace boost

COPY --from=builder /build/avplumber /usr/local/bin/
COPY examples/complicated_transcoder.avplumber /usr/local/etc/complicated_transcoder.avplumber
ENTRYPOINT ["/usr/local/bin/avplumber"]
