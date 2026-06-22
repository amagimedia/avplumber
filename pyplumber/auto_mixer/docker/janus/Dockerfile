FROM fedora:42 AS builder

ARG JANUS_REPO=https://github.com/meetecho/janus-gateway.git
ARG JANUS_REF=v1.4.1

RUN dnf -y install --setopt=install_weak_deps=False \
        autoconf \
        automake \
        ca-certificates \
        diffutils \
        findutils \
        gcc \
        git \
        glib2-devel \
        jansson-devel \
        libconfig-devel \
        libcurl-devel \
        libmicrohttpd-devel \
        libnice-devel \
        libsrtp-devel \
        libtool \
        make \
        openssl-devel \
        pkgconf-pkg-config \
        which \
        zlib-devel \
    && dnf clean all

WORKDIR /build
RUN git clone --depth 1 --branch "${JANUS_REF}" "${JANUS_REPO}" janus-gateway

WORKDIR /build/janus-gateway
RUN sh autogen.sh \
    && ./configure \
        --prefix=/opt/janus-avp \
        --disable-all-handlers \
        --disable-all-plugins \
        --enable-plugin-streaming \
        --disable-all-transports \
        --enable-rest \
        --disable-data-channels \
        --disable-rabbitmq \
        --disable-mqtt \
        --disable-nanomsg \
        --disable-unix-sockets \
        --disable-websockets \
    && make -j"$(nproc)" \
    && make install \
    && make configs

FROM fedora:42

RUN dnf -y install --setopt=install_weak_deps=False \
        bash \
        ca-certificates \
        curl \
        glib2 \
        jansson \
        libconfig \
        libcurl \
        libmicrohttpd \
        libnice \
        libsrtp \
        openssl-libs \
        zlib \
    && dnf clean all

COPY --from=builder /opt/janus-avp /opt/janus-avp
COPY janus/config/ /opt/janus-avp/etc/janus/
COPY janus/janus-entrypoint.sh /usr/local/bin/janus-entrypoint
RUN chmod +x /usr/local/bin/janus-entrypoint

ENTRYPOINT ["/usr/local/bin/janus-entrypoint"]
