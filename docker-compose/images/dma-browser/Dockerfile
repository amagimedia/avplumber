FROM fedora:42 AS builder

RUN dnf -y install --setopt=install_weak_deps=False \
        "https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm" \
        "https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm" \
    && dnf -y install --setopt=install_weak_deps=False \
        gcc \
        gcc-c++ \
        make \
        nodejs \
        npm \
        python3 \
        python3-setuptools \
    && dnf clean all

WORKDIR /opt/dma-browser

COPY deps/dma-browser/package.json deps/dma-browser/package-lock.json deps/dma-browser/.npmrc ./
COPY deps/dma-browser/addons ./addons
RUN npm ci

COPY deps/dma-browser ./
RUN npm run build \
    && npm run rebuild:addons

FROM fedora:42

RUN dnf -y install --setopt=install_weak_deps=False \
        "https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm" \
        "https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm" \
    && dnf -y install --setopt=install_weak_deps=False \
        alsa-lib \
        at-spi2-atk \
        bash \
        ca-certificates \
        cairo \
        cups-libs \
        curl \
        dbus-libs \
        expat \
        gtk3 \
        libdrm \
        libwayland-client \
        libX11 \
        libxcb \
        libXcomposite \
        libXcursor \
        libXdamage \
        libXext \
        libXfixes \
        libXi \
        libXrandr \
        libXrender \
        libXScrnSaver \
        libxshmfence \
        libXtst \
        libxkbcommon \
        mesa-dri-drivers \
        mesa-libEGL \
        mesa-libgbm \
        mesa-libGL \
        nodejs \
        nspr \
        nss \
        pango \
        pciutils \
    && dnf clean all

COPY --from=builder /opt/dma-browser /opt/dma-browser

WORKDIR /opt/dma-browser

ENV DMA_BROWSER_REST_HOST=0.0.0.0
ENV DMA_BROWSER_REST_PORT=9009
ENV DMA_BROWSER_SOCKET_DIR=/tmp/dma-page
ENV DMA_BROWSER_FORCE_NVIDIA=1
ENV DMA_BROWSER_ALLOWED_DIMS=1080x1920,1920x1080,1280x720
ENV DMA_BROWSER_DMABUF_POOL_SIZE=11

HEALTHCHECK --interval=5s --timeout=5s --start-period=20s --retries=12 \
  CMD curl -fsS "http://127.0.0.1:${DMA_BROWSER_REST_PORT}/status" >/dev/null || exit 1

ENTRYPOINT ["bin/run.sh"]
