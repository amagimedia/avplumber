FROM fedora:44 AS builder

RUN dnf -y install --setopt=install_weak_deps=False \
        "https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm" \
        "https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm" \
    && dnf -y install --setopt=install_weak_deps=False \
        curl \
        gcc \
        gcc-c++ \
        make \
        nodejs \
        npm \
        python3 \
        python3-setuptools \
        unzip \
    && dnf clean all

WORKDIR /opt/dma-browser

COPY deps/dma-browser/package.json deps/dma-browser/package-lock.json deps/dma-browser/.npmrc ./
COPY deps/dma-browser/addons ./addons
# Electron 42+ no longer downloads its binary on install; fetch it explicitly.
RUN npm ci && node node_modules/electron/install.js

# Electron built with the zero-copy export patches (deps/dma-browser/patches/zero-copy)
# replaces stock Electron: Chromium renders into a fixed, preallocated buffer pool and
# hands each frame over with its acquire fence. The manifest's version 2 certifies that
# this binary sizes the pool from AvpZeroCopyOffscreenExport:buffers/N. An empty
# ELECTRON_ARCHIVE_URL keeps stock Electron and copy capture.
ARG ELECTRON_ARCHIVE_URL=https://github.com/amagimedia/avplumber/releases/download/dma-browser-electron-v44.5.1-avp-zero-copy-1/electron-v44.5.1-linux-x64.zip
ARG ELECTRON_ARCHIVE_SHA256=06491fff01fd4ae1c59d55b62efad7e32a91512d36488fefcb135fd9be5464bf
RUN if [ -n "${ELECTRON_ARCHIVE_URL}" ]; then \
        curl -fsSL --retry 3 -o /tmp/electron.zip "${ELECTRON_ARCHIVE_URL}" \
        && echo "${ELECTRON_ARCHIVE_SHA256}  /tmp/electron.zip" | sha256sum -c - \
        && rm -rf node_modules/electron/dist \
        && unzip -q /tmp/electron.zip -d node_modules/electron/dist \
        && rm /tmp/electron.zip \
        && test "$(cat node_modules/electron/dist/version)" = "$(node -p 'require("electron/package.json").version')" \
        && printf '{"version":2,"size":11}\n' > node_modules/electron/dist/resources/avp-capture-pool.json; \
    fi

COPY deps/dma-browser ./
RUN npm run build \
    && npm run rebuild:addons \
    && npm run rebuild:shim

FROM fedora:44

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
# The NVIDIA runtime preserves host library paths, including on Ubuntu hosts.
ENV GBM_BACKEND=nvidia-drm
ENV GBM_BACKENDS_PATH=/usr/lib/x86_64-linux-gnu/gbm:/usr/lib64/gbm
ENV LD_PRELOAD=/opt/dma-browser/native/gbm-linear-shim/libgbm_linear_shim.so
ENV GBM_LINEAR_SHIM=1
ENV GBM_LINEAR_SHIM_ADD_SCANOUT=1
ENV DMA_BROWSER_ALLOWED_DIMS=1080x1920,1920x1080,1280x720
ENV DMA_BROWSER_DMABUF_POOL_SIZE=9
# The patched Electron's fixed export pool; the mixer registers all nine buffers per window
# before going on air. Stock Electron (empty ELECTRON_ARCHIVE_URL) ignores both features.
ENV DMA_BROWSER_CHROMIUM_EXTRA_FEATURES=AvpZeroCopyOffscreenExport:buffers/9,RenderableMappableSharedImageForceScanout

HEALTHCHECK --interval=5s --timeout=5s --start-period=20s --retries=12 \
  CMD curl -fsS "http://127.0.0.1:${DMA_BROWSER_REST_PORT}/status" >/dev/null || exit 1

ENTRYPOINT ["bin/run.sh"]
