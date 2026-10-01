FROM fedora:44

# libglvnd-gles obsoletes mesa-libGLES: Mesa's GLES goes through glvnd's libGLESv2.
RUN dnf -y install --setopt=install_weak_deps=False \
        bash \
        libglvnd-gles \
        mesa-dri-drivers \
        mesa-libEGL \
        mesa-libGL \
        shadow-utils \
        socat \
        sway \
        util-linux \
        xorg-x11-server-Xwayland \
    && dnf clean all

RUN useradd -m -u 1000 avp \
    && mkdir -p /run/avp-wayland \
    && chown avp:avp /run/avp-wayland

COPY sway-headless.conf /etc/sway/avp-headless.conf
COPY entrypoint.sh /usr/local/bin/avp-wayland-entrypoint
RUN chmod +x /usr/local/bin/avp-wayland-entrypoint

ENV XDG_RUNTIME_DIR=/run/avp-wayland
ENV WAYLAND_DISPLAY=wayland-1
ENV WLR_BACKENDS=headless
ENV WLR_LIBINPUT_NO_DEVICES=1
ENV WLR_NO_HARDWARE_CURSORS=1
ENV WLR_RENDERER=gles2
ENV WLR_RENDER_DRM_DEVICE=/dev/dri/renderD128

ENTRYPOINT ["/usr/local/bin/avp-wayland-entrypoint"]
