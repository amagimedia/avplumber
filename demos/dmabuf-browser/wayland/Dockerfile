FROM fedora:42

# Headless wlroots/sway compositor + GL userspace loaders (glvnd) + mesa GBM
# frontend. No NVIDIA driver is baked in: the NVIDIA EGL/GL *core* (matching the
# host kernel driver) is provided by the NVIDIA container runtime at run time.
RUN dnf -y install --setopt=install_weak_deps=False \
        bash \
        libglvnd-egl \
        libglvnd-gles \
        libglvnd-glx \
        libglvnd-opengl \
        mesa-dri-drivers \
        mesa-libEGL \
        mesa-libGL \
        mesa-libGLES \
        mesa-libgbm \
        shadow-utils \
        socat \
        sway \
        util-linux \
        xorg-x11-server-Xwayland \
    && dnf clean all

# NVIDIA GL wiring for an Ubuntu/toolkit host:
#  - the EGL vendor ICD makes glvnd dispatch to the toolkit-mounted libEGL_nvidia
#  - nvidia-drm_gbm.so is the NVIDIA GBM backend; libnvidia-egl-gbm.so.1 itself
#    is bind-mounted from the host at run time (matches the host driver version).
RUN mkdir -p /usr/share/glvnd/egl_vendor.d /usr/lib64/gbm \
    && printf '%s\n' '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    && ln -sf /usr/lib64/libnvidia-egl-gbm.so.1 /usr/lib64/gbm/nvidia-drm_gbm.so

RUN useradd -m -u 1000 avp \
    && mkdir -p /run/avp-wayland \
    && chown avp:avp /run/avp-wayland

COPY wayland/sway-headless.conf /etc/sway/avp-headless.conf
COPY wayland/entrypoint.sh /usr/local/bin/avp-wayland-entrypoint
RUN chmod +x /usr/local/bin/avp-wayland-entrypoint

ENV XDG_RUNTIME_DIR=/run/avp-wayland
ENV WAYLAND_DISPLAY=wayland-1
ENV WLR_BACKENDS=headless
ENV WLR_LIBINPUT_NO_DEVICES=1
ENV WLR_NO_HARDWARE_CURSORS=1
ENV WLR_RENDERER=gles2
ENV WLR_RENDER_DRM_DEVICE=/dev/dri/renderD128
ENV GBM_BACKEND=nvidia-drm
ENV __GLX_VENDOR_LIBRARY_NAME=nvidia
ENV __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json

ENTRYPOINT ["/usr/local/bin/avp-wayland-entrypoint"]
