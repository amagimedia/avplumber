#!/usr/bin/env bash
# Headless sway compositor for the DMA-BUF browser demo.
set -euo pipefail

runtime_dir="${XDG_RUNTIME_DIR:-/run/avp-wayland}"
wayland_display="${WAYLAND_DISPLAY:-wayland-1}"
render_node="${WLR_RENDER_DRM_DEVICE:-/dev/dri/renderD128}"

# The DRM render node is root:render 0660 on the host; sway runs as the
# unprivileged 'avp' user, and a non-root user doesn't get CAP_DAC_OVERRIDE
# even in a privileged container. Loosen the node perms (we start as root here,
# before dropping to avp) so the GLES2/GBM renderer can open it.
chmod o+rw "$render_node" /dev/dri/card* 2>/dev/null || true

mkdir -p "${runtime_dir}"
chown avp:avp "${runtime_dir}"
chmod 700 "${runtime_dir}"
rm -f "${runtime_dir}/${wayland_display}" "${runtime_dir}/${wayland_display}.lock"

exec runuser -u avp --preserve-environment -- \
  /usr/bin/sway -d --unsupported-gpu -c /etc/sway/avp-headless.conf
