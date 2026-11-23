#!/usr/bin/env bash
# Headless sway compositor for the DMA-BUF browser demo.
set -euo pipefail

runtime_dir="${XDG_RUNTIME_DIR:-/run/avp-wayland}"
wayland_display="${WAYLAND_DISPLAY:-wayland-1}"
render_node="${WLR_RENDER_DRM_DEVICE:-/dev/dri/renderD128}"

# The mounted DRM nodes keep the host's numeric group IDs, which can differ
# from the container's render/video groups. Give sway those supplementary groups.
device_groups=$({
    id -G avp | tr ' ' '\n'
    for device in "$render_node" /dev/dri/card*; do
        if [[ -c "$device" ]]; then stat -c '%g' "$device"; fi
    done
} | sort -un | paste -sd, -)

mkdir -p "${runtime_dir}"
chown avp:avp "${runtime_dir}"
chmod 700 "${runtime_dir}"
rm -f "${runtime_dir}/${wayland_display}" "${runtime_dir}/${wayland_display}.lock"

export HOME=/home/avp USER=avp LOGNAME=avp
exec setpriv --reuid=avp --regid=avp --groups="$device_groups" -- \
  /usr/bin/sway -d --unsupported-gpu -c /etc/sway/avp-headless.conf
