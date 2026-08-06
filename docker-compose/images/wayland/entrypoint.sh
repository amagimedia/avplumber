#!/usr/bin/env bash
set -euo pipefail

runtime_dir="${XDG_RUNTIME_DIR:-/run/avp-wayland}"
wayland_display="${WAYLAND_DISPLAY:-wayland-1}"

mkdir -p "${runtime_dir}"
chown avp:avp "${runtime_dir}"
chmod 700 "${runtime_dir}"
rm -f "${runtime_dir}/${wayland_display}" "${runtime_dir}/${wayland_display}.lock"

exec runuser -u avp --preserve-environment -- /usr/bin/sway -d --unsupported-gpu -c /etc/sway/avp-headless.conf
