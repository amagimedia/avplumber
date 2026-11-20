#!/usr/bin/env bash
# Prepare discovery and the SDK on the Ubuntu L4 host.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../../../.." && pwd)
packages=(avahi-daemon curl ca-certificates xdg-dbus-proxy)
if ! docker buildx version >/dev/null 2>&1; then
    packages+=(docker-buildx)
fi
apt-get -o DPkg::Lock::Timeout=120 update -q
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${packages[@]}"
systemctl enable --now avahi-daemon
bash "$repo/deps/ffmpeg/install-ndi-sdk.sh"
echo /opt/ndi/6.4/lib > /etc/ld.so.conf.d/avplumber-ndi.conf
ldconfig
install -m 0644 "$repo/docker-compose/mixer/deploy/l4/avplumber-ndi-discovery.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now avplumber-ndi-discovery.service
