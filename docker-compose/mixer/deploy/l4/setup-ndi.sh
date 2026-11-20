#!/usr/bin/env bash
# Prepare discovery and the SDK on the Ubuntu L4 host.
set -euo pipefail
repo=$(cd "$(dirname "$0")/../../../.." && pwd)
apt-get -o DPkg::Lock::Timeout=120 update -q
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends avahi-daemon curl ca-certificates
systemctl enable --now avahi-daemon
bash "$repo/deps/ffmpeg/install-ndi-sdk.sh"
