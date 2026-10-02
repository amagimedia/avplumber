#!/usr/bin/env bash
# Boot check for a mixer GPU host (GCE startup script, runs as root on every boot).
# The host starts from the avplumber-mixer-host image, which already carries the NVIDIA driver,
# Docker with Compose, the NVIDIA Container Toolkit as Docker's default runtime, the video, GL/EGL
# and GBM userspace and the mixer images. Nothing is installed or compiled here: a kernel module
# built at launch takes minutes. This only verifies the host and writes the ready file.
set -euo pipefail
exec > >(tee -a /var/log/avplumber-host-setup.log) 2>&1
state=/var/lib/avplumber-host
mkdir -p "$state"
rm -f "$state/ready"
meta() {
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null || true
}
expected="$(meta avplumber-driver-version)"

# Ubuntu's background package upgrades take every CPU for minutes and could replace the kernel
# the driver was built for; a host that needs patches is replaced from a new image instead.
systemctl disable --now apt-daily.timer apt-daily-upgrade.timer unattended-upgrades.service \
  >/dev/null 2>&1 || true

driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0 2>/dev/null || true)"
[ -n "$driver" ] || { echo 'ERROR: no NVIDIA driver loaded: start the host from the avplumber-mixer-host image'; exit 1; }
[ -z "$expected" ] || [ "$driver" = "$expected" ] || {
  echo "ERROR: driver $driver, expected $expected: use the matching avplumber-mixer-host image"; exit 1;
}

# The graph runs about 200 threads; CPU-vulnerability mitigations tax every syscall and thread
# wake-up (the reframer measured 2-3 fps more at 60 fps without them). The image boots with
# mitigations=off; this only refuses a host that does not.
grep -qw mitigations=off /proc/cmdline || { echo 'ERROR: kernel booted without mitigations=off'; exit 1; }

# The browser sources render through /dev/dri/renderD128; NVIDIA GBM needs modeset.
[ "$(cat /sys/module/nvidia_drm/parameters/modeset 2>/dev/null)" = Y ] || modprobe nvidia-drm modeset=1

# Fail loudly rather than leave a host that runs the graph but not the browser sources.
test -e /usr/share/glvnd/egl_vendor.d/10_nvidia.json
test -e /usr/lib/x86_64-linux-gnu/gbm/nvidia-drm_gbm.so
test -e /usr/lib/x86_64-linux-gnu/libnvidia-opticalflow.so.1
test -e /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1
test "$(cat /sys/module/nvidia_drm/parameters/modeset)" = Y
test -c /dev/dri/renderD128
docker info --format '{{.DefaultRuntime}}' | grep -qx nvidia
docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all --entrypoint nvidia-smi ubuntu:24.04 -L
printf 'driver=%s mitigations=off\n' "$driver" > "$state/ready"
echo "mixer host ready: driver $driver"
