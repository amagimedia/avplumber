#!/usr/bin/env bash
# First-boot setup for a mixer GPU host (GCE startup script, runs as root).
# Plain Ubuntu 26.04 uses the requested, pinned NVIDIA driver; legacy
# accelerator images can retain their preinstalled driver. This adds:
# Docker, the NVIDIA Container Toolkit, the driver's video (NVDEC/NVENC/optical
# flow) and GL/EGL userspace, its GBM
# backend (libnvidia-allocator, in libnvidia-extra) and a modesetting nvidia-drm
# render node (demos/dmabuf-browser/docs/guide.md). Without any one of these,
# sway's gbm_bo_create fails and the browser sources have no buffers. The mixer
# image stays Fedora 44 / CUDA 13.4 and is built on the host (push-source.sh);
# the host provides only the driver, Docker with Compose and the runtime.
# Adapted from the sports-live-reframer GCP host setup.
# Idempotent: every boot re-verifies and rewrites /var/lib/avplumber-host/ready.
set -euo pipefail
exec > >(tee -a /var/log/avplumber-host-setup.log) 2>&1
state=/var/lib/avplumber-host
mkdir -p "$state"
rm -f "$state/ready"
meta() {
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null || true
}
requested_driver="$(meta avplumber-driver-version)"
export DEBIAN_FRONTEND=noninteractive

# Ubuntu's background package upgrades take every CPU for minutes: on
# 2026-10-01 a 60 fps run fell to 35 fps while unattended-upgrade was running.
# Stop the timers so a run is never interrupted; a host that needs patches is
# replaced, not upgraded in place. Disabling a timer does not stop a job it
# already started, so let that job finish: neither the reboot below nor the
# installs may cut dpkg off mid-transaction.
systemctl disable --now apt-daily.timer apt-daily-upgrade.timer unattended-upgrades.service \
  >/dev/null 2>&1 || true
for unit in apt-daily.service apt-daily-upgrade.service; do
  while [ "$(systemctl is-active "$unit" 2>/dev/null || true)" = activating ]; do
    echo "waiting for $unit to finish"
    sleep 5
  done
done
echo 'DPkg::Lock::Timeout "600";' > /etc/apt/apt.conf.d/90avplumber-lock-wait

# The guest kernel's CPU-vulnerability mitigations (PTI, legacy IBRS and buffer
# clears on N1 Skylake) tax every syscall and thread wake-up of the graph's
# ~200 threads (the reframer measured 2-3 fps more at 60 fps without them).
# This host runs one trusted workload, and the hypervisor's own isolation is
# unaffected, so switch them off. The first boot sets the kernel
# parameter and reboots once, before the driver build; later boots only verify
# it.
if ! grep -qw mitigations=off /proc/cmdline; then
  [ ! -f "$state/mitigations-off-requested" ] || {
    echo 'ERROR: mitigations=off is configured in GRUB but missing from the kernel command line'; exit 1;
  }
  # shellcheck disable=SC2016  # GRUB expands the variable when it sources this file.
  echo 'GRUB_CMDLINE_LINUX_DEFAULT="$GRUB_CMDLINE_LINUX_DEFAULT mitigations=off"' \
    > /etc/default/grub.d/99-avplumber-mitigations.cfg
  update-grub
  touch "$state/mitigations-off-requested"
  echo 'rebooting once to apply mitigations=off'
  systemctl reboot
  exit 0
fi

if [ -n "$requested_driver" ]; then
  # Use NVIDIA's version-lock package and Ubuntu DEBs, never Fedora RPMs.
  # Start from plain Ubuntu to avoid mixing Canonical's branch-suffixed
  # packages with NVIDIA's unversioned (590+) package names.
  [[ "$requested_driver" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
    echo "ERROR: invalid avplumber-driver-version: $requested_driver"; exit 2;
  }
  loaded_driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0 2>/dev/null || true)"
  if [ -n "$loaded_driver" ] && [ "$loaded_driver" != "$requested_driver" ]; then
    echo "ERROR: loaded driver $loaded_driver differs from requested $requested_driver; use a fresh plain-Ubuntu image"
    exit 1
  fi
  if [ ! -f "$state/pinned-driver" ]; then
    . /etc/os-release
    [ "$ID:$VERSION_ID" = ubuntu:26.04 ] || { echo 'ERROR: pinned driver requires Ubuntu 26.04'; exit 1; }
    apt-get update -q
    apt-get install -y -q ca-certificates curl gnupg "linux-headers-$(uname -r)"
    repo=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64
    curl -fsSL --retry 3 "$repo/cuda-keyring_1.1-1_all.deb" -o "$state/cuda-keyring.deb"
    echo "f7f474b5f6a4adf987aa587920df00e713285958ef6a913dda1945a544a3099e  $state/cuda-keyring.deb" | sha256sum -c -
    dpkg -i "$state/cuda-keyring.deb"
    apt-get update -q
    apt-get install -y -q "nvidia-driver-pinning-$requested_driver"
    # Use NVIDIA's open kernel module on fresh Ubuntu hosts. Include graphics,
    # GBM and video libraries, not just the compute-only driver.
    apt-get install -y -q nvidia-driver-open libnvidia-extra nvidia-modprobe
    printf '%s\n' "$requested_driver" > "$state/pinned-driver"
  fi
  [ "$(cat "$state/pinned-driver")" = "$requested_driver" ] || {
    echo 'ERROR: image driver pin differs from requested version'; exit 1;
  }
  modprobe nvidia
fi

# Plain Ubuntu images have no driver: install Canonical's signed, prebuilt
# 580-server modules and compute userspace (what the accelerator image ships).
if [ -z "$requested_driver" ] && ! nvidia-smi -L >/dev/null 2>&1; then
  apt-get update -q
  apt-get install -y -q ubuntu-drivers-common
  ubuntu-drivers install --gpgpu nvidia:580-server
  # That targets the newest kernel it pulls in; also cover the running one so
  # the driver loads now instead of after a reboot. --gpgpu omits nvidia-smi.
  apt-get install -y -q "linux-modules-nvidia-580-server-$(uname -r)" nvidia-utils-580-server
  modprobe nvidia
fi

driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0)"
[ -z "$requested_driver" ] || [ "$driver" = "$requested_driver" ] || {
  echo "ERROR: expected NVIDIA $requested_driver, loaded $driver"; exit 1;
}
major="${driver%%.*}"
if [ -f "$state/pinned-driver" ]; then
  driver_packages=(libnvidia-decode libnvidia-encode libnvidia-gl libnvidia-extra)
else
  # Match Canonical's preinstalled compute userspace (-server or not).
  flavour="$(dpkg-query -W -f '${Package}\n' "libnvidia-compute-${major}*" | head -1)"
  flavour="${flavour#libnvidia-compute-${major}}"
  driver_packages=("libnvidia-decode-${major}${flavour}" "libnvidia-encode-${major}${flavour}"
    "libnvidia-gl-${major}${flavour}" "libnvidia-extra-${major}${flavour}")
fi

if [ ! -f /etc/apt/sources.list.d/nvidia-container-toolkit.list ]; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
fi
apt-get update -q
# decode/encode: NVDEC, NVENC and the optical-flow library _avplumber links.
apt-get install -y -q docker.io docker-compose-v2 nvidia-container-toolkit "${driver_packages[@]}"
# Only the nvidia runtime injects the GBM backend; --gpus alone uses the runc hook.
nvidia-ctk runtime configure --runtime=docker --set-as-default
systemctl restart docker

# The browser renders through /dev/dri/renderD128; NVIDIA GBM needs modeset.
echo nvidia-drm > /etc/modules-load.d/nvidia-drm.conf
echo 'options nvidia-drm modeset=1' > /etc/modprobe.d/nvidia-drm-modeset.conf
if [ "$(cat /sys/module/nvidia_drm/parameters/modeset 2>/dev/null)" != Y ]; then
  modprobe -r nvidia_drm 2>/dev/null || true
fi
modprobe nvidia-drm

# Fail loudly rather than leave a host that runs the graph but not the browser sources.
test -e /usr/share/glvnd/egl_vendor.d/10_nvidia.json
test -e /usr/lib/x86_64-linux-gnu/gbm/nvidia-drm_gbm.so
test -e /usr/lib/x86_64-linux-gnu/libnvidia-opticalflow.so.1
test -e /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1
test "$(cat /sys/module/nvidia_drm/parameters/modeset)" = Y
test -c /dev/dri/renderD128
docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all --entrypoint nvidia-smi \
  ubuntu:24.04 -L
printf 'driver=%s mitigations=off\n' "$driver" > "$state/ready"
echo "mixer host ready: driver $driver"
