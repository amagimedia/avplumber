#!/usr/bin/env bash
# GCE startup script, run as root before application workloads.
# Default mode verifies a prepared host. avplumber-host-mode=install installs
# pinned NVIDIA drivers, Docker/Compose and the NVIDIA Container Toolkit on
# fresh x86_64 Ubuntu 26.04, then reboots once. Driver installation may use DKMS.
# Keep this startup script registered so the final boot verifies readiness.
# CUDA/TensorRT SDKs and application images are supplied by the container stack.
set -euo pipefail
exec > >(tee -a /var/log/avplumber-host-setup.log) 2>&1
state=/var/lib/avplumber-host
mkdir -p "$state"
rm -f "$state/ready" "$state/ready-boot-id"
meta() {
  curl -fsS -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null || true
}
expected="$(meta avplumber-driver-version)"
mode="$(meta avplumber-host-mode)"
mode="${mode:-verify}"
disable_mitigations="$(meta avplumber-disable-mitigations)"
disable_upgrades="$(meta avplumber-disable-auto-upgrades)"
disable_mitigations="${disable_mitigations:-0}"
disable_upgrades="${disable_upgrades:-0}"
[[ "$mode" = verify || "$mode" = install ]] || { echo 'ERROR: invalid host mode'; exit 2; }
for flag in "$disable_mitigations" "$disable_upgrades"; do
  [[ "$flag" = 0 || "$flag" = 1 ]] || { echo 'ERROR: invalid performance flag'; exit 2; }
done
[[ -z "$expected" || "$expected" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
  echo 'ERROR: invalid driver version'; exit 2;
}
[ -z "$expected" ] || [ "${expected%%.*}" -ge 615 ] || {
  echo 'ERROR: this stack requires NVIDIA driver R615 or newer'; exit 2;
}

# Leave Ubuntu's security defaults intact unless the operator opts into the
# dedicated-host benchmark settings. Do not terminate an active apt transaction.
if [ "$disable_upgrades" = 1 ]; then
  systemctl disable --now apt-daily.timer apt-daily-upgrade.timer >/dev/null 2>&1 || true
  systemctl disable unattended-upgrades.service >/dev/null 2>&1 || true
fi

reboot_needed=0
if [ "$mode" = install ]; then
  . /etc/os-release
  [ "$ID:$VERSION_ID:$(uname -m)" = ubuntu:26.04:x86_64 ] || {
    echo 'ERROR: install mode requires fresh x86_64 Ubuntu 26.04'; exit 1;
  }
  [ -n "$expected" ] || { echo 'ERROR: install mode requires avplumber-driver-version'; exit 2; }
  loaded="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0 2>/dev/null || true)"
  [ -z "$loaded" ] || [ "$loaded" = "$expected" ] || {
    echo "ERROR: driver $loaded is already loaded; use a fresh plain-Ubuntu image for $expected"; exit 1;
  }
  if [ ! -f "$state/installed-driver" ]; then
    export DEBIAN_FRONTEND=noninteractive
    apt() { apt-get -o DPkg::Lock::Timeout=600 "$@"; }
    apt update -q
    apt install -y -q ca-certificates curl gnupg "linux-headers-$(uname -r)"
    repo=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64
    curl -fsSL --retry 3 "$repo/cuda-keyring_1.1-1_all.deb" -o "$state/cuda-keyring.deb"
    echo "f7f474b5f6a4adf987aa587920df00e713285958ef6a913dda1945a544a3099e  $state/cuda-keyring.deb" | sha256sum -c -
    apt install -y -q "$state/cuda-keyring.deb"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt update -q
    apt install -y -q "nvidia-driver-pinning-$expected"
    # libnvidia-extra supplies the GBM backend; EGL support alone is insufficient.
    apt install -y -q nvidia-driver-open nvidia-modprobe libnvidia-decode \
      libnvidia-encode libnvidia-gl libnvidia-extra docker.io docker-compose-v2 git rsync \
      nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker --set-as-default
    systemctl enable docker
    systemctl restart docker
    echo nvidia-drm > /etc/modules-load.d/nvidia-drm.conf
    echo 'options nvidia-drm modeset=1' > /etc/modprobe.d/nvidia-drm-modeset.conf
    update-initramfs -u
    printf '%s\n' "$expected" > "$state/installed-driver"
    reboot_needed=1
  fi
  [ "$(cat "$state/installed-driver")" = "$expected" ] || {
    echo 'ERROR: installed driver pin differs from requested version'; exit 1;
  }
fi

if [ "$disable_mitigations" = 1 ] && ! grep -qw mitigations=off /proc/cmdline; then
  [ "$mode" = install ] && [ ! -f "$state/mitigations-off-requested" ] || {
    echo 'ERROR: requested mitigations=off is absent from the kernel command line'; exit 1;
  }
  # shellcheck disable=SC2016  # GRUB expands this variable when reading its configuration.
  echo 'GRUB_CMDLINE_LINUX_DEFAULT="$GRUB_CMDLINE_LINUX_DEFAULT mitigations=off"' \
    > /etc/default/grub.d/99-avplumber-mitigations.cfg
  update-grub
  touch "$state/mitigations-off-requested"
  reboot_needed=1
fi
if [ "$reboot_needed" = 1 ]; then
  echo 'rebooting to verify the installed driver and boot settings'
  systemctl reboot
  exit 0
fi

driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0 2>/dev/null || true)"
[ -n "$driver" ] || { echo 'ERROR: no NVIDIA driver loaded; check driver installation and the boot kernel'; exit 1; }
[ "${driver%%.*}" -ge 615 ] || { echo 'ERROR: the CUDA 13.4 stack requires driver R615 or newer'; exit 1; }
[ -z "$expected" ] || [ "$driver" = "$expected" ] || {
  echo "ERROR: driver $driver, expected $expected: use a matching host image"; exit 1;
}

# The browser sources render through /dev/dri/renderD128; NVIDIA GBM needs modeset.
[ "$(cat /sys/module/nvidia_drm/parameters/modeset 2>/dev/null)" = Y ] || modprobe nvidia-drm modeset=1

# Fail loudly rather than leave a host that runs the graph but not the browser sources.
test -e /usr/share/glvnd/egl_vendor.d/10_nvidia.json
test -e /usr/lib/x86_64-linux-gnu/gbm/nvidia-drm_gbm.so
test -e /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1
test -e /usr/lib/x86_64-linux-gnu/libnvidia-opticalflow.so.1
test -e /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1
test "$(cat /sys/module/nvidia_drm/parameters/modeset)" = Y
test -c /dev/dri/renderD128
docker info --format '{{.DefaultRuntime}}' | grep -qx nvidia
docker compose version
docker run --rm --gpus all --device /dev/dri -e NVIDIA_DRIVER_CAPABILITIES=all \
  --entrypoint sh ubuntu:24.04 -ec '
    nvidia-smi -L
    test -c /dev/dri/renderD128
    test -e /usr/lib/x86_64-linux-gnu/gbm/nvidia-drm_gbm.so
    test -e /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1
    test -e /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1
  '
dpkg-query -W -f '${Package}\t${Version}\n' \
  | awk '$1 ~ /^(docker|containerd|runc|nvidia-container|libnvidia-container)/' > "$state/packages.txt"
printf 'driver=%s mode=%s\n' "$driver" "$mode" > "$state/ready"
cat /proc/sys/kernel/random/boot_id > "$state/ready-boot-id"
echo "mixer host ready: driver $driver"
