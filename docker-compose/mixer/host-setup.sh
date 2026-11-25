#!/usr/bin/env bash
# Prepare or check a Linux x86-64 host for the mixer stack: any NVIDIA GPU from Turing on, a
# desktop or a server, any cloud.
#
#   docker-compose/mixer/host-setup.sh [check]          read-only: each missing piece and its fix
#   sudo docker-compose/mixer/host-setup.sh install     Ubuntu 24.04 or 26.04: Docker, Compose,
#        [--driver <version>]                             the NVIDIA Container Toolkit, DRM modeset;
#                                                         with --driver also the pinned NVIDIA driver
#                                                         (Ubuntu 26.04, NVIDIA's CUDA repository)
#
# The GCE startup script (deploy/gcp/host-setup.sh) runs the same steps on a VM without this
# checkout, so it carries its own copy.
set -euo pipefail

readonly min_driver=615   # the image's CUDA 13.4 and the Video Codec SDK 13.1 headers

failures=0
ok() { printf '  ok    %s\n' "$*"; }
bad() { printf '  FIX   %s\n' "$*"; failures=$((failures + 1)); }

# A driver library the container toolkit mounts into the containers, wherever the distro puts it.
have_lib() { ldconfig -p 2>/dev/null | grep -q "/$1\$"; }
gbm_backend() {
    local dir
    for dir in /usr/lib/x86_64-linux-gnu/gbm /usr/lib64/gbm /usr/lib/gbm; do
        [[ -e $dir/nvidia-drm_gbm.so ]] && return 0
    done
    return 1
}

check() {
    echo "Mixer host check"
    [[ "$(uname -sm)" == "Linux x86_64" ]] && ok "Linux x86-64" || bad "the mixer image needs native x86-64 Linux"

    local driver="" name=""
    if command -v nvidia-smi >/dev/null && IFS=, read -r name driver < <(
            nvidia-smi -i 0 --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null); then
        driver=${driver// /}
        if [[ ${driver%%.*} =~ ^[0-9]+$ ]] && (( ${driver%%.*} >= min_driver )); then
            ok "NVIDIA driver $driver on ${name}"
        else
            bad "NVIDIA driver R$min_driver or newer is needed (found $driver): install it from NVIDIA's" \
                "CUDA repository or your distribution, or run: sudo $0 install --driver <version>"
        fi
    else
        bad "no NVIDIA driver answers nvidia-smi: install driver R$min_driver or newer"
    fi

    have_lib libnvcuvid.so.1 && ok "NVDEC library" ||
        bad "libnvcuvid.so.1 missing: install the driver's video decode package (Ubuntu: libnvidia-decode)"
    have_lib libnvidia-encode.so.1 && ok "NVENC library" ||
        bad "libnvidia-encode.so.1 missing: install the driver's encode package (Ubuntu: libnvidia-encode)"
    [[ -e /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]] && ok "NVIDIA EGL" ||
        bad "NVIDIA EGL vendor file missing: install the driver's GL libraries (Ubuntu: libnvidia-gl), not a compute-only driver"
    gbm_backend && ok "NVIDIA GBM backend" ||
        bad "nvidia-drm_gbm.so missing: browser capture needs it (Ubuntu: libnvidia-extra)"
    if [[ "$(cat /sys/module/nvidia_drm/parameters/modeset 2>/dev/null)" == Y ]]; then
        ok "nvidia-drm modeset"
    else
        bad "nvidia-drm modeset is off: sudo $0 install sets it (then reboot), or add nvidia-drm.modeset=1 to the kernel command line"
    fi
    compgen -G '/dev/dri/renderD*' >/dev/null && ok "DRM render node" ||
        bad "no /dev/dri/renderD* node: load nvidia-drm with modeset=1"

    if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
        ok "Docker"
        docker compose version >/dev/null 2>&1 && ok "Docker Compose v2" ||
            bad "Docker Compose v2 missing (Ubuntu: docker-compose-v2)"
        # The runtime names, not `index .Runtimes "nvidia"`: a missing key gives an empty struct, which
        # a template's `if` takes as true.
        docker info --format '{{range $name, $runtime := .Runtimes}}{{$name}} {{end}}' 2>/dev/null | grep -qw nvidia &&
            ok "NVIDIA Container Toolkit runtime" ||
            bad "Docker has no nvidia runtime: install nvidia-container-toolkit and run" \
                "'sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker'"
    else
        bad "Docker is not running or not reachable for $(id -un): install docker and add the user to the docker group"
    fi

    if (( failures )); then
        echo "$failures item(s) to fix before the mixer can run."
        return 1
    fi
    echo "Host ready."
}

install() {
    local driver_version=""
    while (( $# )); do
        case "$1" in
            --driver) driver_version="${2:?--driver needs a version such as 615.71.09}"; shift 2 ;;
            *) echo "unknown option $1" >&2; exit 2 ;;
        esac
    done
    (( EUID == 0 )) || { echo "install needs root: sudo $0 install" >&2; exit 2; }
    . /etc/os-release
    [[ "$ID:$(uname -m)" == ubuntu:x86_64 && ( $VERSION_ID == 24.04 || $VERSION_ID == 26.04 ) ]] || {
        echo "install supports Ubuntu 24.04 and 26.04 on x86-64; on other systems install the pieces" \
             "'$0 check' lists by hand" >&2; exit 2; }
    export DEBIAN_FRONTEND=noninteractive
    apt() { apt-get -o DPkg::Lock::Timeout=600 "$@"; }
    apt update -q
    apt install -y -q ca-certificates curl gnupg docker.io docker-compose-v2

    if [[ -n $driver_version ]]; then
        [[ $driver_version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ && ${driver_version%%.*} -ge $min_driver ]] || {
            echo "--driver must be R$min_driver or newer, such as 615.71.09" >&2; exit 2; }
        [[ $VERSION_ID == 26.04 ]] || {
            echo "--driver is supported on Ubuntu 26.04; on $VERSION_ID install R$min_driver+ from NVIDIA's" \
                 "CUDA repository (https://developer.nvidia.com/cuda-downloads) and rerun without --driver" >&2; exit 2; }
        local loaded
        loaded="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0 2>/dev/null || true)"
        [[ -z $loaded || $loaded == "$driver_version" ]] || {
            echo "driver $loaded is loaded; remove it before installing $driver_version" >&2; exit 1; }
        local state=/var/lib/avplumber-host repo=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64
        mkdir -p "$state"
        apt install -y -q "linux-headers-$(uname -r)"
        curl -fsSL --retry 3 "$repo/cuda-keyring_1.1-1_all.deb" -o "$state/cuda-keyring.deb"
        echo "f7f474b5f6a4adf987aa587920df00e713285958ef6a913dda1945a544a3099e  $state/cuda-keyring.deb" | sha256sum -c -
        apt install -y -q "$state/cuda-keyring.deb"
        apt update -q
        apt install -y -q "nvidia-driver-pinning-$driver_version"
        # libnvidia-extra supplies the GBM backend; EGL support alone is insufficient.
        apt install -y -q nvidia-driver-open nvidia-modprobe libnvidia-decode libnvidia-encode \
            libnvidia-gl libnvidia-extra
    fi

    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |
        gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list |
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt update -q
    apt install -y -q nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl enable docker
    systemctl restart docker
    [[ -z ${SUDO_USER:-} ]] || usermod -aG docker "$SUDO_USER"

    echo nvidia-drm > /etc/modules-load.d/nvidia-drm.conf
    echo 'options nvidia-drm modeset=1' > /etc/modprobe.d/nvidia-drm-modeset.conf
    update-initramfs -u
    echo "Installed. Reboot, then run '$0 check' (as your user) before starting the mixer."
}

case "${1:-check}" in
    check) check ;;
    install) shift; install "$@" ;;
    *) echo "usage: $0 [check | install [--driver <version>]]" >&2; exit 2 ;;
esac
