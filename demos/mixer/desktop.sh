#!/usr/bin/env bash
# Run the portable mixer on a Linux GeForce desktop. With no arguments, build
# the services and start (or update) the show. Other Compose commands pass through.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
stack="$repo/docker-compose/mixer/deploy/l4/stack.sh"
export MIXER_INSTANCE_TYPE=geforce
export WITH_MXL="${WITH_MXL:-0}"
export MIXER_STACK_ENV_FILE=/dev/null
export MIXER_MEDIA_DIR="${MIXER_MEDIA_DIR:-$repo/media/desktop}"
export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-avplumber-mixer-desktop}"
export MIXER_IMAGE_PREFIX="${MIXER_IMAGE_PREFIX:-avplumber-mixer-desktop}"

fail() { echo "Desktop mixer: $*" >&2; exit 1; }

check_host() {
    [[ "$(uname -sm)" == "Linux x86_64" ]] || fail "the mixer image requires native x86-64 Linux"
    docker compose version
    [[ "$(docker info --format '{{if index .Runtimes "nvidia"}}yes{{end}}')" == yes ]] ||
        fail "enable the NVIDIA Container Toolkit runtime in the Docker daemon"
    local info pci driver name free render
    info="$(nvidia-smi -i 0 --query-gpu=pci.bus_id,driver_version,name,memory.free --format=csv,noheader,nounits)" ||
        fail "nvidia-smi could not query GPU 0"
    IFS=, read -r pci driver name free <<< "$info"
    driver=${driver// /}
    [[ ${driver%%.*} =~ ^[0-9]+$ ]] && (( ${driver%%.*} >= 615 )) ||
        fail "the Fedora/CUDA 13.4 image requires R615 or newer (found $driver)"
    # nvidia-smi uses an eight-digit PCI domain; Linux sysfs uses four digits.
    pci=${pci,,}
    local -a nodes=(/sys/bus/pci/devices/"${pci: -12}"/drm/renderD*)
    render="${WLR_RENDER_DRM_DEVICE:-/dev/dri/${nodes[0]##*/}}"
    [[ -c "$render" ]] || fail "no NVIDIA DRM render node for GPU 0 ($pci); check nvidia-drm modesetting"
    export WLR_RENDER_DRM_DEVICE="$render"
    # Janus skips loopback when gathering ICE candidates, even for a local viewer.
    if [[ -z "${JANUS_HOST_IP:-}" ]]; then
        command -v ip >/dev/null || fail "install iproute2 or set JANUS_HOST_IP to the desktop's LAN address"
        local interface
        interface=$(ip -4 route show default | awk '!dev {for (i=1; i<NF; ++i) if ($i=="dev") dev=$(i+1)} END {print dev}')
        if [[ -n "$interface" ]]; then
            JANUS_HOST_IP=$(ip -o -4 addr show dev "$interface" scope global | awk 'NR==1 {split($4,a,"/"); print a[1]}')
        fi
    fi
    [[ -n "${JANUS_HOST_IP:-}" && "$JANUS_HOST_IP" != 127.* ]] ||
        fail "set JANUS_HOST_IP to a non-loopback IPv4 address on the desktop for WebRTC previews"
    export JANUS_HOST_IP
    echo "GPU ${name# }, driver $driver, ${free# } MiB free, render node $render"
    echo "Janus WebRTC address: $JANUS_HOST_IP"
}

case "${1:-}" in
    check) check_host; "$stack" config --quiet ;;
    smoke) check_host; exec "$stack" "$@" ;;
    "")
        check_host
        "$stack" build
        "$stack" up -d --no-build
        echo "Open http://127.0.0.1:${MIXER_PUBLIC_PORT:-7681}/setup/ for media preparation and mixer status."
        ;;
    up|start|restart|run) check_host; exec "$stack" "$@" ;;
    *) exec "$stack" "$@" ;;
esac
