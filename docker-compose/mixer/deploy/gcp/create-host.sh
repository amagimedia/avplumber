#!/usr/bin/env bash
# Create a mixer GPU host from a prepared image, or install its prerequisites
# on plain Ubuntu 26.04 with HOST_SETUP_MODE=install. Wait for boot verification.
#   PROJECT=<project> docker-compose/mixer/deploy/gcp/create-host.sh <name>
# Defaults: one L4 on a g2-standard-16 (16 vCPU like the T4 reference host), a
# 256 GB disk. MACHINE_TYPE=n1-standard-16 GPU=nvidia-tesla-t4 gives the T4 host.
# ZONES are tried in order, since single zones regularly run out of GPUs.
set -euo pipefail
name="${1:?usage: PROJECT=<project> create-host.sh <name>}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="${PROJECT:?set PROJECT to the GCP project}"
ZONES="${ZONES:-europe-west2-b europe-west2-a europe-west4-c europe-west4-b europe-west4-a europe-west1-b europe-west1-c europe-west3-a europe-west3-b}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-16}"
GPU="${GPU:-nvidia-l4}"
DISK_GB="${DISK_GB:-256}"
HOST_SETUP_MODE="${HOST_SETUP_MODE:-verify}"
case "$HOST_SETUP_MODE" in
  verify)
    IMAGE_FAMILY="${IMAGE_FAMILY:-avplumber-mixer-host}"
    IMAGE_PROJECT="${IMAGE_PROJECT:-$PROJECT}"
    ;;
  install)
    IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-2604-lts-amd64}"
    IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-cloud}"
    ;;
  *) echo 'ERROR: HOST_SETUP_MODE must be verify or install' >&2; exit 2 ;;
esac
IMAGE_NAME="${IMAGE_NAME:-}"
# The driver the image carries; the Fedora 44 / CUDA 13.4 mixer image needs R615 or newer.
NVIDIA_DRIVER_VERSION="${NVIDIA_DRIVER_VERSION-615.71.09}"
DISABLE_MITIGATIONS="${DISABLE_MITIGATIONS:-0}"
DISABLE_AUTO_UPGRADES="${DISABLE_AUTO_UPGRADES:-0}"
for flag in "$DISABLE_MITIGATIONS" "$DISABLE_AUTO_UPGRADES"; do
  [[ "$flag" = 0 || "$flag" = 1 ]] || { echo 'ERROR: DISABLE_* flags must be 0 or 1' >&2; exit 2; }
done
[[ -z "$NVIDIA_DRIVER_VERSION" || "$NVIDIA_DRIVER_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || {
  echo 'ERROR: invalid NVIDIA_DRIVER_VERSION' >&2; exit 2;
}
[[ "$HOST_SETUP_MODE" != install || -n "$NVIDIA_DRIVER_VERSION" ]] || {
  echo 'ERROR: install mode requires NVIDIA_DRIVER_VERSION' >&2; exit 2;
}
[[ -z "$NVIDIA_DRIVER_VERSION" ]] || [ "${NVIDIA_DRIVER_VERSION%%.*}" -ge 615 ] || {
  echo 'ERROR: this stack requires NVIDIA driver R615 or newer' >&2; exit 2;
}
image_args=(--image-family "$IMAGE_FAMILY")
[ -z "$IMAGE_NAME" ] || image_args=(--image "$IMAGE_NAME")

# A second create under the same name would land in the next zone and leave two hosts.
existing="$(gcloud compute instances list --project "$PROJECT" --filter="name=($name)" --format='value(zone.basename())')"
[ -z "$existing" ] || { echo "ERROR: $name already exists in $existing; delete it or choose another name" >&2; exit 2; }

# Accelerator-optimized types (g2-*) bring their GPU; N1 needs it attached.
gpu_args=()
case "$MACHINE_TYPE" in n1-*) gpu_args=(--accelerator "type=${GPU},count=1");; esac
zone=""
for candidate in $ZONES; do
  if gcloud compute instances create "$name" --project "$PROJECT" --zone "$candidate" \
      --machine-type "$MACHINE_TYPE" ${gpu_args[@]+"${gpu_args[@]}"} \
      --maintenance-policy TERMINATE --subnet default \
      "${image_args[@]}" --image-project "$IMAGE_PROJECT" \
      --boot-disk-size "${DISK_GB}GB" --boot-disk-type pd-balanced \
      --labels app=avplumber-mixer --scopes cloud-platform \
      --metadata "avplumber-driver-version=${NVIDIA_DRIVER_VERSION},avplumber-host-mode=${HOST_SETUP_MODE},avplumber-disable-mitigations=${DISABLE_MITIGATIONS},avplumber-disable-auto-upgrades=${DISABLE_AUTO_UPGRADES}" \
      --metadata-from-file "startup-script=${here}/host-setup.sh"; then
    zone="$candidate"
    break
  fi
done
[ -n "$zone" ] || { echo "ERROR: no zone in '$ZONES' could create $name" >&2; exit 1; }
echo "created $name in $zone; delete with: gcloud compute instances delete $name --project $PROJECT --zone $zone"

echo "waiting for host-setup (log: /var/log/avplumber-host-setup.log)"
for _ in $(seq 60); do
  if gcloud compute ssh "$name" --project "$PROJECT" --zone "$zone" --quiet \
      --command 'test "$(cat /var/lib/avplumber-host/ready-boot-id)" = "$(cat /proc/sys/kernel/random/boot_id)" && cat /var/lib/avplumber-host/ready' 2>/dev/null; then
    exit 0
  fi
  sleep 20
done
echo "ERROR: host-setup did not finish; $name is still running in $zone and billed." >&2
echo "Read /var/log/avplumber-host-setup.log on it, or delete it:" >&2
echo "  gcloud compute instances delete $name --project $PROJECT --zone $zone" >&2
exit 1
