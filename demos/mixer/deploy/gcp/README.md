# GCP GPU hosts for the mixer

These scripts prepare or verify the **host**. Deploy the portable
[L4 container stack](../l4/README.md) separately; the host does not need the
CUDA/TensorRT SDKs or a copy of another VM's Docker state.

## Fresh Ubuntu 26.04

Use `HOST_SETUP_MODE=install` for a fresh x86_64 Ubuntu 26.04 GPU VM. It selects
the public `ubuntu-2604-lts-amd64` image family in `ubuntu-os-cloud`, installs
the host prerequisites and reboots once before reporting readiness:

```sh
PROJECT=<project> ZONES='<zone> <fallback-zone>' HOST_SETUP_MODE=install \
  demos/mixer/deploy/gcp/create-host.sh <name>
```

Defaults are one L4 on `g2-standard-16` and a 256 GB balanced persistent boot
disk. Override `ZONES` for the destination region; zones are tried in order.
`MACHINE_TYPE=n1-standard-16 GPU=nvidia-tesla-t4` selects a T4 host, whose mixer
capacity must be measured separately. Use `IMAGE_NAME=<image>` with
`IMAGE_PROJECT=<project>` to select an immutable image instead of a family.

The installer requires GCE metadata and outbound Ubuntu/NVIDIA APT and HTTPS
access. It uses NVIDIA's official Ubuntu repository and the checksum-verified
`cuda-keyring_1.1-1_all.deb`, then installs:

- `nvidia-driver-pinning-615.71.09` and `nvidia-driver-open`, with matching NVDEC,
  NVENC, optical-flow, EGL/GL and GBM libraries. `NVIDIA_DRIVER_VERSION` changes
  the requested exact version; this CUDA 13.4 stack requires R615 or newer.
- Ubuntu's `docker.io` and `docker-compose-v2`, plus NVIDIA Container Toolkit.
  Docker's default runtime is set to `nvidia` so the GBM backend is injected
  alongside the GPU libraries. Git and rsync support source deployment.
- Persistent `nvidia-drm modeset=1` configuration, including the initramfs,
  for the browser's `/dev/dri/renderD128` device.

Run this before application workloads. It restarts Docker and reboots the VM;
keep its GCE startup script registered so installation can finish verification
after that reboot. It refuses to replace a different loaded NVIDIA driver.
The verified `nvidia-driver-open` package depends on `nvidia-dkms-open`, so
kernel-module installation can compile code and take several minutes. Availability
of a prebuilt R615 module for a particular Ubuntu kernel has **not** been
verified; environments forbidding compilation need a prepared host image.

Ubuntu's CPU mitigations and automatic-upgrade policy are retained by default.
The explicit benchmark options `DISABLE_MITIGATIONS=1` and
`DISABLE_AUTO_UPGRADES=1` respectively disable guest CPU vulnerability
mitigations and background APT upgrades. Use them only under the operator's
security and maintenance policy; they are not prerequisites for container
startup. Measurements taken with different host settings need revalidation.

Read-only checks on 2026-10-05 confirmed the public image family resolves to
`ubuntu-2604-resolute-amd64-v20260918`, the pinned driver and matching amd64
libraries exist in [NVIDIA's Ubuntu 26.04 repository](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/),
and the keyring SHA-256 matches the script. NVIDIA's
[toolkit package index](https://nvidia.github.io/libnvidia-container/stable/deb/amd64/Packages)
includes 1.20.1-1; Canonical provides
[`docker-compose-v2` for Ubuntu 26.04](https://packages.ubuntu.com/resolute/amd64/docker-compose-v2).
These checks establish availability, not successful installation on a new VM.

## Prepared host image

The default `HOST_SETUP_MODE=verify` uses the `avplumber-mixer-host` image family
in `PROJECT`. That is an operator-supplied image, not a public image provided
by this repository. It must already contain a matching driver, Docker with
Compose, NVIDIA Container Toolkit, graphics/video libraries and DRM settings:

```sh
PROJECT=<project> ZONES='<zone>' IMAGE_NAME=<prepared-image> \
  IMAGE_PROJECT=<image-project> demos/mixer/deploy/gcp/create-host.sh <name>
```

This path installs no packages. A failed check means the image needs correcting;
it does not switch to the installer. Neither mode requires application images
to be baked into the boot image. The readiness probe may pull `ubuntu:24.04`;
that probe container's distribution is independent of the Ubuntu 26.04 host
and Fedora application containers.

## Readiness and deployment

On each boot, `host-setup.sh` clears old readiness and verifies the exact driver,
DRM render node, GBM/EGL/video libraries, Docker's NVIDIA default runtime and
Compose. A probe container must see the GPU, render node, GBM backend, NVDEC
and NVENC libraries. Success writes `/var/lib/avplumber-host/ready` and a
`ready-boot-id` matching `/proc/sys/kernel/random/boot_id`. The creator checks
both, so readiness inherited from a boot image cannot satisfy the wait.

Read `/var/log/avplumber-host-setup.log` for failures; installed container-runtime
versions are recorded in `/var/lib/avplumber-host/packages.txt`. A creation or
setup timeout leaves the VM running and prints its deletion command.

```sh
PROJECT=<project> demos/mixer/deploy/gcp/push-source.sh <name>
```

`push-source.sh` copies the committed tree and checked-out submodules, including
nested submodules. It preserves existing media and both mixer stack `.env`
files on the destination. It does not send local source modifications or media
from another host. Follow the portable stack's instructions to build/pull its
images, configure public endpoints and prepare media on the destination. The
preview proxy must forward WebSocket upgrades for the player's `/janus` path;
Janus REST remains available for mountpoint management.

The stock-Ubuntu mixer provisioning path has been checked for shell syntax and
launch argument handling, but has **not** been boot-tested on a fresh L4 VM.
Destination GPU quota/capacity, package availability for its selected kernel,
DKMS/Secure Boot compatibility, container registry access and firewall/UDP
reachability still require verification there. Host readiness does not prove
browser buffer allocation or sustained mixer performance: test the actual
Fedora browser overlay and the chosen show after deployment. Capacity profiles
in [`../../instance_profiles.py`](../../instance_profiles.py) are measured per
host type and must not be scaled to an unmeasured GPU.
