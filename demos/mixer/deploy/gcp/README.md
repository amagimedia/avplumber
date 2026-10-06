# GCP mixer hosts

Provision the host here, then deploy the [L4 stack](../l4/README.md).
No host CUDA/TensorRT SDK or copied Docker state is needed.

## Fresh Ubuntu 26.04

```sh
PROJECT=<project> ZONES='<zone> <fallback-zone>' HOST_SETUP_MODE=install \
  demos/mixer/deploy/gcp/create-host.sh <name>
```

Defaults: one L4 on `g2-standard-16`, 256 GB balanced disk, public Ubuntu 26.04.
The installer pins the NVIDIA driver, installs Docker/Compose and Container
Toolkit, configures DRM/GBM and reboots. Run before application workloads;
it refuses to replace a different loaded driver. DKMS may compile modules.
See [host-setup.sh](host-setup.sh) for current versions and overrides.

CPU mitigations and automatic upgrades stay enabled by default. Benchmark-only
`DISABLE_MITIGATIONS` / `DISABLE_AUTO_UPGRADES` are explicit operator choices.

## Prepared host image

```sh
PROJECT=<project> ZONES='<zone>' IMAGE_NAME=<prepared-image> \
  IMAGE_PROJECT=<image-project> demos/mixer/deploy/gcp/create-host.sh <name>
```

Default `HOST_SETUP_MODE=verify` installs nothing. The image must already have
the matching driver, DRM/GBM/video libraries, Docker/Compose and NVIDIA runtime.
The default image family is operator-provided, not distributed by this repository.

## Readiness and deployment

Each boot checks host and container GPU/DRM/video access; readiness must match
the current boot ID. Failures are in `/var/log/avplumber-host-setup.log`.
A timeout leaves the VM running and prints its deletion command.

```sh
PROJECT=<project> demos/mixer/deploy/gcp/push-source.sh <name>
```

This transfers committed source and checked-out submodules, preserving remote
media and `.env` files. It excludes local uncommitted changes. Follow the L4
stack instructions for images, endpoints, firewall and media generation.

The installer passed fresh-host validation on 2026-10-05 with Secure Boot off.
That does not qualify other kernels/Secure Boot settings or sustained mixer
performance. Validate actual browser capture, WebRTC and the chosen workload.
