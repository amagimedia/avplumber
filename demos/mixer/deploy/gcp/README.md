# GCP GPU hosts for the mixer

Reproducible mixer hosts on GCP, adapted from the sports-live-reframer host setup.

```sh
PROJECT=<project> demos/mixer/deploy/gcp/create-host.sh <name>       # one L4, g2-standard-16, 256 GB
PROJECT=<project> demos/mixer/deploy/gcp/push-source.sh <name>       # committed tree + submodules to /srv/avplumber
```

`create-host.sh` creates the host from Canonical's plain `ubuntu-2604-lts-amd64` image, tries each zone
in `ZONES`, and waits until `host-setup.sh` writes `/var/lib/avplumber-host/ready`. It defaults to one L4
on a g2-standard-16 (16 vCPU, 64 GB), the same vCPU count as the T4 reference host.
`MACHINE_TYPE=n1-standard-16 GPU=nvidia-tesla-t4` creates a T4 host instead.

`host-setup.sh` runs as the startup script on every boot and is idempotent. It:

- installs NVIDIA **615.71.09** (open kernel module, pinned), the version the Fedora 44 / CUDA 13.4 mixer image needs;
- installs the video, GL/EGL and GBM userspace, Docker with Compose and the NVIDIA Container Toolkit as Docker's default runtime;
- loads nvidia-drm with modeset, which the DMA-BUF browser sources need;
- disables Ubuntu's apt timers, whose background upgrades take every CPU for minutes;
- boots the kernel with `mitigations=off`, which takes a reboot on the first boot. The graph runs about 200
  threads, and the mitigations make every syscall and thread wake-up more expensive.

The setup log is `/var/log/avplumber-host-setup.log`.

On the host, build the image from the pushed tree with
`docker build -f demos/mixer/Dockerfile.fedora44 -t avplumber-mixer:local --build-arg BUILD_JOBS=16 .`,
or bring up the whole stack with [`../../compose.yaml`](../../compose.yaml).

Source limits are measured per host type and live in [`../../instance_profiles.py`](../../instance_profiles.py);
a new GPU type needs its own measured entry, never a scaled one.
