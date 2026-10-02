# GCP GPU hosts for the mixer

A mixer host starts from the **`avplumber-mixer-host`** image, so a launch installs and compiles
nothing and the host is ready in a minute or two.

```sh
PROJECT=<project> demos/mixer/deploy/gcp/create-host.sh <name>       # one L4, g2-standard-16, 256 GB
PROJECT=<project> demos/mixer/deploy/gcp/push-source.sh <name>       # committed tree + submodules to /srv/avplumber
```

`create-host.sh` creates the host from the newest image in the `avplumber-mixer-host` family, tries
each zone in `ZONES` (single zones regularly run out of GPUs), and waits until `host-setup.sh` writes
`/var/lib/avplumber-host/ready`. It defaults to one L4 on a g2-standard-16 (16 vCPU, 64 GB);
`MACHINE_TYPE=n1-standard-16 GPU=nvidia-tesla-t4` creates a T4 host.

`host-setup.sh` runs as the startup script on every boot and only verifies. It installs nothing:
- the expected NVIDIA driver is loaded (615.71.09, which the Fedora 44 / CUDA 13.4 mixer image needs);
- the kernel booted with `mitigations=off`;
- Ubuntu's apt timers are off;
- nvidia-drm runs with modeset, and the GBM, EGL, optical-flow and NVENC libraries are present;
- Docker's default runtime is nvidia and a container sees the GPU.

A host that fails a check is replaced from a matching image, never repaired in place. The log is
`/var/log/avplumber-host-setup.log`.

The image carries Ubuntu 26.04 with NVIDIA 615.71.09, Docker with Compose, the NVIDIA Container
Toolkit and the mixer, browser and Janus images. No prebuilt 615 kernel module exists for Ubuntu 26.04
(Canonical's archive has signed prebuilt modules up to 610), so the module in the image was built once,
when the image was made. A new driver or kernel means a new image, never a change on a running host.

Source limits are measured per host type and live in [`../../instance_profiles.py`](../../instance_profiles.py);
a new GPU type needs its own measured entry, never a scaled one.
