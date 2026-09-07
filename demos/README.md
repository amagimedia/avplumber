# Demos

See the [demo comparison](../README.md#demos) to choose a demo. All commands
below run from the repository root. The Python video runtime below builds from public sources; no company
account, private image, or separately installed CUDA toolkit is needed.

Start with [Playlist](playlist/README.md): its Docker image includes generated
clips, and its UI-only preview works without a GPU.

## NVIDIA host setup

Use a Linux x86-64 host with Docker, Docker Compose, an NVIDIA driver, and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Follow that guide to enable Docker GPU access, then check:

```sh
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.7.1-base-ubuntu22.04 nvidia-smi
```

The CUDA demo images compile for compute capability 7.0 or newer.
Playback and streaming also need hardware decode and H.264 NVENC support;
having CUDA alone is insufficient. Browser capture additionally requires the
NVIDIA graphics driver libraries and `/dev/dri`, not a compute-only driver.
See each demo's requirements for details.

```sh
git clone --depth 1 --recurse-submodules --shallow-submodules \
  https://github.com/amagimedia/avplumber
cd avplumber
```

For an existing checkout, run `git submodule update --init --recursive`.

The Docker commands use `:z` on media mounts so they also work with SELinux
enabled, as on Fedora. Mount only the demo's media directory.

## Build the Python video runtime

The [mixer image](mixer/README.md#run) builds avplumber, its Python module,
and patched FFmpeg together. Playlist and Replay can reuse this image:

```sh
docker build -f demos/mixer/Dockerfile -t avplumber-mixer:local .
```

Then follow [Mixer](mixer/README.md#run),
[Playlist](playlist/README.md#run-in-docker), or
[Replay](replay/README.md#run-in-docker) for commands and controls.
The first build compiles FFmpeg and avplumber; subsequent builds reuse Docker's
cache. These demos do not require neural models or TensorRT.

For an inference application, see the
[official TensorRT installation options](../doc/research/2026-09-07-public-tensorrt-installation.md),
including the limits of native Fedora support and minimal runtime packages.

## Start a WebRTC preview

Playlist and Replay require Janus; Mixer can also write directly to a file or
stream. Start the public Janus and preview services using the browser demo's
Compose file:

```sh
docker compose --env-file demos/dmabuf-browser/.env.example \
  -f demos/dmabuf-browser/compose.yaml up -d --build janus janus-preview
```

Open <http://127.0.0.1:8080> after starting a player. The included video-only
mountpoint accepts the demos' default RTP/RTCP ports, 5004/5005. Run one demo
output at a time on those ports.

For a remote host, prefix the Compose command with `JANUS_HOST_IP=<host>` using
the host address reachable by your browser, then open `http://<host>:8080`.
Allow the preview and Janus HTTP ports (8080, 8088) and WebRTC UDP ports
20000–20100 through the host firewall.

Stop the preview services with:

```sh
docker compose --env-file demos/dmabuf-browser/.env.example \
  -f demos/dmabuf-browser/compose.yaml down
```

## Other starting points

- [Playlist UI preview](playlist/README.md#preview-without-avplumber-cuda-or-janus):
  try the controls without video or GPU setup.
- [Browser capture](dmabuf-browser/README.md#run): a complete Docker stack with
  a bundled animated page, Electron, GPU capture, Janus, and browser preview.
- [CUDA overlay validation](cuda-overlay/README.md#run): run
  `./demos/cuda-overlay/run.sh` to build, generate fixtures, and compare results.

