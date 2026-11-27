# Live mixer demo

Mix video and browser sources on a CUDA canvas, with Preview/Program,
Cut/Fade/media wipes, downstream keys and AUX monitors. Video only.

[![192-input mixer: Program/Preview and dirty Program with four downstream keys](https://amagimedia.github.io/avplumber/demos/mixer/docs/mixer-demo-poster.jpg)](https://amagimedia.github.io/avplumber/demos/mixer/docs/)

[Watch the 20-second demo](https://amagimedia.github.io/avplumber/demos/mixer/docs/) ·
[WebGL graph demo and zoom levels](https://amagimedia.github.io/avplumber/) ·
[MP4](https://github.com/amagimedia/avplumber/releases/download/mixer-demo-media-2026-10/mixer-192-input-tbar-retina-20261028-30fps.mp4) ·
[HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/) ·
[Configuration](docs/config.md) · [Run manually](docs/guide.md)

### On a desktop

Mixer is the main window and has a dark menu bar. Setup, Graph and the other tools open as separate windows you can place on another monitor. On Linux, tap **Alt** in a tool window to reveal the menu, or press **Ctrl+1** to bring Mixer back, even if you closed it.

1. Run `./demos/mixer/desktop.sh` from the checkout as your normal user. On a local Linux graphical login with an NVIDIA GPU, the launcher builds and opens the **Mixer** desktop app. First startup opens **Setup** while the demo prepares media. A running show opens directly in **Mixer**.
2. In **Setup**, choose the sources, canvas and outputs. **Apply setup** pauses output while the mixer prepares the show and restarts. When the status says **Mixer ready**, choose **Open mixer controls**. Select a scene on Preview, then press **C** for Cut or **Space** for the selected transition. The keyboard legend at the top of Mixer lists the other shortcuts.
3. Open **Graph** from Mixer to inspect the live nodes and queues in a separate window. Its bottom bar shows delivered Program fps, total frame traffic across the queues and the Program compositor’s mean, p95 and p99 draw times once frames are flowing. Compare the draw time with one frame period. **Setup** and **Graph** each reopen their existing window, preserving your settings edits and graph position.
4. Open **Outputs** for the output wall. In the desktop app, Program and the configured monitoring buses use local GPU textures. Choose the Preview bus to see the scene queued for the next transition. The monitors show SDR video, including for an HDR show. Audio meters and mixer controls remain available. Quitting the GUI leaves the show running. Run `./demos/mixer/desktop.sh app` to reopen it.

**Desktop detection.** Automatic launch requires an active, local graphical login and a usable Wayland or X11 socket. A headless or SSH launch builds the show services only. Set `MIXER_DESKTOP_UI=off` to select this explicitly, or `MIXER_DESKTOP_UI=on` to request the GUI from a graphical session the launcher cannot identify. The headless HTML input service still uses Electron to render pages. The optional GUI container reuses that runtime. The host needs Docker Compose with BuildKit and the NVIDIA Container Toolkit.

**Browser access.** For a headless show, or to operate from the browser, open [Setup](http://127.0.0.1:7681/setup/), then [Mixer](http://127.0.0.1:7681/). Mixer links to [Graph](http://127.0.0.1:7681/graph/) and [Outputs](http://127.0.0.1:7681/wall). Browser players use the configured Janus feeds.

Captured 2026-10-28 on the 192-input develop show: 105 NVDEC videos, 36 browser scene
sources, 47 raw NV12 feeds and four browser keys, with 256 scenes and 24 encoded outputs
at 1920×1080p30 SDR on one NVIDIA L4. All four keys are enabled.

The two viewers use native encoded pictures captured on the host before Janus. The actual
control page replays the recorded mouse-operated T-bar and sampled performance meters at
2656×1468, 2× Retina scale. The sequence includes a manual fade, halfway hold, completion,
return stroke and cut. CPU/GPU meters show captured measurements; this is an offline UI
replay, not a WebRTC screen recording. Both streams have continuous 30 fps RTP timestamps
and zero packet gaps; all 599 measured ticker steps advance by exactly four pixels.

Setup counts the 188 scene sources separately from the keys. This documentation recording
does not refresh the historical [capacity measurements](docs/capacity.md).

### Inspect the 192-input graph

The [WebGL viewer demo](https://amagimedia.github.io/avplumber/) follows the same
192-input workload from the whole show down to individual processing nodes. Its
[four zoom-level screenshots](https://amagimedia.github.io/avplumber/#zoom) and
24-second video show 999 nodes and 5,378 connected queues captured on 2026-10-29.
The viewer replays recorded telemetry at 2× Retina scale and 30 fps; graph counts
belong to this capture, independently of older architecture or capacity reports.
See the [WebUI documentation](../../tools/web-ui/README.md) to connect it to your
own mixer. The portable L4 deployment exposes it through the controls' **Graph** link.

## Run

Use a Linux NVIDIA host with NVDEC/NVENC, Docker Compose, NVIDIA Container
Toolkit and DRM/EGL/GBM for browser capture. Start from a recursive checkout;
`docker-compose/mixer/host-setup.sh check` lists anything missing with its fix, and
`sudo docker-compose/mixer/host-setup.sh install` sets up Ubuntu 24.04/26.04
([host setup](../README.md#nvidia-host-setup)).

```sh
docker compose -f docker-compose/mixer/compose.yaml up --build
```

The Fedora/CUDA image requires driver R615 or newer.
The [L4 preset](../../docker-compose/mixer/deploy/l4/README.md) supplies a separate measured configuration.

Open <http://127.0.0.1:7681/setup/>, choose sources, canvas and outputs, then
**Apply setup**. Controls are at `/`, all outputs at `/wall`, and the standalone
player at <http://127.0.0.1:8080>. Select SDR if HEVC playback is unavailable.

Janus offers the address of the default route's interface to WebRTC viewers, and on GCE or EC2
also the VM's public address from the instance metadata. Set `JANUS_HOST_IP` when viewers reach
this host on another interface (a VPN, a second NIC), or `JANUS_NAT_1_1` for other 1:1 NAT.
The page and the player have no authentication, so they listen on 127.0.0.1 only. For a remote
host set `MIXER_BIND=0.0.0.0 JANUS_PREVIEW_BIND=0.0.0.0`, allow TCP 7681/8080 and UDP 20000–20100
from trusted networks only, or use the L4 preset's proxy with `MIXER_HTPASSWD`. The mixer's control
port 7777 stays on loopback. Stop with `docker compose -f docker-compose/mixer/compose.yaml down`.

Setup saves settings and generated assets in `media/`; applying changes pauses
output before asset preparation and restarts the mixer afterward. Synthetic
clip generation uses NVENC and must not compete with the live show for GPU
memory. Cached clips are reused. Keep that directory writable and mount it
whole: Setup atomically replaces the generated show. Capacity comes from the
[instance profile](instance_profiles.py), not a universal source limit. Setup picks it from the GPU:
a T4 or L4 gets its measured profile, any other GPU the [GeForce sizes](#geforce-desktops);
`MIXER_INSTANCE_TYPE` overrides the choice.

### GeForce desktops

RTX 20, 30, 40 and 50 series cards, and other GPUs without a measured profile.

Use native x86-64 Linux, Docker Compose and the NVIDIA Container Toolkit.
**R615 is the minimum host driver for the desktop image;** 620.70.10 ran the desktop show on a
T4 on 2026-11-25. The build uses CUDA Toolkit 13.4. At runtime the Container Toolkit
mounts the installed host driver libraries. The host needs no CUDA Toolkit.

| Linux driver | CUDA version reported by the driver | Desktop mixer image |
| --- | --- | --- |
| R595 | 13.2 | Below the minimum for Blackwell PTX and the pinned video codec API |
| R615 | 13.4 | Meets the minimum |
| R620 | 13.5 | Runs the CUDA 13.4 build (620.70.10 run on a T4) |

Ubuntu 26.04 kernels from 7.0.0-1014 (the fresh GCP image ships 7.0.0-1017) do not build the
NVIDIA 615 or 620 module (2026-11-25): they lack `dma_buf_move_notify`. Install an earlier kernel
(7.0.0-1011 ran 620.70.10 on a T4 and an RTX PRO 6000 Blackwell), make it the GRUB default and
remove the newer ones, or use Ubuntu 24.04. `host-setup.sh check` flags a boot kernel without the
module and `install` says so instead of failing in dpkg.

On the RTX 5080 and 5090 the driver compiles the image's CUDA 13.4 PTX on
first use. FFmpeg also uses Video Codec SDK 13.1 headers, which require
R610 or newer. [CUDA driver compatibility](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/#cuda-driver) ·
[FFmpeg codec headers](https://github.com/FFmpeg/nv-codec-headers/blob/n13.1.15.0/README)

From a fresh checkout, run the single-stream GPU smoke check first, then start
the 96-input mixer.

```sh
git clone --branch develop --recurse-submodules https://github.com/amagimedia/avplumber.git
cd avplumber
./demos/mixer/desktop.sh smoke
./demos/mixer/desktop.sh
```

First start builds the remaining images and generates the largest show Setup suggests for the
GPU (below). The desktop
build leaves the optional MXL SDK disabled (`WITH_MXL=0`). Open
<http://127.0.0.1:7681/setup/> to follow preparation. When it says **Mixer ready**,
open the [controls](http://127.0.0.1:7681/), [wall](http://127.0.0.1:7681/wall)
or [live graph](http://127.0.0.1:7681/graph/). To change the load, choose a size, then
**Apply setup**. Every size is a 1920×1080p25 SDR show with up to 256 scenes, four downstream keys,
Program and clean Program, Program/Preview and Multiviewer. You can lower **Unique sources**
before applying. The launcher keeps shows and the CUDA JIT cache in
`media/desktop/` across restarts.

The desktop app shows Program, the clean feed and every AUX bus from the GPU, and the AUX buses
are then not encoded: a browser beside the app plays Program and the clean feed only
(`MIXER_LOCAL_PREVIEWS=0 ./demos/mixer/desktop.sh` keeps the AUX WebRTC streams). **Source
thumbnails** in Setup adds one 8 × 8 grid output per 64 sources.

For an 8-bit 4:2:2 show, choose **SDR · 8-bit · 4:2:2** in Setup. The NV16 canvas preserves each input chroma row from SDR v210 through compositing. H.264 and HEVC outputs use 4:2:0. Setup estimates NV16 capacity from the 10-bit 4:2:2 profile. The desktop smoke check compares NV16 output pixels with a reference during fades, dips and wipes.

| Size | HEVC decoded | Raw NV12 | Browser scene sources | Browser keys |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 15 | 4 | 1 | 4 |
| 48 | 30 | 8 | 6 | 4 |
| 96 | 60 | 16 | 16 | 4 |
| 192 | 120 | 32 | 36 | 4 |

Setup reads GPU 0's name and free VRAM, looks up its NVDEC and NVENC engines in
[NVIDIA's support matrix](https://developer.nvidia.com/video-encode-decode-support-matrix)
(`gpu_engines.json`) and suggests the largest size that fits. The limits scale a measured
reference by those engines and the free memory, never above it: the T4 for Turing and Ampere
cards, the L4 for Ada and newer. The 192-input, 26-output L4 reference run peaked at 19,295 MiB.
An RTX 5080 (2 NVDEC, 16 GB) gets 96, an 8 GB RTX 4060 48, an RTX 3060 24. A T4 or L4 keeps its
own measured profile. The sizes are estimates for GeForce cards. On 2026-11-25 `desktop.sh` ran the 192-input show on an
RTX PRO 6000 Blackwell (GCP G4, driver 620.70.10) with no missed deadline, decoders at 46% and
17 GB of VRAM, and 48 inputs on a T4. Told it was an RTX 5080 (2 NVDEC, 2 NVENC, 16 GB), Setup
suggested 96 inputs; on the Blackwell card that show peaked at 9.0 GB of VRAM with 4 encode sessions
and about 45% of two decoders' worth of NVDEC, with no missed deadline.
The driver allows 12 concurrent GeForce encode sessions per system, shared
with other applications. [L4 measurements](https://amagimedia.github.io/avplumber/demos/mixer/docs/capacity-l4.json) ·
[NVIDIA session limits](https://developer.nvidia.com/video-encode-decode-support-matrix)

The launcher selects GPU 0's NVIDIA render node for both Wayland and Chromium.
It advertises the default network interface's IPv4 address for WebRTC. Set
`JANUS_HOST_IP` to the desktop's LAN address if the default route uses a VPN.
The page has no password, so the launcher serves it on 127.0.0.1 only; set
`MIXER_PUBLIC_BIND=0.0.0.0` to open it to the LAN.
Use `./demos/mixer/desktop.sh check` for the host prerequisite checks alone.

The raw NV12 inputs in both presets keep their loops in page-locked host RAM.
CUDA allocation failures stop the raw input. The pinned buffers consume system
RAM in addition to the GPU memory needed for the show.

`smoke` builds the mixer image, compares raw uploads from pinned memory byte
for byte, then encodes a short HEVC clip and checks 50 decoded frames through the CUDA compositor against the linear CUDA
reference, including timestamps. It can run before the show starts.
Use `./demos/mixer/desktop.sh logs -f mixer` for startup logs and
`./demos/mixer/desktop.sh down` to stop the stack.

## Controls

- Preview/Program: select the next scene, or enable Direct to cut on selection.
- Cut, Auto, Fade, colour Dip and media wipes; Auto can halt/resume or complete a take.
- T-bar: horizontal or upright, manual fade with a hold, cancellation or return stroke;
  cached wipes can follow the lever too.
- DSK: independent per-key fades or cuts, a shared ALL rate with per-key overrides,
  and Shift+click for an immediate key cut.
- NEXT: arm background and keys for one transition, or take only the armed keys.
- FTB: fade the keyed program to black while the clean feed remains available.
- Two resizable viewers, editable multiview slots, AUX layouts/source paging and an output wall.
- Live GPU, encoder/decoder, memory, power, CPU, thread and delivery statistics.

See the [operator controls](docs/guide.md#controls) or the
[illustrated Pages guide](https://amagimedia.github.io/avplumber/demos/mixer/docs/controls.html)
for rate editing, NEXT behavior and shortcuts.

## Configure and extend

- [Show configuration](docs/config.md) and [example](config.example.json): sources, scenes and outputs.
- [Recipes](docs/recipe.md): generate test media and scene layouts.
- [Operator guide](docs/guide.md): CLI, TUI, Docker, proxy and tests.
- [Capacity](docs/capacity.md) and [latency](docs/latency.md): scoped measurements and reproduction.
- [HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/): implementation and tuning notes.
- [DMA-BUF integration](../../doc/dmabuf.md): shared browser capture and the GBM shim.
- [Mixer base API](../../doc/mixer.md): reuse `pyplumber.mixer.build_application` in other applications; no demo imports required.
- [Audio-follow-video](../../doc/mixer_audio_follow_video.md): the `audio_router_node` signal and its `transition_ended` / `transition_progress` / `transition_start` / `transition_paused` protocol; open `audio_mixer` demo: [demos/mixer_audio](../mixer_audio).
