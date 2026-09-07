# CUDA overlay demo

[![One base plus fifteen transparent overlays composed on the GPU](https://github.com/amagimedia/avplumber/releases/download/cuda-overlay-demo-media-2026-09/cuda-overlay-result.png)](https://amagimedia.github.io/avplumber/demos/cuda-overlay/docs/)

[View the demo](https://amagimedia.github.io/avplumber/demos/cuda-overlay/docs/) · [Full processing graph](https://github.com/amagimedia/avplumber/releases/download/cuda-overlay-demo-media-2026-09/cuda-overlay-graph.png) · [Reference](docs/guide.md)

Compose one base image and up to fifteen transparent overlays with
`overlay_many_cuda`. The generated labels, alpha ramps and overlapping shapes
make layer order and transparency visible. Each result is compared pixel for
pixel with an independent CPU reference. This is a static validation demo.

## Run

Use Docker and the [shared NVIDIA setup](../README.md), with an NVIDIA GPU of
compute capability 7.0 or newer. The image builds public FFmpeg sources with
this repository's patches and CUDA 11.7; it uses its own toolkit.

From the repository root:

```sh
./demos/cuda-overlay/run.sh
```

For the sixteen-input 1080p example shown above:

```sh
./demos/cuda-overlay/run.sh --width 1920 --height 1080 --counts 15
```

The default sweep checks 1–15 overlays in three supported YUV/alpha format
combinations on a 641×360 canvas, including odd-width edge cases. Any pixel
mismatch returns a failing exit status. See the [reference](docs/guide.md) for
supported formats, a smaller sweep and build-only instructions.

## Processing graph

[![CUDA overlay WebUI: source readers and decoders feed one GPU filter and raw output](https://github.com/amagimedia/avplumber/releases/download/cuda-overlay-demo-media-2026-09/cuda-overlay-graph.png)](https://github.com/amagimedia/avplumber/releases/download/cuda-overlay-demo-media-2026-09/cuda-overlay-graph.png)

**50 nodes / 49 queues** for sixteen inputs. The capture shows the completed
641×360 validation graph; click it to inspect every node. Software fixtures
are uploaded once, composed on CUDA, and downloaded once for exact comparison.
Those transfers are explicit validation boundaries.

## Results

Each run writes a timestamped directory under `demos/cuda-overlay/artifacts/`:
source images, GPU results, contact sheets, logs and `report.json` with per-plane
comparison results. Raw Y/U/V data determines correctness; PNGs are previews.

The [demo page](https://amagimedia.github.io/avplumber/demos/cuda-overlay/docs/) uses the same presentation as
[Replay](../replay/README.md), [Mixer](../mixer/README.md) and
[DMA-BUF browser capture](../dmabuf-browser/README.md). New preview media is hosted
as public release assets, outside Git history.
