# Raw v210 to CUDA

`v210_to_cuda` uploads packed, little-endian v210 bytes and unpacks them on the
GPU. Each input `av::Packet` must hold one complete raw 10-bit 4:2:2 frame.
Output is an `av::VideoFrame` with `format=cuda` and an FFmpeg hardware frame
context whose software format is `p210le` or `yuv422p10le`.

This is a raw byte conversion node, not a compressed-video decoder. It needs
CUDA and NVCC at build time, and FFmpeg 8.1 CUDA format support at runtime.
It does not use NVDEC, require Blackwell, or depend on MXL. An MXL adapter can
later feed its packed frames into the same packet edge.

```text
host v210 packet -> pinned staging copy -> GPU upload -> CUDA unpack -> CUDA frame
```

The pinned host buffer and packed GPU buffer are allocated once per node.
Output surfaces use FFmpeg's frame pool. The node completes its private CUDA
stream before publishing a frame, so downstream filters can use other streams
and input owners can safely release their buffers. This is a host/GPU ingress
copy; subsequent processing can remain on the GPU.

## Parameters

| Parameter | Meaning |
| --- | --- |
| `src`, `dst` | Packet input edge and video-frame output edge. |
| `hwaccel` | Name of an initialized CUDA hardware device. |
| `width`, `height` | Required dimensions; width must be even. |
| `fps` | Required frame rate, such as `60/1` or `60000/1001`; does not pace input. |
| `stride` | Bytes per input row. Default is `ceil(width / 48) * 128`, the standard v210 alignment. Explicit strides must be multiples of four and fit every active sample. |
| `format` | `p210le` (default) or `yuv422p10le`. Both retain every 10-bit sample and 4:2:2 chroma. |
| `timebase` | Output timestamp base; defaults to the reciprocal of `fps`. Packet PTS, DTS and duration are rescaled from the packet's time base. |
| `sample_aspect_ratio` | Defaults to `1/1`. |
| `color_range`, `colorspace`, `color_primaries`, `color_trc`, `chroma_location` | FFmpeg names, e.g. `tv`, `bt709`, `bt709`, `bt709`, `left`. Default is unspecified; raw v210 bytes do not describe these properties. |

Packets need valid PTS and exactly `stride * height` bytes. Missing timestamps,
truncated frames and mismatched layouts are errors. EOF is forwarded and ends
the node. Dimensions and format are fixed for the node's lifetime.

P210 stores each 10-bit sample in the high bits of a 16-bit word (`sample << 6`),
with a Y plane and interleaved UV plane. Planar `yuv422p10le` stores samples in
the low ten bits of three 16-bit planes. Neither representation discards bits.
v210's two unused bits per 32-bit input word are ignored.

## Byte simulator and verification

Generate one second of moving 1080p60 test frames, with no MXL service:

```sh
python3 tests/cuda/v210_fixture.py <path>/frames.v210 --frames 60
```

Feed the file through the existing `input` and `demux` nodes, then this node:

```text
hwaccel.init {"name":"gpu","type":"cuda"}
node.add {"name":"input","type":"input","url":"<path>/frames.v210","format":"v210","options":{"video_size":"1920x1080","framerate":"60"},"dst":"packets"}
node.add {"name":"demux","type":"demux","src":"packets","routing":{"v:0":"v210"}}
node.add {"name":"upload","type":"v210_to_cuda","src":"v210","dst":"cuda_422","hwaccel":"gpu","width":1920,"height":1080,"fps":"60/1","color_range":"tv","colorspace":"bt709"}
```

Attach a CUDA-capable consumer before starting the graph. The current mixer,
custom padding and transition paths are still 8-bit; they cannot preserve this
format without additional work.

The tests require NumPy, and fixture unit tests additionally require pytest:

```sh
python3 -m pytest -q tests/cuda/test_v210_fixture.py
nvcc -std=c++17 tests/cuda/test_v210_unpack.cu -o <path>/test_v210_unpack
<path>/test_v210_unpack
python3 tests/cuda/smoke_v210_to_cuda.py
python3 tests/cuda/smoke_v210_to_cuda.py --width 50 --height 9 --stride 144 --frames 3 --scale
```

Run the CUDA tests on an NVIDIA host. The kernel test covers both output
layouts, 10-bit codes, partial packing groups and independent plane pitches.
The graph test compares generated pixels with FFmpeg's CPU v210 unpacker and
GPU results, checking PTS, EOF and optional CUDA scaling. Its `hwdownload` is
the explicit GPU-to-CPU verification boundary, not part of a production graph.

The 13 simulator unit tests pass locally. Validated on a Tesla T4 with
FFmpeg 8.1: 62 CUDA kernel cases, 60 full-HD frames per output format at 60 fps timestamps, and
three 50x9 frames per format with a 144-byte stride and 2x CUDA scaling.
Every active output sample matched the generated reference; the standard
v210 fixture also matched FFmpeg's CPU decoder. Graph checks covered PTS
rescaling, sample aspect ratio and clean EOF. These are correctness checks,
not a CPU-utilization or maximum-throughput benchmark.
