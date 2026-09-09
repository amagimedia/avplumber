# MXL demo

End-to-end proof that avplumber can publish and consume flows on the
[MXL](https://github.com/dmf-mxl/mxl) shared-memory transport, using
avplumber's generic `input` / `output` nodes with `format="mxl"` — no
MXL-specific node types are introduced.

Two graphs run in one process:

* **writer** — `lavfi testsrc` → decode → rescale to yuv422p10le →
  encode as `v210` (10-bit 4:2:2 uncompressed, per SMPTE ST 2110) →
  mux → `output(format="mxl", url="/dev/shm/mxl")` (the MXL FFmpeg
  muxer takes a plain filesystem path).
* **reader** — `input(format="mxl", url="mxl:///dev/shm/mxl?id=<uuid>")`
  → demux → decode v210 → rescale to yuv420p → encode as mpeg4 → mux
  → fragmented mp4.

Verified end-to-end on Docker Desktop (aarch64) producing 28k mpeg4
packets of the testsrc pattern in a 10-second run.

## Status

NVIDIA support is temporarily disabled in this image. The Dockerfile
applies only patches `0001`, `0005`, `0006`, `0008` from
`deps/ffmpeg-patches/`; `0002` (CUDA composition suite), `0003` (NPP
CUDA13 compat), `0004` (NVDEC intra), and `0007` (NDI) are held back
until the mxl path is folded into `demos/mixer/Dockerfile` alongside
CUDA.

## Requirements

* Linux x86_64 or aarch64 (MXL SDK is Linux-only — no macOS support).
* Docker with enough tmpfs at `/dev/shm` (default is fine for the demo).
* The 8-patch stack applied to `n7.1.5`. `deps/ffmpeg-patches/verify.sh`
  confirms the tree hash; the MXL patch (`0008-...`) is generated
  from `cbcrc/FFmpeg` via `deps/ffmpeg-patches/Dockerfile.mkpatch`
  — see that directory's README.

## Build

From the repository root (after `git submodule update --init --recursive`):

```sh
docker build -f demos/mxl/Dockerfile -t avplumber-mxl:local .
```

The build:

1. Bootstraps `microsoft/vcpkg` and Rust 1.88.0.
2. Builds and installs the MXL SDK (`dmf-mxl/mxl` @ `v1.1.0-beta-1`)
   into `/usr/local` — libmxl is statically linked against spdlog, so
   its `.pc` file is scrubbed of the private `Requires` before FFmpeg
   configure.
3. Applies the reduced patch stack to FFmpeg `n7.1.5`, strips a
   handful of fork-side test scaffolding that references files not
   present in `n7.1.5` (`tests/fate/ogg-*.mak`, duplicated
   `fate-mxl-uri` rule), and configures with `--enable-libmxl
   --enable-demuxer=mxl --enable-muxer=mxl --enable-protocol=mxl`.
4. Builds `pyplumber` with `HAVE_CUDA=0 NEURAL_NET=0` (`-j2` to stay
   inside Docker Desktop's memory ceiling on Apple Silicon).

Verified inside the image:

```sh
ffmpeg -hide_banner -demuxers | grep mxl   # D   mxl   Media eXchange Layer
ffmpeg -hide_banner -muxers   | grep mxl   #  E  mxl   Media eXchange Layer
python3 -c 'import pyplumber'              # OK
```

## Run

The demo needs a shared `/dev/shm/mxl` domain directory with an
`options.json` file. On a single host use `--ipc=host`:

```sh
mkdir -p /dev/shm/mxl
echo '{"urn:x-mxl:option:history_duration/v1.0": 100000000}' \
  > /dev/shm/mxl/options.json

docker run --rm --ipc=host \
    -v /dev/shm/mxl:/dev/shm/mxl \
    -v "$PWD/demos/mxl/test-media:/media" \
    -e AVP_OUTPUT=/media/out.mp4 \
    avplumber-mxl:local
```

Defaults to publishing an `lavfi testsrc` pattern; override with
`-e AVP_INPUT=/media/your-file.mp4` for a real file (played back
looped and realtime-paced through `InputRec`).

Runs both graphs in one process. Ctrl-C to stop — the fragmented mp4
stays playable even on interrupt because `movflags=frag_keyframe+
empty_moov+default_base_moof` is set on the reader-side output.

Split writer and reader across two containers by passing
`--writer-only` / `--reader-only` and sharing the flow UUID via
`AVP_MXL_VIDEO_ID`.

## Codec choices

MXL flows carry uncompressed frames. The muxer registered by
`0008-*.patch` insists on `v210` (10-bit 4:2:2 packed) for video —
`rawvideo` is rejected at header write. The reader re-encodes to
`mpeg4` (rather than H.264) because the NVIDIA-off image does not
link `libx264`.

## Known gaps

* Passing `blocking=1` to the MXL demuxer through avplumber's `options`
  dict does not currently reach the demuxer's private AVOptions
  — the reader survives EAGAIN via `auto_restart:"group"` instead.
  Root-cause is in the order of format assignment vs `openInput` in
  `src/nodes/input.cpp`; direct `ffmpeg -blocking 1` on the CLI works.
* The demo runs the reader at wall-clock max (~2800 fps into mpeg4
  at 320×240) because we didn't wire in a realtime pacer on the
  reader side. Adding a `RealtimeVideoFrame` node between decode and
  encode would cap the reader to the source frame rate.

## References

* MXL SDK: <https://github.com/dmf-mxl/mxl>
* MXL FFmpeg fork (source for the 0008 patch): <https://github.com/cbcrc/FFmpeg>
* Reference build guidance: <https://github.com/cbcrc/guidance-for-building-ffmpeg-with-mxl>
