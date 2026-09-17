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
  → demux → v210 unpack → encode → fragmented mp4. The unpack step has
  two implementations, picked by `--gpu-unpack` (default `auto`):
  * **GPU** (a CUDA device is present) — `v210_to_cuda` copies each
    grain once into pinned memory and unpacks it to a CUDA `p210le`
    frame with a PTX kernel, then `scale_cuda` and `h264_nvenc` finish
    the file on the GPU. No CPU codec, no swscale.
  * **CPU** (no device) — the libavcodec `v210` decoder plus swscale to
    yuv420p and `mpeg4`.

Grains are taken **zero-copy** by default: the demuxer's `zero_copy=1`
points each AVPacket straight at the MXL ring buffer in `/dev/shm`
instead of copying it out. See "Zero-copy grains" below.

Verified end-to-end on an x86_64 Fedora host with an RTX 4000 Ada
(driver 615.71, Docker + NVIDIA Container Toolkit), on the FFmpeg 8.1
build of this patch stack (`n8.1-12-g93aafbb`, mxl demuxer and muxer
registered). Runs covered: GPU zero-copy, GPU with `--no-zero-copy` and
`--gpu-unpack off`, against both an `lavfi testsrc` and a looped file,
at 320×240p25 and 640×480p25. All of them ran without node failures,
started at PTS 0, reported 25.0 fps (25.06 on the NVENC path) in the
container header, and decoded back to the source pattern.

## Requirements

* Linux x86_64 (MXL SDK is Linux-only — no macOS support).
* Docker with enough tmpfs at `/dev/shm` (default is fine for the demo).
* NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  are optional: with them (`--gpus all`) the reader unpacks on the GPU
  and encodes with NVENC; without them it falls back to the CPU v210
  decoder. Other demos in the shared image do require CUDA.
* The FFmpeg 8.x series applied to `n8.1` (or `n8.0`).
  `deps/ffmpeg/verify.sh n8.1 <ffmpeg-repo>` confirms the tree hash; the
  MXL patch (`8/0011-...`) is generated from `cbcrc/FFmpeg` branch
  `dmf-mxl/8.1` via `deps/ffmpeg/Dockerfile.mkpatch` — see
  `deps/ffmpeg/README.md`.

## Build

This demo shares its runtime image with `demos/mixer/`. Build once
from the repository root (after `git submodule update --init --recursive`):

```sh
docker build -f demos/mixer/Dockerfile -t avplumber-mixer:local .
```

The image build:

1. Installs distro `gcc-11` plus `gcc-13` from
   `ppa:ubuntu-toolchain-r/test` (the MXL SDK needs C++20; FFmpeg and
   avplumber keep using gcc-11).
2. Bootstraps `microsoft/vcpkg` and Rust 1.88.0.
3. Builds and installs the MXL SDK (`dmf-mxl/mxl` @ `v1.1.0`)
   into `/usr/local` — libmxl is statically linked against spdlog, so
   its `.pc` file is scrubbed of the private `Requires` before FFmpeg
   configure.
4. Applies the eleven-patch FFmpeg 8.x series (`deps/ffmpeg/apply.sh`)
   to FFmpeg `n8.1` and configures with CUDA (`--enable-cuda
   --enable-cuda-nvcc --enable-cuvid --enable-nvdec --enable-nvenc`)
   plus MXL (`--enable-libmxl --enable-demuxer=mxl --enable-muxer=mxl
   --enable-protocol=mxl`).
5. Builds `pyplumber` with `HAVE_CUDA=1 HAVE_NVCC=1`.

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
# 1 s of history. The 100 ms the MXL samples use is only ~2 grains at
# 25 fps, which leaves zero-copy readers no slack at all.
echo '{"urn:x-mxl:option:history_duration/v1.0": 1000000000}' \
  > /dev/shm/mxl/options.json

docker run --rm --ipc=host --gpus all \
    --entrypoint python3 \
    -v /dev/shm/mxl:/dev/shm/mxl \
    -v "$PWD/demos/mxl/test-media:/media" \
    -e AVP_OUTPUT=/media/out.mp4 \
    avplumber-mixer:local /build/demos/mxl/mxl_demo.py
```

Drop `--gpus all` to force the CPU reader path (`--gpu-unpack` prints
which one it picked). The flow geometry defaults to the built-in
testsrc's 320×240p25; pass `--width/--height/--fps` (or `AVP_WIDTH`,
`AVP_HEIGHT`, `AVP_FPS`) to change it — the writer rescales to match,
because packed v210 carries no dimensions and the reader derives the
row stride from `--width`.

Defaults to publishing an `lavfi testsrc` pattern; override with
`-e AVP_INPUT=/media/your-file.mp4` for a real file (played back
looped and realtime-paced through `InputRec`).

Runs both graphs in one process. Stop it with `docker kill` (see Known
gaps — Ctrl-C hangs in teardown). The output stays playable anyway:
the reader-side output sets `movflags=frag_keyframe+empty_moov+
default_base_moof` *and* `flush_packets=1`. The second one is what
makes the promise real — without it the muxer's 256 KiB avio buffer is
written out only when it fills, so a low-bitrate run that is killed
rather than closed leaves a 28-byte file containing just the `ftyp`
box.

Split writer and reader across two containers by passing
`--writer-only` / `--reader-only` and sharing the flow UUID via
`AVP_MXL_VIDEO_ID`.

## Codec choices

MXL flows carry uncompressed frames. The muxer registered by
`8/0011-*.patch` insists on `v210` (10-bit 4:2:2 packed) for video —
`rawvideo` is rejected at header write. The GPU reader path encodes
H.264 with NVENC; the CPU fallback uses `mpeg4` (rather than H.264)
because the image does not link `libx264`.

NVENC is given `g=<fps>` — one keyframe per second instead of its
default 250-frame GOP — because `frag_keyframe` cuts a fragment per
keyframe, and that is what makes the growing file playable a second in
rather than ten.

The GPU path converts to 8-bit nv12 before NVENC even though the flow
is 10-bit 4:2:2, because NVENC only accepts 4:2:2 10-bit on
Blackwell-class hardware. Change `scale_cuda=format=nv12` to `p010le`
plus `hevc_nvenc`/`main10` to keep 10 bits through the encoder.

## Timing and grain indices

MXL is a wall-clock transport, and the muxer ignores PTS entirely: it
takes its own grain index from `mxlGetCurrentIndex` at header write and
increments it once per packet. So the *writer* must be paced, or grain
N lands in shared memory long before wall-clock N/fps. An unpaced lavfi
source published ~2300 grains/s here, which ran the flow seconds into
the future and collapsed the ring's usable history to milliseconds —
every reader was then "too late". Hence `realtime(set_pts=1)` followed
by `force_fps` between the writer's decoder and the v210 encoder.

Three demuxer options matter on the reader:

* `grain_index_init=head` — start at the newest grain. `tail` hangs
  even plain `ffmpeg` on this build.
* `on_too_late=reset` — `avformat_open_input`'s probe already reads a
  grain, so the index is chosen when the node is *created*, several
  seconds before the group starts (CUDA and NVENC init sit in between).
  By then it is behind the ring tail. Without `reset` the demuxer
  returns `EAGAIN`, which `input.cpp` treats as fatal, and the node
  fails with "Resource temporarily unavailable".
* `reset_on_drop=1` — resetting the index leaves a hole where the
  skipped grains would have been (a 5.8 s leading PTS gap, and a
  container frame rate of 19.4 fps instead of 25). This rebases the
  timestamps after the reset.

## Zero-copy grains

`zero_copy=1` (default; `--no-zero-copy` opts out) makes the demuxer
wrap the grain payload in an `AVBufferRef` pointing into the shared
memory ring rather than `memcpy`-ing it into a fresh packet. The
release callback in the patch is a no-op — **nothing holds a reference
on the grain** — so the bytes are valid only until the writer laps that
slot in the ring. Two things keep that safe here:

* `queue.plan_capacity r_vpkt 1` bounds the packet queue to a single
  in-flight grain, so a packet cannot sit and age behind others.
* `history_duration` in `options.json` sizes the ring; 1 s (25 grains at
  25 fps) gives a large margin over that one queued packet.

A reader that stalls for longer than the ring depth will silently see
overwritten pixels rather than an error, which is why the FFmpeg option
is marked experimental. `--no-zero-copy` trades one memcpy per frame
for immunity.

Note that zero-copy removes the shm→packet copy only. `v210_to_cuda`
still stages each grain through pinned host memory on its way to the
GPU; making that leg copy-free would mean `cudaHostRegister`-ing the MXL
ring, which is a change to the node in `src/`.

## Known gaps

* The reader is not paced: it runs at wall-clock max. Adding a
  `realtime` node (as the writer has) before the encoder would cap it
  to the flow frame rate. Running
  uncapped is harmless in itself — `blocking=1` parks the reader on the
  writer's head — it just spends CPU on `blocking` waits.
* **Ctrl-C does not shut the demo down.** SIGINT leaves the process
  stuck in teardown: every node thread is alive in a timed poll and the
  main thread waits in `futex_do_wait`. This reproduces on the GPU and
  CPU reader paths alike and predates this demo's reader wiring, so it
  looks like a framework-level stop-ordering issue rather than anything
  MXL-specific. Use `docker kill` / SIGKILL meanwhile.
* Restarting the reader group has the same shape of problem: it parks at
  "Stopping node r_unpack ...". `auto_restart` is therefore `off` on
  every reader node except `r_input`.
* With a looped file source, each 5 s wraparound logs `EventLoop
  negative time to wait, resyncing` on the writer and a couple of
  dropped frames in `force_fps`, and the reader sees a burst of "too
  early" waits. Output is unaffected, but a live flow would rather have
  a seamless looper.
* The write path still goes through swscale and the CPU `v210` encoder,
  because nothing packs CUDA frames back to v210 yet. The read path no
  longer does (with `--gpu-unpack`).

## References

* MXL SDK: <https://github.com/dmf-mxl/mxl>
* MXL FFmpeg fork (source for the 0011 patch, branch `dmf-mxl/8.1`): <https://github.com/cbcrc/FFmpeg>
* Reference build guidance: <https://github.com/cbcrc/guidance-for-building-ffmpeg-with-mxl>
