# MXL demo

Publishes and consumes flows on the
[MXL](https://github.com/dmf-mxl/mxl) shared-memory transport through
avplumber's generic `input` / `output` nodes with `format="mxl"` — no
MXL-specific node types.

Two graphs run in one process:

* **writer** — `lavfi testsrc` → decode → `realtime` + `force_fps` →
  rescale to yuv422p10le → `v210` encode (10-bit 4:2:2 uncompressed, per
  SMPTE ST 2110) → mux → `output(format="mxl", url="/dev/shm/mxl")`. The
  MXL muxer takes a plain filesystem path.
* **reader** — `input(format="mxl", url="mxl:///dev/shm/mxl?id=<uuid>")`
  → demux → v210 unpack → encode → fragmented mp4. `--gpu-unpack`
  (default `auto`) picks the unpack:
  * **GPU** — `v210_to_cuda` copies each grain once into pinned memory
    and unpacks it to a CUDA `p210le` frame with a PTX kernel;
    `scale_cuda` and `h264_nvenc` finish the file. No CPU codec, no
    swscale.
  * **CPU** (no CUDA device) — libavcodec `v210` decoder, swscale to
    yuv420p, `mpeg4`.

Grains are taken zero-copy by default — see [below](#zero-copy-grains).

Verified end-to-end on x86_64 Fedora with an RTX 4000 Ada (driver
615.71, Docker + NVIDIA Container Toolkit) against FFmpeg
`n8.1-12-g93aafbb` from this patch stack: GPU zero-copy, GPU with
`--no-zero-copy`, `--gpu-unpack off`, from both `lavfi testsrc` and a
looped file, at 320×240p25 and 640×480p25. All started at PTS 0,
reported 25.0 fps (25.06 on NVENC) and decoded back to the source
pattern.

## Requirements

* Linux x86_64 — the MXL SDK is Linux-only.
* Docker with a shared `/dev/shm` (the default size is enough).
* NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  are optional here (`--gpus all` selects the GPU reader path), though
  the other demos in the shared image do need CUDA.
* The FFmpeg 8.x series applied to `n8.1` (or `n8.0`). The MXL patch
  `8/0010-*` is generated from `cbcrc/FFmpeg` branch `dmf-mxl/8.1` —
  see [`deps/ffmpeg/README.md`](../../deps/ffmpeg/README.md).

## Build

Shares its runtime image with `demos/mixer/`. From the repository root
(after `git submodule update --init --recursive`):

```sh
docker build -f demos/mixer/Dockerfile -t avplumber-mixer:local .
```

MXL-specific parts of that build: gcc-13 from `ppa:ubuntu-toolchain-r/test`
(the SDK needs C++20; FFmpeg and avplumber keep gcc-11), vcpkg and Rust
for the SDK's dependencies, the MXL SDK (`dmf-mxl/mxl` @ `v1.1.0`) into
`/usr/local` with the private `Requires` scrubbed from its `.pc` file
(libmxl links spdlog statically), and FFmpeg configured with
`--enable-libmxl --enable-demuxer=mxl --enable-muxer=mxl
--enable-protocol=mxl`. The build fails if the mxl demuxer and muxer are
not registered afterwards.

## Run

The demo needs a `/dev/shm/mxl` domain directory holding an
`options.json`. On a single host use `--ipc=host`:

```sh
mkdir -p /dev/shm/mxl
# 1 s of history. The 100 ms the MXL samples use is ~2 grains at 25 fps,
# which leaves zero-copy readers no slack at all.
echo '{"urn:x-mxl:option:history_duration/v1.0": 1000000000}' \
  > /dev/shm/mxl/options.json

docker run --rm --ipc=host --gpus all \
    --entrypoint python3 \
    -v /dev/shm/mxl:/dev/shm/mxl \
    -v "$PWD/demos/mxl/test-media:/media" \
    -e AVP_OUTPUT=/media/out.mp4 \
    avplumber-mixer:local /build/demos/mxl/mxl_demo.py
```

Drop `--gpus all` to take the CPU reader path; the demo prints which one
it picked. Geometry defaults to the built-in testsrc's 320×240p25;
`--width/--height/--fps` (or `AVP_WIDTH`, `AVP_HEIGHT`, `AVP_FPS`)
change it — the writer rescales to match, because packed v210 carries no
dimensions and the reader derives its row stride from `--width`.
`-e AVP_INPUT=/media/your-file.mp4` publishes a file instead of the
testsrc (looped and realtime-paced through `InputRec`).
`--writer-only` / `--reader-only` plus a shared `AVP_MXL_VIDEO_ID` split
the two halves across containers.

`--bench-seconds N` samples `enqueued_total` on the writer and reader
encoder edges once a second, prints per-second fps plus a summary, and
exits without teardown. `--writer-pace off` drops `realtime` +
`force_fps` so the writer publishes as fast as it can — throughput
measurement only, since it starves readers of history.

Stop the demo with `docker kill` — Ctrl-C hangs in teardown (see
[Known gaps](#known-gaps)). The output stays playable regardless: the
reader's output sets `movflags=frag_keyframe+empty_moov+default_base_moof`
*and* `flush_packets=1`. The second one is what makes the promise real —
without it the muxer's 256 KiB avio buffer is written only when it
fills, so a killed low-bitrate run leaves a 28-byte file holding just
the `ftyp` box.

## Measured throughput

1920×1080p59.94 `smptehdbars`, 1 s of ring history, on the host above
(4 vCPU of an EPYC 9474F, RTX 4000 Ada), 60 s per case after a 10 s
warmup, via `--bench-seconds`:

| case | writer | reader | CPU | SM | NVENC |
|---|---|---|---|---|---|
| GPU unpack, zero-copy | 59.94 fps | 59.94 fps | 0.27 cores | 2% | 5% |
| GPU unpack, `--no-zero-copy` | 59.94 fps | 59.94 fps | 0.28 cores | 2% | 5% |
| CPU unpack, zero-copy | 59.94 fps | 59.94 fps | 0.51 cores | — | — |
| `--writer-pace off`, GPU, zero-copy | 507 fps | 507 fps | 1.71 cores | 16% | 42% |
| `--writer-pace off`, GPU, `--no-zero-copy` | 483 fps | 483 fps | 1.80 cores | 15% | 40% |
| `--writer-pace off`, CPU unpack | 419 fps | 192 fps | 3.05 cores | — | — |
| `--writer-pace off`, `--writer-only` | 567 fps | — | 1.33 cores | — | — |

Realtime 1080p59.94 is not demanding: about a tenth of a core per side
and an idle GPU. The writer — swscale to yuv422p10le plus the CPU `v210`
encoder — reaches 567 fps (3.0 GB/s into `/dev/shm`) on its own, and the
GPU reader keeps up with it, so unpaced both sides settle at 507 fps on
1.7 of the 4 vCPU. Zero-copy buys 5% throughput at that rate for less
CPU; the unpack itself matters more, halving realtime CPU (0.51 → 0.27
cores) and, unpaced, capping the CPU reader at 192 fps against the
writer's 419 while saturating all four cores.

## Codec choices

MXL flows carry uncompressed frames, and the muxer insists on `v210` for
video — `rawvideo` is rejected at header write. The CPU reader encodes
`mpeg4` rather than H.264 because the image does not link `libx264`.

NVENC gets `g=<fps>`, one keyframe per second instead of its default
250-frame GOP, because `frag_keyframe` cuts a fragment per keyframe: the
growing file becomes playable a second in rather than ten.

The GPU path converts to 8-bit nv12 before NVENC even though the flow is
10-bit 4:2:2, because NVENC only accepts 4:2:2 10-bit on Blackwell-class
hardware. Change `scale_cuda=format=nv12` to `p010le` plus
`hevc_nvenc`/`main10` to keep 10 bits through the encoder.

## Timing and grain indices

MXL is a wall-clock transport and the muxer ignores PTS: it takes its
grain index from `mxlGetCurrentIndex` at header write and increments it
per packet. An unpaced writer therefore runs the flow into the future —
lavfi published ~2300 grains/s here, collapsing the ring's usable
history to milliseconds so that every reader was "too late". Hence
`realtime(set_pts=1)` and `force_fps` ahead of the writer's v210 encoder.

Reader-side demuxer options:

* `blocking=1` — wait up to a frame period for the next grain instead of
  returning `EAGAIN` immediately.
* `grain_index_init=head` — start at the newest grain. `tail` hangs even
  plain `ffmpeg` on this build.
* `on_too_late=reset` — `avformat_open_input`'s probe reads a grain, so
  the index is chosen at node *creation*, seconds before the group
  starts (CUDA and NVENC init sit in between), by which time it is behind
  the ring tail. Without `reset` the demuxer returns `EAGAIN`, which
  `input.cpp` treats as fatal.
* `reset_on_drop=1` — rebases timestamps after such a reset, which
  otherwise leaves a hole where the skipped grains would have been (a
  5.8 s leading PTS gap, and 19.4 fps in the container instead of 25).
  The fork rebased onto the resumed grain, which is right at startup but
  sends the PTS back to 0 on a *mid-stream* reset, tripping the
  monotonic-PTS `av_assert0` a few lines below — reproducible in seconds
  with `--writer-pace off --gpu-unpack off`, where the reader cannot keep
  up with the flood. Patch `0010` now continues from the last delivered
  PTS instead (see `deps/ffmpeg/mkpatch-fixups/`).

## Zero-copy grains

`zero_copy=1` (default; `--no-zero-copy` opts out) wraps the grain
payload in an `AVBufferRef` pointing into the shared-memory ring instead
of `memcpy`-ing it into a fresh packet. The patch's release callback is a
no-op — **nothing holds a reference on the grain** — so the bytes are
valid only until the writer laps that slot. Two things keep it safe here:
`queue.plan_capacity r_vpkt 1` bounds the packet queue to one in-flight
grain, and `history_duration` of 1 s (25 grains) leaves a wide margin
over that. A reader stalled longer than the ring depth silently sees
overwritten pixels rather than an error, which is why the FFmpeg option
is marked experimental.

Zero-copy removes the shm→packet copy only: `v210_to_cuda` still stages
each grain through pinned host memory. Making that leg copy-free would
mean `cudaHostRegister`-ing the MXL ring, i.e. a change to the node in
`src/`.

## Known gaps

* **Ctrl-C does not shut the demo down.** SIGINT leaves every node
  thread in a timed poll and the main thread in `futex_do_wait`. Same on
  the GPU and CPU paths, and it predates this demo's reader wiring, so it
  looks like framework-level stop ordering rather than anything
  MXL-specific. Restarting the reader group parks the same way, at
  "Stopping node r_unpack ..." — hence `auto_restart` is `off` on every
  reader node except `r_input`. Use `docker kill` meanwhile.
* The reader is unpaced and runs at wall-clock max. Harmless — `blocking=1`
  parks it on the writer's head — it just spends CPU waiting. A
  `realtime` node before the encoder would cap it.
* With a looped file source, each wraparound logs `EventLoop negative
  time to wait, resyncing` on the writer, drops a couple of frames in
  `force_fps`, and gives the reader a burst of "too early" waits. Output
  is unaffected, but a live flow would want a seamless looper.
* The write path still goes through swscale and the CPU `v210` encoder;
  nothing packs CUDA frames back to v210 yet.

## References

* MXL SDK: <https://github.com/dmf-mxl/mxl>
* MXL FFmpeg fork (source for patch 0010, branch `dmf-mxl/8.1`): <https://github.com/cbcrc/FFmpeg>
* Reference build guidance: <https://github.com/cbcrc/guidance-for-building-ffmpeg-with-mxl>
