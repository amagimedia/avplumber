# MXL demo

Publishes and consumes flows on the
[MXL](https://github.com/dmf-mxl/mxl) shared-memory transport through
avplumber's generic `input` / `output` nodes with `format="mxl"` — no
MXL-specific node types.

Two graphs run in one process:

* **writer** — `lavfi testsrc` → decode → `realtime` + `force_fps` →
  rescale to yuv422p10le → `v210` encode (10-bit 4:2:2 uncompressed, per
  SMPTE ST 2110) → mux → `output(format="mxl", url="/dev/shm/mxl")`. The
  MXL muxer takes a plain filesystem path. `--writer-pack gpu` converts
  *and* packs on the GPU instead, with `cuda_to_v210` in place of the
  encoder — see [Packing on the GPU](#packing-on-the-gpu).
* **reader** — `input(format="mxl", url="mxl:///dev/shm/mxl?id=<uuid>")`
  → demux → v210 unpack → encode → fragmented mp4. `--gpu-unpack`
  (default `auto`) picks the unpack:
  * **GPU** — `v210_to_cuda` copies each grain once into pinned memory
    and unpacks it to a CUDA `p210le` frame with a PTX kernel;
    `scale_cuda` and `h264_nvenc` finish the file. No CPU codec, no
    swscale.
  * **CPU** (no CUDA device) — libavcodec `v210` decoder, swscale to
    yuv420p, `mpeg4`.

  The two ends are independent: `--reader-encoder` picks the encoder
  whatever the unpack was, and `--gpu-scale` moves the conversion between
  them onto the GPU on either leg — see
  [Conversion on the GPU](#conversion-on-the-gpu).

Grains are taken zero-copy by default — see [below](#zero-copy-grains).

Verified end-to-end on x86_64 Fedora with an RTX 4000 Ada (driver
615.71, Docker + NVIDIA Container Toolkit) against FFmpeg
`n8.1-12-g93aafbb` from this patch stack: GPU zero-copy, GPU with
`--no-zero-copy`, `--gpu-unpack off`, from both `lavfi testsrc` and a
looped file, at 320×240p25 and 640×480p25. All started at PTS 0,
reported 25.0 fps (25.06 on NVENC) and decoded back to the source
pattern. `--gpu-scale` and `--writer-pack gpu` were checked separately at
1920×1080p59.94 — see [Conversion on the GPU](#conversion-on-the-gpu) and
[Packing on the GPU](#packing-on-the-gpu).

## Requirements

* Linux x86_64 — the MXL SDK is Linux-only.
* Docker with a shared `/dev/shm` (the default size is enough).
* NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  are optional here (`--gpus all` selects the GPU reader path), though
  the other demos in the shared image do need CUDA.
* The FFmpeg 8.x series applied to `n8.1` (or `n8.0`). The MXL patch
  `8/0011-*` is generated from `cbcrc/FFmpeg` branch `dmf-mxl/8.1` —
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

The summary ends with a per-node CPU table. Every avplumber node runs in
a thread named after the node, so reading `/proc/self/task` before and
after the measured window and dividing by the frames that crossed the
last edge gives the CPU cost of each node in milliseconds per frame —
no profiler, and nothing on the media path.

`--reader-tail` shortens the reader for such measurements: `demux` drops
each grain as it leaves the demuxer, `unpack` drops the unpacked frame,
`scale` drops it after the conversion to the encoder's format, and the
default `encode` writes `--output`. The output encoder is the most
expensive node of the CPU reader by a factor of two, so with the full
tail a measurement of "the MXL reader" is mostly a measurement of mpeg4.

The conversion itself can be moved or redirected. `--sws-flags` picks the
swscale algorithm for the writer's and the CPU reader's `rescale_video`
nodes (`fast_bilinear`, `neighbor`, ...). `--gpu-scale
{writer,reader,both}` replaces those nodes with
`hwupload,scale_cuda,hwdownload`, and `--cuda-interp` is its
`--sws-flags`. `--reader-encoder {mpeg4,nvenc}` decouples the output
encoder from `--gpu-unpack`, so either unpack can feed either encoder —
which is what makes the upload and the download visible one at a time.
See [Conversion on the GPU](#conversion-on-the-gpu).

`--writer-pack {auto,cpu,gpu}` (or `AVP_WRITER_PACK`) picks the writer's
v210 pack the same way. `gpu` drops the `hwdownload` and the CPU `v210`
encoder for a single `cuda_to_v210` node, so it requires the conversion
on the GPU as well (`--gpu-scale writer` or `both`); `auto` turns it on
exactly when that holds. See [Packing on the GPU](#packing-on-the-gpu).

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
| `--writer-pace off`, `--writer-only`, `--sws-flags fast_bilinear` | 707 fps | — | 1.41 cores | — | — |
| `--writer-pace off`, `--writer-only`, `--gpu-scale writer` | 855 fps | — | 1.21 cores | 51% | — |
| `--writer-pace off`, `--writer-only`, `--gpu-scale writer --writer-pack gpu` | 1583 fps | — | 1.32 cores | 82% | — |
| `--writer-pace off`, `--gpu-scale writer --writer-pack gpu`, GPU unpack | 526 fps | 526 fps | 0.81 cores | 43% | 48% |

Realtime 1080p59.94 is not demanding: about a tenth of a core per side
and an idle GPU. The writer — swscale to yuv422p10le plus the CPU `v210`
encoder — reaches 567 fps (3.0 GB/s into `/dev/shm`) on its own, and the
GPU reader keeps up with it, so unpaced both sides settle at 507 fps on
1.7 of the 4 vCPU. Zero-copy buys 5% throughput at that rate for less
CPU. The GPU reader halves realtime CPU (0.51 → 0.27 cores) and, unpaced,
lets the reader follow the writer instead of capping at 192 fps against
its 419 while saturating all four cores — but that is the output encoder
talking, not the unpack; see below.

The last two rows are what this host gets to once neither end packs or
unpacks on the CPU: a writer alone at 1583 fps, and a round trip that
edges past the CPU-packed one (507 fps) on less than half its CPU. That
round trip saturates nothing measurable — 0.81 of four cores, 43% SM, 48%
NVENC — and lands where it landed before the pack moved, so its 526 fps is
a limit in the reader rather than a resource the writer is short of.

## Where the CPU goes

Per-node CPU from the same paced case (1080p59.94, 20 s window, two runs
agreeing within 2%), in milliseconds of CPU per frame:

| node | work | ms/frame |
|---|---|---|
| `r_enc` | `mpeg4` encoder, the demo's output | 3.58 |
| `r_scale` | swscale yuv422p10le → yuv420p | 1.79 |
| `w_scale` | swscale yuv420p → yuv422p10le | 1.67 |
| `r_unpack` | `v210_to_cuda`: stage into pinned host memory, PTX unpack | 0.40 |
| `r_dec` | `v210` decoder, the CPU unpack | 0.37 |
| `w_enc` | `v210` encoder, the pack | 0.30 |
| `w_output` | MXL muxer: `memcpy` into the opened grain | 0.27 |
| `r_input` | MXL demuxer, `--no-zero-copy` | 0.38 |
| `r_input` | MXL demuxer, `zero_copy=1` | 0.07 |

Everything else — demux, mux, `realtime`, `force_fps`,
`assume_video_format`, the mp4 output — stays under 0.1 ms/frame together.

So MXL itself is not where the time goes. Publishing costs one 5.5 MB
`memcpy` into the grain (0.27 ms), and reading costs nothing measurable
as long as `zero_copy` is on: 0.07 against 0.38 ms/frame, the difference
being exactly one more copy of the same 5.5 MB (three runs each way,
spread under 0.02 ms). Everything expensive is per-byte pixel work in
libswscale and libavcodec.

That reframes the GPU reader too. `v210_to_cuda` spends as much *host*
CPU as the libavcodec `v210` decoder it replaces (0.40 vs 0.37 ms/frame),
because it still stages every grain through pinned memory and waits on
the stream; its win is what comes after — no swscale, no CPU encoder.

The unpaced ladder makes the cap explicit. With `--reader-tail demux` the
reader follows the writer at 554 fps; adding the CPU unpack costs 38 fps
(516), and the GPU unpack lands in the same place (511) — in all three
cases the *writer* is the limit. Only the full tail caps the reader, at
195 fps, with `r_enc` pinned at exactly 1.00 core: mpeg4 gets no useful
thread parallelism here, so a quarter of this host is the ceiling.

Of the remaining per-frame cost, the swscale flag matters more than one
would expect for a conversion that does not resize. Left to
`rescale_video`'s default (area), the writer's 4:2:0 → 4:2:2 upsample
takes 1.67 ms/frame and the reader's 4:2:2 → 4:2:0 downsample 1.79;
`--sws-flags fast_bilinear` takes the writer's to 1.31 (and raises peak
writer throughput from 561 to 707 fps) while leaving the reader's alone,
and `--sws-flags neighbor` takes the reader's to 1.55 and the writer's to
1.57. Each is a chroma-quality trade, so the demo keeps the default and
leaves the choice to the flag.

Reproducing these numbers needs this branch's `rescale_video`: the node
parsed its `flags` parameter and then built the rescaler without it, so
every graph in the tree was scaling with the default algorithm. It also
now hands a frame straight through when it already has the requested
geometry and format, instead of paying swscale a full-frame copy — worth
1.0 ms/frame at 1080p 10-bit 4:2:2, which is what a writer fed by a
source that is already 10-bit 4:2:2 would otherwise burn for nothing.

## Conversion on the GPU

The writer's conversion is its largest cost and the reader's is second
only to the output encoder, so `--gpu-scale` hands them to `scale_cuda`.
What that saves depends entirely on how many times the frame has to cross
PCIe, which is why
`--reader-encoder` exists: naming the encoder independently of
`--gpu-unpack` puts the same conversion in front of an upload, a download,
both, or neither. 1080p59.94, paced, 30 s windows, host CPU of the
conversion node alone (`--reader-tail scale` where it is the reader's) —
the swscale rows repeat the previous section's measurement on the longer
window and land within 5% of it:

| conversion | host↔device per frame | ms/frame |
|---|---|---|
| swscale, writer 4:2:0 → 4:2:2 10-bit, area | — | 1.64 |
| swscale, same with `--sws-flags fast_bilinear` | — | 1.20 |
| swscale, reader 4:2:2 10-bit → 4:2:0, area | — | 1.73 |
| `scale_cuda`, either leg, host frames both sides | 11.4 MB | 0.61 |
| `scale_cuda`, reader, GPU unpack → mpeg4 | 3.1 MB down | 0.22 |
| `scale_cuda`, reader, GPU unpack → NVENC | none | 0.04 |

The line through those four GPU rows is 0.04 ms of fixed overhead plus
every transferred byte at about 19 GB/s — one host copy, near memcpy
speed. That is the price of unpinned frames: the driver stages each
transfer through its own pinned buffer, and the CPU pays for that copy
even though the DMA itself is free. The conversion adds nothing
measurable on top, which is what the 0.04 ms row says. With both legs
converting, `nvidia-smi dmon` reports 7% SM and 680 MB/s of PCIe traffic
in each direction — 11.4 MB per frame at 59.94 fps, i.e. exactly the two
transfers and nothing else. The all-GPU reader instead shows 289 MB/s in
(the v210 grains), 33 MB/s out (the H.264 stream), 2% SM and 6% NVENC.

So the GPU wins a conversion it can be handed without a round trip, and
wins less than it looks like on one where it pays both transfers: 0.61
against swscale's 1.2–1.7 ms/frame is a real saving, but a third of it
comes back elsewhere — with `--gpu-scale both` the mpeg4 encoder slows
from 3.66 to
4.16 ms/frame, its frames now arriving cold from a DMA rather than warm
from swscale. Paced 1080p59.94 round trips, total process CPU:

| round trip | cores |
|---|---|
| all CPU | 0.50 |
| CPU unpack, `--gpu-scale both` | 0.41 |
| GPU unpack → mpeg4 (one download) | 0.41 |
| GPU unpack → NVENC (no transfer) | 0.17 |
| the same plus `--writer-pack gpu` (no transfer either way) | 0.09 |

Unpaced, the same three shapes move the caps: all-CPU runs at 420 fps
writer / 200 fps reader on 3.14 cores, `--gpu-scale both` at 619 / 161 on
2.25 (the freed writer outruns the reader and takes memory bandwidth with
it), and GPU unpack into mpeg4 at 460 / 249 on 2.62 — the CPU reader
reaching its encoder ceiling instead of its swscale one.

Where the GPU is simply better is a conversion that also resizes, because
there the CPU cost does not come from moving bytes. Publishing a 1080p
source into a 720p flow, unpaced `--writer-only`:

| writer conversion | peak | ms/frame |
|---|---|---|
| swscale default (area) | 955 fps | 1.04 |
| `--sws-flags lanczos` | 314 fps | 3.18 |
| `--gpu-scale writer --cuda-interp lanczos` | 1435 fps | 0.36 |

A lanczos resize on the GPU costs a third of what an area resize costs on
the CPU, and its transfers are smaller than the 1080p case's because the
downloaded frame is 720p.

The pixels agree. Reading the same published flow back through the CPU
reader with the writer converting each way, the two output files differ
by PSNR y 90.3 dB, u 49.4, v 53.4 (average 55.7) and SSIM 0.9996: the
luma is untouched by either path at this geometry, and the chroma
difference is the two upsamplers' kernels rather than an error.
`smptehdbars` is static, so that comparison needed no frame alignment.

## Packing on the GPU

`--writer-pack gpu` removes the writer's last two CPU costs at once. The
new `cuda_to_v210` node packs the converted CUDA frame with a kernel and
DMAs the v210 bytes into a pinned buffer the muxer hands straight to the
grain, so both the `hwdownload` that ended `--gpu-scale writer` and the
libavcodec `v210` encoder behind it disappear. It *is* the encoder as far
as the muxer is concerned — it answers `IEncoder`, so no `enc_video`
belongs between it and `output`. Paced 1080p59.94, `--writer-only`, 30 s
windows, host CPU per frame — the CPU-pack column re-measures the two
sections above on that window and agrees with them within 0.04 ms:

| node | work | CPU pack | GPU pack |
|---|---|---|---|
| `w_scale` | `hwupload`, `scale_cuda`, and the `hwdownload` only the CPU pack needs | 0.62 | 0.21 |
| `w_enc` → `w_pack` | the v210 pack | 0.29 | 0.04 |
| `w_output` | MXL muxer: `memcpy` into the grain | 0.23 | 0.34 |
| whole process | every thread, `--writer-only` | 0.09 cores | 0.05 cores |

Two of those lines are the point. The conversion node loses exactly its
download — 0.62 → 0.21 ms/frame, and what is left is the 3.1 MB upload at
the 19 GB/s of the section above plus the fixed 0.04. And the pack itself
costs that fixed overhead alone: the 5.5 MB leaves the GPU by DMA into
pinned memory, so unlike `hwdownload` the CPU pays nothing per byte.

The third line is the same catch the mpeg4 encoder hit above, from the
other side: the muxer's `memcpy` gets more expensive (0.23 → 0.34
ms/frame) because it now reads bytes no CPU has touched. Even so the
writer falls from 1.15 to 0.59 ms/frame of host CPU, and the paced
writer process from 0.09 to 0.05 cores.

Unpaced the writer's peak nearly doubles, 855 → 1583 fps on the same
1.2–1.3 cores, and the GPU goes from 51% to 82% SM. `w_output` is then
0.42 ms/frame — 8.8 GB/s of `memcpy` into `/dev/shm` — which makes the
muxer's copy into the grain the writer's limit, the first time in this
demo that MXL itself is what caps anything.

The grains are bit-exact. Dumped with `ffmpeg -c copy` from otherwise
identical runs, the kernel's v210 is byte-for-byte what the libavcodec
encoder produces at 1920×1080, 1280×720 and 1918×1080 — the last two
exercising rows that end in a partial 6-pixel block, all three the row
padding to a 128-byte multiple. Round trips still come out decodable:
paced 30 s runs with the CPU reader (0.38 cores) and with the GPU reader
(0.09, against 0.17 for the same reader behind a CPU-packed writer) each
wrote 1800 frames at 60000/1001.

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
  up with the flood. Patch `0011` now continues from the last delivered
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

The copy it removes is worth 0.31 ms of CPU per frame at 1080p59.94 — see
[Where the CPU goes](#where-the-cpu-goes).

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
* `--writer-pack gpu` only helps a writer whose frames are already on the
  GPU, which is why it insists on `--gpu-scale`. A writer fed 10-bit 4:2:2
  frames that need no conversion still ends in the libavcodec encoder:
  uploading purely in order to pack would trade a 3.1 MB transfer for the
  0.29 ms/frame the CPU encoder costs, which is close to a wash.
* One copy is left on the write path — the muxer's `memcpy` into the
  opened grain, 0.34 ms/frame and the unpaced writer's actual cap. Getting
  rid of it means having the muxer hand out the grain's address before the
  packet is built, so the pack could DMA into the ring itself; that is a
  change to FFmpeg patch `0011`, and it mirrors the `cudaHostRegister`
  note under [Zero-copy grains](#zero-copy-grains) on the read side.

## References

* MXL SDK: <https://github.com/dmf-mxl/mxl>
* MXL FFmpeg fork (source for patch 0011, branch `dmf-mxl/8.1`): <https://github.com/cbcrc/FFmpeg>
* Reference build guidance: <https://github.com/cbcrc/guidance-for-building-ffmpeg-with-mxl>
