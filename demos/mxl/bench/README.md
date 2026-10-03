# Measuring the MXL demo

The scripts behind every number in [`../README.md`](../README.md). They are
all wrappers around `mxl_demo.py --bench-seconds`; what they add is a clean
MXL domain per case, the host-side sampling the demo cannot do from inside
its own process, and a digest of the run instead of a log to read.

Linux with Docker and the shared demo image (`avplumber-mixer:local`, see
[Build](../README.md#build)); an NVIDIA GPU and the container toolkit for
everything except the `nodecpu` group's CPU-only cases. `nvidia-smi` on the
host is optional — without it the GPU block is simply absent.

## Two ways to reach the demo

```sh
demos/mxl/bench/cases.sh throughput                   # a fresh container per case
demos/mxl/bench/cases.sh throughput --exec avpbuild   # a container that is already up
```

The default (`--image`, run mode) starts one container per case. It is the
only mode whose cgroup holds nothing but the run, so the CPU figure is
trustworthy; it can also drop the GPU (`--no-gpu`) and bind this checkout's
`demos/mxl` over the image's copy (`--mount-demos`).

`--exec CONTAINER` runs `docker exec` in a container that is already up —
the incremental build loop, where `/build` is a tree you keep rebuilding in
place. Per-node CPU is just as good there, because the demo measures its own
threads through `/proc/self/task`; whole-container CPU is not, and `--no-gpu`
cannot take a GPU away from a running container. The container needs
`--ipc=host` so that it sees the domain the harness creates in `/dev/shm`.

## Quick start

```sh
demos/mxl/bench/cases.sh list
demos/mxl/bench/cases.sh smoke          # does every graph shape still run?
demos/mxl/bench/cases.sh pack           # the GPU v210 pack, paced and unpaced
demos/mxl/bench/pack_bitexact.sh        # the pack against libavcodec's, byte for byte
demos/mxl/bench/pixel_compare.sh psnr   # the GPU conversion against swscale

# One case, any demo flag, geometry and window of your own:
demos/mxl/bench/bench.sh --label mine --seconds 20 \
    --writer-only --writer-pace off --gpu-scale writer --writer-pack gpu
```

Anything `bench.sh` and `cases.sh` do not recognise goes to `mxl_demo.py`, and
arguments after a group name are appended to every case in it — so
`cases.sh pack --seconds 10 --no-sample` is a quick pass over the same cases.

## What each group reproduces

| group | cases | window | in [`../README.md`](../README.md) |
|---|---|---|---|
| `throughput` | 11 | 60 s | the [Measured throughput](../README.md#measured-throughput) table |
| `nodecpu` | 6 | 30 s | the [Where the CPU goes](../README.md#where-the-cpu-goes) table |
| `conversion` | 20 | 40 s | the three [Conversion on the GPU](../README.md#conversion-on-the-gpu) tables |
| `pack` | 7 | 30 s | the [Packing on the GPU](../README.md#packing-on-the-gpu) table and its round trips |
| `zerocopy` | 6 | 25 s | the two `r_input` rows, hence [Zero-copy grains](../README.md#zero-copy-grains) |
| `swsflags` | 7 | 30 s | the `--sws-flags` figures under "Where the CPU goes" |
| `srcfmt` | 4 | 30 s | that a source already in 10-bit 4:2:2 costs the writer no swscale |
| `tails` | 5 | 30 s | the unpaced ladder: which leg caps the reader |
| `smoke` | 11 | 14 s | nothing — the cases only have to build and run |

`throughput` and `conversion` take 15–20 minutes each; the rest are a few
minutes. Start from `smoke` after touching the graph: 14 s windows make
worthless numbers, but a shape that no longer builds shows up immediately.

The remaining two scripts answer questions an fps number cannot:

* `pack_bitexact.sh` publishes the same converted frames with the CPU and the
  GPU pack, dumps the grains with `ffmpeg -c copy` and compares them byte for
  byte, at 1920×1080, 1280×720 and 1918×1080 — the last two ending each row
  mid-block. Exits nonzero on any difference, so it works as a gate.
* `pixel_compare.sh` has three sections: `psnr` (swscale against `scale_cuda`
  on the writer, read back through the CPU reader both times), `pcie`
  (`nvidia-smi dmon` over a paced run, for the SM and PCIe columns) and
  `formats` (which pixel formats this build's `scale_cuda` accepts).

## Reading a case

```
=== gpu-zc-paced (exit 0, wall 71.3s) args: --gpu-unpack on
bench: target 60 fps, 1920x1080, pace on, ...
bench: writer: mean 59.94 fps (min 59.88, max 60.01) over 50 s, 100.0% of target, ...
bench: per-node CPU over 50.0 s, 13.4 s total = 0.27 cores
bench:   r_enc            8.20 s,  0.16 cores,   3.58 ms/frame
--- log counts
  too late: 3
--- gpu (mean/max)
  sm 2.0%/4%  enc 5.1%/7%  gpumem 412/418 MiB
--- cpu (whole container)
  0.27 cores mean over 70 s
--- output
  width=1920 height=1080 pix_fmt=yuv420p nb_read_frames=2996 avg_frame_rate=60000/1001
--- log: /tmp/mxl-bench/gpu-zc-paced.log
```

The `bench:` lines are the demo's own summary (the per-second rows stay in the
log). `log counts` is the triage: `too late` and `too early` grains are
expected in the unpaced cases, which publish faster than any reader can
follow, and suspicious anywhere else. Two entries are normal everywhere — a
handful of `setting edge more than once` while the graph is built, and the two
`libcuda` lines a `--no-gpu` container prints on the way to its CPU path. The
two sampler blocks are the host's
view — mean/max GPU utilisation, and cumulative cgroup CPU time over a known
window, which is an exact mean core count where `docker stats` percentages are
sampled too coarsely to trust.

Then the caveats worth remembering:

* Paced numbers only mean something on an otherwise idle host. One SM figure
  in `../README.md` had to be re-measured because a leftover run from the
  previous case was still on the GPU.
* The CPU block needs cgroup v2 with the systemd driver (`docker-<id>.scope`);
  elsewhere it is silently absent, and the per-node table from the demo is
  the better figure anyway.
* Every case wipes the domain first, because a stale flow directory keeps the
  old geometry and grain indices. The harness refuses to wipe a domain some
  other container has bind-mounted — deleting it would leave that writer
  publishing into an unlinked directory no reader can find.
* `--domain` defaults to `/dev/shm/mxlbench`, away from the demo's own
  `/dev/shm/mxl`, so a bench run does not disturb one you are watching.

## Files

| file | |
|---|---|
| `bench.sh` | one case: fresh domain, samplers, run, digest |
| `cases.sh` | the groups above, as lists of `bench.sh` invocations |
| `pack_bitexact.sh` | GPU pack against the libavcodec `v210` encoder |
| `pixel_compare.sh` | PSNR/SSIM, PCIe and SM cost, `scale_cuda` formats |
| `lib.sh` | shared: container modes, domain reset, samplers, digest |
