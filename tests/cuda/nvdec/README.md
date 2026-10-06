# NVDEC and CUarray checks

Run on an NVIDIA host with the Python module built against the selected FFmpeg
libraries. Set `PYTHONPATH` to the checkout/module location and select the matching
libraries with the build's rpath or `LD_LIBRARY_PATH`. Keep the same CUDA, DRM/GL
and optional neural feature flags between binary and Python-module builds.
The CUarray tests require the pinned `deps/ffmpeg/9` series and its codec headers.
Use a separate FFmpeg 8.1 build for the linear/CPU baseline.

The scripts accept private fixture/report paths at runtime. Pixel checks download
only at the explicit verification boundary; they are not production graphs or
performance benchmarks. Retention checks hold GPU references without pixel reads.

| Script | Check |
|---|---|
| `probe.py` | Finite CPU/CUDA/CUarray decode; normalized pixel hashes, PTS, EOF, retained references and shutdown watchdog. Use `--reference` to compare saved reports across builds. |
| `compositor_decode.py` | Real NVDEC through identity normalization and two compositors; array-handle passthrough, pixels/PTS and repeated teardown. `--filter-linear` inserts pad/crop to test ordering on linear frames from a private producer stream. `--rounds 0` works on FFmpeg 8.1. |
| `mixed_decode.py` | One shared device with HEVC CUarray plus H264/AV1 linear CUDA, compared against an all-linear canvas. Requires all three fixture paths. |
| `camera_motion.py` | NV12 CUarray versus linear camera-motion metadata/PTS, original frame handles, private producer streams and repeated teardown. Requires `HAVE_NVOF=1`; the default GPU IRLS check also needs `HAVE_NVCC=1`. |
| `luma_diff.py` | Linear/CUarray scores against a CPU oracle at lookahead 0/2/4/16, ring wrap, EOF tails and source-handle preservation. Both metric scripts support `--linear-only --report` and `--reference` for before/after linear checks. |
| `compositor_arrays.cu` | Synthetic NV12/P010/P210 arrays against linear compositor output, including crops, resizing and mixed layers. The build command is at the top of the file. |
| `filters.py` | Synthetic pad/crop/transition pixel matrix. `--clip` adds real HEVC decode cases; `--decode-only` limits the run to those cases. |
| `capture.py` | Whole-GPU, per-thread CPU and mixer counter deltas on a running show. `--control-port` additionally proves decoded/paced/encoded edge progress. It records evidence rather than applying arbitrary pass thresholds. |
| `aux_cycles.py` | Changes subscriptions on extra AUX buses advertising source pages, waits for applied compositions, then restores layouts/pages and checks scene assignments. Pair with `capture.py`; use only an exclusively controlled test show. |

For example, with a 50-frame 1080p NV12 HEVC fixture and a writable report directory:

```sh
python3 tests/cuda/nvdec/probe.py "$clip" --mode cuda --expected-frames 50 --output "$reports/linear.json"
python3 tests/cuda/nvdec/probe.py "$clip" --mode cuarray --expected-frames 50 --reference "$reports/linear.json" --output "$reports/array.json"
python3 tests/cuda/nvdec/probe.py "$clip" --mode cuarray --action hold --hold 8 --expected-frames 50 --output "$reports/held.json"
python3 tests/cuda/nvdec/compositor_decode.py --input "$clip" --report "$reports/compositor.json"
python3 tests/cuda/nvdec/camera_motion.py --input "$clip" --report "$reports/camera-motion.json"
python3 tests/cuda/nvdec/luma_diff.py --input "$clip" --report "$reports/luma-diff.json"
python3 tests/cuda/nvdec/filters.py --clip "$clip" --output "$reports/filters.json"
```

Use `--sw-format p010le` for Main10 probe fixtures. Different codecs and SPS
requirements allocate different base pools. The finite probes' default three
extra surfaces are not a full-mixer sizing recommendation. No timing comparison
is valid if inputs stalled, output counts changed, compilation overlapped, or
the two builds silently loaded different libraries than intended.

For the measured workload, failures and remaining replay/Blackwell boundaries,
see the [capacity report](../../../doc/research/2026-10-03-cuarray-capacity.md).
