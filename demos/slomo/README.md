# Slomo demo

Frame-rate multiplication and slow-motion for 1080p video, via two
interpolators avplumber ships:

- **`nvof_fruc`** — NVIDIA Optical Flow FRUC. Hardware optical flow +
  warp. Fast, no ML dependencies, but visibly poor on disocclusion
  regions (revealed background behind moving objects), sharp object
  boundaries, and phone-quality footage with baked-in motion blur.
- **`rife_vfi`** — RIFE 4.26 (Practical-RIFE checkpoint, TensorRT
  fp16). Learned VFI with occlusion masks + refinement network.
  Dramatically better quality; ~3× the per-interp compute of FRUC.

## Prerequisites

- NVIDIA GPU, Turing+ (T4/RTX 20xx or newer). Tested on RTX 4000 Ada.
- CUDA 12.x/13.x, NVIDIA driver 570+.
- For `nvof_fruc`: NVIDIA Optical Flow SDK 5.0.7 (or newer). Unpack the
  `NvOFFRUC/` subtree into `deps/Optical_Flow_SDK_5.0.7/NvOFFRUC/`; the
  headers-only dense-flow SDK is already in `deps/`.
- For `rife_vfi`: TensorRT 10+ C++ SDK and a pre-built engine. See
  [`src/nodes/neural_net/rife/tools/README.md`](../../src/nodes/neural_net/rife/tools/README.md)
  for the ONNX export and engine build.
- Build with `HAVE_CUDA=1 HAVE_NVCC=1 HAVE_NVOF_FRUC=1 HAVE_TENSORRT=1
  NEURAL_NET_COMMON=1` (see [`build.sh`](../../build.sh) on a working
  host for the full set of flags with library paths).

## Example scripts

All under `examples/` at the repo root. Edit each to point at your
input file and (for RIFE) your `.engine` path.

| Script | Interpolator | What it does | Perf on RTX 4000 Ada |
|---|---|---|---|
| `super_slomo.avplumber` | `nvof_fruc` factor=4 | 4× slomo at 60 fps CFR (with duplication artifact) | ~1.05× realtime |
| `super_slomo_cascade.avplumber` | `nvof_fruc` 2×2×2 cascade | 4× slomo at 60 fps CFR (recommended FRUC) | ~1.05× realtime |
| `rife_60fps.avplumber` | `rife_vfi` factor=2 | Frame doubling (30→60 fps), no slomo | ~1.8× input realtime |
| `rife_2x_slomo.avplumber` | `rife_vfi` factor=4 | 2× slomo at 60 fps CFR | 0.75× input, 1.48× output realtime |
| `super_slomo_rife.avplumber` | `rife_vfi` factor=8 | 4× slomo at 60 fps CFR | 0.30× input, 1.17× output realtime |

## The sizing formula that matters

`force_fps` is for CFR normalization, **not** for making up frames. If
you undersize `factor`, `force_fps` will silently pad the output with
duplicated frames and the result looks like judder ("one step forward,
half step back"). Correct formula:

    factor >= ceil(target_output_fps * slomo_factor / source_fps)

For a 29.25 fps source at 60 fps output:

| slomo | required factor |
|---|---|
| 1× (no slomo) | 2 |
| 2× | 5 (use 4 — the 2.6% residual is fine) |
| 4× | 9 (use 8) |
| 8× | 17 (use 16) |

Measured `force_fps` duplication with correct sizing: **~2.6%**
(residual jitter only). With `factor=4` at 60 fps for 2× slomo (was
undersized): **51%** — every source frame duplicated, hence judder.

## RIFE quality notes

- Single-node `factor > 2` works correctly on RIFE 4.26 **only with a
  fixed warp shim during ONNX export**. Measured on a real 29.25 fps
  clip, frame 109→110 (a heavy-motion camera pan):
  - Broken warp: step-size ratio 4.5× → non-uniform motion
  - Fixed warp: step-size ratio 1.04× → essentially linear
  See `src/nodes/neural_net/rife/tools/export_onnx_fp16.py` for the
  shim; it's not obvious.
- Fp16 quality on Ada matches fp32 within noise; fp32 costs 2× memory
  for no benefit.
- 1080p engine is fixed at 1088×1920 (padded to a multiple of 32).
  Other resolutions need a re-export with the matching H/W.

## Compute cost

Per interpolated frame at 1080p on RTX 4000 Ada:
- `nvof_fruc`: ~4.6 ms
- `rife_vfi` (fp16 TRT): ~15 ms

Linear scaling with `factor` on RIFE; FRUC's per-interp cost is
slightly lower for higher factors because the hardware pipelines are
shared across intermediate samples.
