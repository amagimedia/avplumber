# court_calibration: real-time (30 fps+) port plan

Status: plan, 2026-06-11. The Python node (`pyplumber/court_calibration.py`)
is the validated prototype (M0); this documents where its time goes and how
to reach 30 fps+ in C++/CUDA (M1/M2 of `tactical_court_calibration_plan.md`).

2026-06-11 (later): the node was split into `pyplumber/court_segm/`
(geometry / evidence / matching / fitting / temporal) along exactly these
port boundaries, and the dead legacy pose-mode architecture (~540 lines)
was removed. The CUDA-bound code is now isolated in `court_segm/evidence.py`
(per-pixel mask work) and `court_segm/matching.py` (template GEMM);
`fitting.py` + `temporal.py` + the orchestrator stay CPU/Python per the
division-of-labor directive.

## Measured throughput (T4, bbl.mp4, full graph)

| build | fps | note |
|---|---|---|
| t53 (pre court-synthesis) | 8.9 | |
| t57+ (with court-synthesis + temporal filter) | 7.3 | synthesis ≈ −1.6 fps |

The graph baseline before the calibration rewrite was already ~8–9 fps:
**four TensorRT engines at 960×544 every frame on a T4 are a co-bottleneck.**
The calibration node alone will not deliver 30 fps; the inference budget needs
work too (engine consolidation, `infer_every_n: 2` + tracker interpolation,
FP16/INT8, or a bigger GPU).

## Where the Python node spends time (per wide frame)

Heavy per-pixel work — all 272×272 mask-scale, all CUDA-trivial, and the
masks ALREADY live on the GPU (`mask_gpu_every_n: 1`); we currently download
them and crunch in numpy:

1. `_reconcile_spillover`: projects the full 272² grid through the prior H
   (≈74k point-homography ops + analytic region tests).
2. `_clean_region_mask` + `_synth_tpl_from_court`: `ndimage.label`,
   `binary_fill_holes`, `binary_dilation` on 272² — runs up to twice per
   frame since the synthesis was added (the −1.6 fps).
3. `_tpl_boundary_points`: `skimage.find_contours` on 272².
4. Template scoring in `_lines_h`: 3191 templates × 5184 cells IoU matmul
   (≈17 M MAC) + a 27-render fine grid.
5. `_consensus_filter`: cKDTree queries over 1–3k boundary points.

Cheap math, expensive wrapper — the LM refines (`_refine_ptz`, promotion
pass, rescue, every-12th rig fit): 3–5 params × ~400 residuals is trivial
arithmetic, but scipy `least_squares` with Python residual callbacks costs
milliseconds per call and runs up to ~4×/frame. In C++/Eigen with analytic
Jacobians this is microseconds.

## Stage 0 (validated 2026-06-11): CuPy in the existing Python node

`cupy-cuda12x` + `nvidia-{cublas,curand,cuda-nvrtc,cuda-runtime}-cu12` pip
wheels install and run in the avp-builder container venv (T4). Measured:

| op | numpy today | CuPy measured |
|---|---|---|
| reconcile/synth per-pixel kernel (272², RawKernel) | ~1–2 ms each | **0.01 ms** |
| template IoU matmul 3191×5184 | several ms | **0.42 ms** |
| `cupyx.scipy.ndimage.label` 272² | ~1 ms | 1.4 ms (batchable) |

Division of labor (user directive: leave easy CPU work in Python): Python
keeps orchestration, LM fits, gates, temporal filter, consensus logic; CUDA
RawKernels take only per-pixel mask ops + template GEMM. RawKernel sources
are literal CUDA C — they port verbatim into the eventual C++ node. Masks
are re-uploaded (272²×N ≈ 0.6 MB/frame ≈ 0.1 ms) until pybind exposes the
GPU-side masks. fill_holes on GPU = invert → label → drop border-touching
components. Replace the KDTree segment assignment with closed-form distances
to the four analytic curves (faster on CPU too). Test stub:
`<path>/cupy_test.py`.

2026-06-11 (later still): K1-K4 implemented in `pyplumber/court_segm/cuda.py`
(CuPy RawKernels + cuBLAS GEMM; numpy fallback via `AVP_NO_CUPY=1`). Parity
and per-op timings: `python -m pyplumber.court_segm.cuda_bench`. See
`doc/court_calibration_cuda_shaders.md` for measured results.

## CUDA rewrite priority (by pixels touched × calls per clip, t66/t67 counters)

All functions below now live in `pyplumber/court_segm/`. "Calls" are from
the 2404-frame bbl.mp4 run; per-call pixel volume is the mask grid 272²≈74k
unless noted. The plan: one `court_segm/cuda.py` with CuPy RawKernels and a
numpy fallback (auto-selected at import), kernels written as literal CUDA C
so they port verbatim into the C++ node.

| # | function (module) | pixels/call × calls | today | CuPy target |
|---|---|---|---|---|
| 1 | `_reconcile_spillover` (evidence) | 74k × ~1800 | ~1–2 ms | 0.01 ms (RawKernel proven: fused per-pixel H-projection + analytic region test + reclassify) |
| 2 | `_synth_tpl_from_court` (evidence) | 74k × ~2300 (every wide frame) | fill_holes+label+dilation ≈ the −1.6 fps | fused invert→CCL→border/ring stats kernel |
| 3 | `_clean_region_mask` (evidence) | 74k × ~2–3/frame (per 3-pt det + per synth) | label+fill_holes | same CCL kernel, batched |
| 4 | `_lines_h` coarse GEMM + 27-render fine grid (matching) | 3191×5184 GEMM + 27×5184 renders × ~1–2/frame | several ms | 0.42 ms cuBLAS (proven) + one fine-grid kernel |
| 5 | `_tpl_boundary_points` (evidence) | find_contours 272² × ~3/frame (yellow, hole, court contour) | ~1 ms each | marching-squares kernel emitting ≤300 pts, or GPU-downsample-then-CPU |
| 6 | `_downsample_mask` (matching) | 272²→5184 gather × ~3/frame | cheap-ish | free as a kernel epilogue of 1–4 |
| 7 | `_coast_verified` grid (temporal) | 5184 × rare | trivial | reuse template kernel; low priority |
| 8 | `_court_top_edge` / `_court_bottom_edge` / `_baseline_edge_points` (evidence) | ~270-column/row PYTHON loops × 2–4/frame | slow for what it does | first vectorize in numpy (argmax tricks) — no GPU needed |

Stays CPU/Python (user directive "leave easy cpu work for python"):
`fitting.py` (LM refines, consensus vote — ≤300 pts), `temporal.py`
(filter, zones, publish), `geometry.py` helpers, all gates, orchestration
in `court_calibration.py`.

## Port mapping

CUDA kernels (operate on the GPU masks in place, output small buffers):
- clean/fill/label → connected components (e.g. CCL kernel or cuCIM), or
  simply run at the 96×54 template grid where 10× cheaper is acceptable.
- reconcile + court-synthesis membership tests → one fused kernel over the
  mask grid (per-pixel homography + analytic region test).
- template scoring → one batched GEMM over a precomputed template matrix
  (cuBLAS, sub-ms) + rim-distance term.
- boundary extraction → marching-squares kernel emitting point list, or
  downsample-then-CPU (the fit only uses ≤300 points).

C++ (Eigen, single thread, sub-ms total):
- `_ptz_to_h8`/`_ptz_project`, LM refine with analytic Jacobians,
  consensus binning (no KDTree needed — direct segment classification),
  temporal filter, all gates.

Keep in Python until last: nothing — the node becomes
`src/nodes/neural_net/sport_specific/court_calibration.cpp` per the
original plan; the Python node stays as the reference implementation for
offline harnesses (`~/feetfix-run/ptz_check.py`, `ridge_eval.py`).

## Cheap interim wins (no port)

- Run `_synth_tpl_from_court` + reconcile every 2nd–3rd frame when the
  filter state is warm (masks barely change frame-to-frame): recovers most
  of the −1.6 fps.
- Skip the every-12th rig fit after `_rig_n` > 50 (converged).
- Cache `_court_top_edge`/`_baseline_edge_points` between reconcile and the
  relative branch (currently computed twice on some paths).

## Validation contract for the port

Parity with the Python node on bbl.mp4 measured by the existing harness:
ridge p50 ≤ 8 px, p90 ≤ 13 px, signed bias |·| ≤ 0.05 ft, apex flicker
p50 ≤ 4 px (`ridge_eval.py` + `AVP_CALIB_LOG` ndjson), close-up windows no
worse than 53/57 (`count_closeup.py`).
