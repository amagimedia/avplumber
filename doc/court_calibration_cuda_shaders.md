# Court calibration: CUDA shaders to implement

Status: spec only, 2026-06-11 — no kernels written yet (user decision).
Companion to `doc/court_calibration_realtime_port.md` (throughput analysis,
validated CuPy timings, division-of-labor). Python sources live in
`pyplumber/court_segm/` after the module split.

Ground rules (user directives):
- Only per-pixel mask ops and the template GEMM go to CUDA; LM fits, gates,
  consensus voting, temporal filter, and orchestration stay in Python.
- Kernels are written as literal CUDA C (CuPy `RawKernel`), so they port
  verbatim into the eventual C++ node.
- Every kernel has a numpy fallback selected at import (`court_segm/cuda.py`
  module, planned); a run with `AVP_NO_CUPY=1` must be bit-comparable.
- Masks are 272×272 float32 (≈74k px), N≈2–6 planes/frame, already on the
  GPU in the pipeline but currently re-uploaded (0.6 MB ≈ 0.1 ms — fine
  until pybind exposes the GPU-side planes).

Shared device helpers (header-style, used by several kernels):
- `apply_h8(h8, xn, yn)` — point through the 8-vec homography (court-norm out).
- `inside_tpl_region(X, Y)` / `inside_region_inset(X, Y, margin)` — analytic
  3-pt region membership (mirror of `geometry.py`).
- `mask_idx_to_norm(c, r)` — letterbox-undoing mask→source-norm mapping
  (constants from node params).

## K1 — reconcile_spillover (priority 1)

Replaces the per-pixel core of `evidence._reconcile_spillover`
(~1800 calls/clip, ~1–2 ms each in numpy; RawKernel prototype measured
0.01 ms — see `~/feetfix-run/cupy_test.py`).

- In: tpl mask, court mask (272² f32), prior h8 (8 f32), margin ft,
  threshold.
- Per pixel: source-norm coords via `mask_idx_to_norm` → `apply_h8` →
  `inside_region_inset` && in-court-rect; if inside && court≥thr &&
  tpl<thr: tpl=max(tpl,court), court=0.
- Out: both masks updated in place + an atomic counter of reclassified px
  (the Python side keeps its no-op/fail-counter logic).

## K2 — hole synthesis (priority 2)

Replaces the ndimage chain in `evidence._synth_tpl_from_court`
(`binary_fill_holes` + `label` + per-component `binary_dilation` ring test —
runs EVERY wide frame; measured ≈ −1.6 fps when introduced).

- In: floor mask (272² f32), threshold, min_region_area.
- Stages (single fused launch sequence, ping-pong buffers):
  1. binarize + close borders (rows 0/H-1, cols 0/W-1 set).
  2. connected components on the INVERSE (holes) — 8-conn label-propagation
     CCL (iterate until stable; 272² converges in ~10–15 sweeps) or
     union-find CCL.
  3. per-component reduction: area; ring test = for each hole px count
     4/8-neighbors that are court px → component ring-surrounded fraction.
  4. write synthetic mask (thr+0.25) for components with area ≥
     min_region_area and ring-court fraction ≥ 0.5.
- Out: synthetic 3-pt mask or "none" flag.

## K3 — region mask cleanup (priority 3)

Replaces `evidence._clean_region_mask` (`label` + keep-largest +
`binary_fill_holes`; 2–3 calls/frame: per confident 3-pt detection and on
K2's output).

- In: mask (272² f32), threshold.
- Stages: binarize → CCL (same kernel as K2 stage 2) → argmax-area
  component → fill holes (CCL of inverse, keep components touching the
  border, everything else = hole) → compose output: zero specks outside
  kept blob, raise filled holes to thr+0.25, leave sub-threshold boundary
  px untouched (sub-pixel contour accuracy depends on this).
- Out: cleaned float mask.
- Note: K2 and K3 share the CCL building block — implement CCL once.

## K4 — template scoring (priority 4)

Replaces the hot path of `matching._lines_h` and
`matching._court_only_match`.

- K4a coarse: IoU of observed downsampled masks vs all pre-rendered
  templates. 3191×5184 GEMM (template matrix × obs vector) — cuBLAS via
  `cp.matmul`, measured 0.42 ms. Plus elementwise score combine
  (2*iou_tpl + coverage − 0.3*excess − 2*min(d_rim,0.3)) and argmax —
  trivial follow-up kernel or do on CPU (3191 floats).
- K4b fine grid: the 27 on-the-fly renders around the winner. One kernel,
  27 blocks (one per (dth,dph,dF) candidate): each block projects the
  5184-cell grid through its candidate h8 (`apply_h8` + `inside_tpl_region`),
  accumulates intersection/union vs obs_tpl in shared memory, writes one
  IoU. Replaces 27 Python-loop renders.
- In: obs_tpl/obs_court (5184 f32), template matrix (persistent on GPU),
  rim table, candidate params. Out: per-template scores / 27 IoUs.

## K5 — boundary extraction (priority 5)

Replaces `evidence._tpl_boundary_points` (`skimage.find_contours`, ~3
calls/frame: yellow, hole, court contour).

- Option A (preferred): marching-squares kernel — per 2×2 cell, detect a
  0.5-crossing, emit interpolated sub-pixel point(s) into a global buffer
  via atomic append. The fit uses ≤300 points after downsampling and does
  NOT need contour ordering — an unordered point set is sufficient for
  segment assignment (verify once against `find_contours` output on a
  dump frame; the border-margin filter applies per point).
  Largest-contour length gates (≥40 / ≥20) become: count crossings per
  CCL component (reuse K2/K3 labels) and keep components above threshold.
- Option B (fallback if ordering turns out to matter): GPU downsample +
  CPU find_contours on the smaller grid.
- In: mask (272² f32), threshold, border margin. Out: (N,2) source-px
  points + count.

## K6 — mask downsample (priority 6, epilogue)

`matching._downsample_mask` (272²→54×96 nearest-neighbor gather + binarize,
~3 calls/frame). Implement as an epilogue/preamble of K1/K4 launches (or a
10-line standalone kernel). Also covers `temporal._coast_verified`'s 5184-pt
grid projection (reuse K4b's projection helper with one candidate).

## Not CUDA — vectorize in numpy instead

`evidence._court_top_edge` / `_court_bottom_edge` / `_baseline_edge_points`
are per-column/row PYTHON loops (~270 iterations, 2–4 calls/frame). Replace
the loops with `argmax` over the binarized mask along the axis + validity
masking — a pure numpy change, no kernel warranted.

## Integration order & expected win

1. K1 + K2 + K3 (shared CCL) — removes the dominant ndimage cost and the
   −1.6 fps synthesis regression.
2. K4a/K4b — removes the multi-ms template path.
3. K5, K6 — finishes the per-pixel exodus; CPU then touches only point
   lists ≤ a few hundred entries.

Each step must hold the validation contract (ridge_eval on bbl.mp4:
p50 ≤ 8.5 px, p90 ≤ 14 px, |bias| ≤ 0.05 ft, close-up windows ≤ 53/57,
zone histogram unchanged) before the next lands.
