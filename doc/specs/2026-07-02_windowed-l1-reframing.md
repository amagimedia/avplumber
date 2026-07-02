# Windowed-L1 (receding-horizon) stabilization + reframing node — porting plan

## Goal

Add a **self-contained avplumber node** implementing **L1-optimal camera-path**
smoothing, for a **standalone demo** driven by an avplumber graph.

The node targets **reframing/stabilization**.

The L1-optimal family solves its program **over the whole video (batch)**. The
Apple paper (below) already gives a **windowed, bounded-latency approximation**;
we adopt that scheme so the node runs **live** with a user-defined latency
`H = round(max_latency_s · fps)` frames, capped at ≤ 0.5 s (0.3 s ideal). The L1
objective produces cinematic *static / linear / parabolic* camera segments. The
node consumes the **affine/homography camera motion** already produced by
`CudaCameraMotion`, which the existing `SmoothCropViewport` ignores.

The node supports **two formulations, switchable at runtime** via a `mode`
parameter (see "Two modes" below).

---

## Source papers & lineage (corrected)

The "L1-optimal camera path" idea has a family tree; getting the provenance right
matters because each member contributes a different piece we want.

1. **Grundmann, Kwatra, Essa — "Auto-Directed Video Stabilization with Robust L1
   Optimal Camera Paths", CVPR 2011** (Google). `pubs/archive/37041.pdf`, read in
   full. **Foundational.** Optimizes a full 2D **affine/similarity** camera path
   so its 1st/2nd/3rd derivatives are L1-sparse → path decomposes into exactly
   static, linear, parabolic arcs. Objective `w1|D(P)|+w2|D²(P)|+w3|D³(P)|`,
   weights ~(10,1,100); hard **inclusion** constraint (crop corners inside frame,
   eq. 8), affine **rigidity** bounds (`0.9≤a,d≤1.1`, `|b|,|c|≤0.1`, `|b+c|≤0.05`,
   `|a−d|≤0.1`), optional **soft saliency**. Solved as an **LP** (COIN CLP
   simplex). Homographies only appear in a post-hoc "residual motion suppression"
   step. **Batch, not causal.**

2. **Achary et al. — "CineFilter" (2019), the "CineConvex" formulation.** Adds a
   **windowing** approach for online processing of the **affine** model. **This is
   what SRS's `sports_reframing/algorithms/viewport/cineconvex/` is named after** —
   SRS implements the shared L1-derivative core, collapsed to a **scalar**
   (viewport-center-x, full-height crop) that follows a salient target
   horizontally. (Earlier notes in this repo mislabeled arXiv 2011.08144 as "the
   paper SRS is built on"; the actual namesake is CineConvex/Achary.)

3. **Bradley, Klivington, Triscari, van der Merwe (Apple) — "Cinematic-L1 Video
   Stabilization with a Log-Homography Model", WACV 2021** (arXiv 2011.08144),
   read in full (TeX). **Most advanced of the family; extends Grundmann.** Adds:
   full **homographies** via a **log-homography (Lie-algebra)** transform (keeps
   composition/inverse linear → convex); a **QP** with an **ℓ₂ fidelity** term
   `½w₀‖p‖₂²` (strictly convex, unique solution, per-frame strength knob);
   **FoV-preserving** crop constraints (area + sidelength, Jacobian-linearized) on
   top of valid-pixel; **distortion** penalties (`‖a−d‖²`, `‖b+c‖²`,
   keystone-translation ratio `‖k−Rt‖²`); **dual saliency** (inclusion constraint
   *and* centering objective `‖p−p_target‖²`); and a **3-frame-Markov windowing**
   scheme (below). Runs 300 fps on iPhone XS.

Neither the batch nor the windowed variants above validate lookahead as short as
ours — see the **lookahead caveat** at the end.

---

## The windowing scheme we adopt (Apple, §"Windowed L1")

This is the causal mechanism, taken directly from the Apple paper — **not
invented here**:

1. Solve the global problem in window `[s_w, s_w + l_w)`.
2. **Fix** the solution for the first `l_s` frames (stride `l_s < l_w`).
3. Advance the window by `l_s − 3` frames.
4. Re-solve `[s_w, s_w + l_w)` subject to an **initialization constraint forcing
   only the first 3 frames to match the previously fixed solution**
   (`p_t = p_t^fixed` for `s_w ≤ t < s_w+3`).
5. Repeat to end of stream.

**Why 3 frames:** a **third-order Markov property**. Past and future are coupled
*only* through derivatives up to 3rd order (the objective has D1/D2/D3), so
fixing exactly 3 frames captures the entire history — the windowed solve then
equals the global solve over `[0, s_w+l_w)` restricted to agree with the fixed
past. This is the principled version of the "anchor carryover" continuity problem
and is **cleaner than CineConvex/Achary's windowing** (which constrains whole
overlapping segments) and than Qu-2013's (independent solves + weighted average).

For our live node: `l_w = H` (lookahead horizon), small stride (`l_s` a few
frames, e.g. `l_s−3 = 1` to emit one output frame per solve), and the 3-frame
init is our exact anchor. Latency = `H` frames = `H/fps` s.
`0.3 s @ 30 fps → H=9`; `0.5 s @ 30 fps → H=15`.

---

## Two modes (runtime-switchable)

`mode` ∈ {`stabilize`, `reframe`}. Same windowing, same solver core, same latency
mechanism, same `CudaCameraMotion` input — different variable set and constraints.

### `mode = stabilize` (Grundmann + Apple full-transform)

Removes camera shake — rotation jitter, vertical bounce, perspective/keystone
wobble, translation shake. This is the "taking camera shake into account"
behavior; `SmoothCropViewport`'s x-pan-follow cannot do it.

- **Variables per frame:** the correction transform. Two sub-levels, gated by
  `motion_model`:
  - `affine`/`similarity` (Grundmann): 4-DOF `(dx,dy,a,b)` or 6-DOF
    `(dx,dy,a,b,c,d)` — LP, simpler, start here.
  - `homography` (Apple): correction **log-homography** `p_t = log P_t ∈ ℝ⁹`,
    `trace(p_t)=0`; composition/inverse become +/− in log space. QP. More
    capable (perspective), more math. v2 of `stabilize`.
- **Input `Ft`:** the per-frame transform from `CudaCameraMotion` — **exactly the
  paper's analysis transform** `F_t` (maps `t→t-1`). We skip the papers' entire
  front-end (Lucas-Kanade + grid-RANSAC + robust fit); `CudaCameraMotion` already
  produces `F_t` on GPU. For homography mode we take `f_t = log F_t`.
- **Objective:** `½w₀‖p‖₂² + w1‖e¹‖₁ + w2‖e²‖₁ + w3‖e³‖₁`, with (Apple eq.):
  `e¹(t)=p_{t+1}+f_{t+1}−p_t`,
  `e²(t)=p_{t+2}+f_{t+2}−2p_{t+1}−f_{t+1}+p_t`,
  `e³(t)=p_{t+3}+f_{t+3}−3p_{t+2}−2f_{t+2}+3p_{t+1}+f_{t+1}−p_t`.
  Jerk `w3` an order of magnitude largest (Grundmann fig. 8). `w0`>0 gives strict
  convexity + fidelity + a per-frame stabilization-strength knob (relax on
  motion-blur frames).
- **Constraints:** derivative-slack bounds; **valid-pixel** inclusion (4 crop
  corners transformed by the correction stay inside the frame, Jacobian-
  linearized — Apple's `D(p_t;c) ≈ ∇D|₀ p_t`); optional **FoV** (area ≥
  `(1−cropfrac)²·frame`, sidelen ≥ `(1−cropfrac)·frame`); rigidity (hard bounds in
  affine mode, or quadratic `‖a−d‖²`/`‖b+c‖²` penalties per Apple); `trace(p_t)=0`
  in homography mode.
- **Detector:** not required (pure stabilization). Saliency optional.

### `mode = reframe` (SRS/CineConvex scalar crop-follow)

Full-height crop that follows a salient target (ball/player) horizontally,
panning with the camera. Numerically closest to the existing SRS offline path.

- **Variables per frame:** scalar viewport center `x` + target slack `e`.
- **Input:** salient target `cx` (detector, e.g. `yolo_detections`) + camera path
  `C_x` integrated from `CudaCameraMotion` `tx`.
- **Objective:** `w1·Σ|Δx| + w2·Σ|Δ²x| + w3·Σ|Δ³x| + Σ w·e` (optionally
  `+ ½w₀·Σ(x−x_target)²` centering, borrowing Apple's centering idea).
- **Constraints:** target proximity (`target−e ≤ x ≤ target+e`), camera proximity
  (`C_x−bound ≤ x ≤ C_x+bound`), box (`vpw/2 ≤ x ≤ vw−vpw/2`).
- **Detector:** required (it's the saliency source).

---

## Why not the existing avplumber smoother

`avplumber/src/nodes/neural_net/utils/smooth_crop_viewport.cpp` is the existing
*live* smoother: Kalman/IIR/FIR lowpass + `DerivSlewAxis` (velocity/accel/jerk
slew limiting) + hold-latch + `lookahead_frames` buffer. It is **greedy and
reactive** — clamps jerk but cannot *plan* an ease toward a target it hasn't yet
reached, does **not** consume camera motion, and cannot remove
rotational/vertical/perspective shake (1-DOF center only). Windowed L1 is a
*planner* and, in `stabilize` mode, a full-transform stabilizer.

---

## Context (avplumber side)

- `CudaCameraMotion`
  (`src/nodes/neural_net/sport_specific/cuda_camera_motion.cpp`) emits per-frame
  metadata under key **`camera_motion`**:
  `{tx, ty, affine_2x3, nvof_cost, has_prev, status, affine_valid, ...}`. Needs
  `HAVE_NVOF=1` + `--gpus all` for dense optical flow. **This is the `F_t` /
  camera-path source for both modes. No change needed.** (It emits a 2×3 affine;
  homography `stabilize` would treat it as an affine homography, or the node is
  extended later if a full homography estimate is wanted — out of scope for v1.)
- `SmoothCropViewport` / `Reframer` — reference implementations of the
  metadata-parse / lookahead-buffer / viewport-metadata patterns we reuse.
- A detector node (YOLO → `yolo_detections`) supplies the saliency target;
  required for `reframe`, optional for `stabilize`.

---

## Standalone demo shape

Deliverable: an **avplumber graph config** that runs on a clip and produces a
**visibly stabilized/reframed mp4** plus per-frame warp/viewport metadata.

```
decode (CUDA) → [detector → yolo_detections]? → CudaCameraMotion (→ camera_motion)
  → windowed_l1 (mode=stabilize|reframe → windowed_l1_v1 metadata)
  → warp/crop/scale node (applies Pt or the crop box) → encode → mp4
```

The node only *decides* the transform/crop and writes it as metadata (mirroring
`SmoothCropViewport`); an existing warp/crop/scale node renders it. Keeps the
node a pure decision-maker, reusable by the future product. Runs via the plain
avplumber binary or the `_avplumber` python module on a GPU host


---

## Design

### Node placement

Do **not** shoehorn into `SmoothCropViewport`'s `LowpassBackend` interface — a
per-frame `step()` with no horizon, structurally wrong for a window-batch solve.
Create a **new node** `windowed_l1` (working name) reusing `SmoothCropViewport`'s
building blocks: metadata parsing, the `buffer_` deque + delayed-output pattern
(the latency mechanism), `metadataRequestsReset()` (scene-cut reset), and the
viewport-metadata output schema.

### Per-frame algorithm (emit output for the oldest committed frame)

1. **Fill the window** `[s_w, s_w+H)` from the buffer: affine/homography `F_t`
   from `camera_motion`, plus (if present) the salient target box.
2. **Build inputs:**
   - `stabilize`: form residuals `e¹,e²,e³` from `f_t` (affine) or `log F_t`
     (homography), per the equations above.
   - `reframe`: integrate camera path `C_x` from `tx`
     (`compose_camera_path_x`), detect static segments (`segment_is_static`).
3. **Solve** the windowed L1 (LP for affine/reframe, QP for homography / with
   ℓ₂ terms) over the window's variables + constraints.
4. **3-frame-Markov init (the anchor):** constrain the first 3 frames of the
   window to the previously fixed solution (`p_t = p_t^fixed`). This is the exact
   continuity mechanism from the Apple paper — no hand-tuned "pin 1–3 vars."
5. **Commit** the stride's frames as output; keep the last 3 as the next window's
   fixed init. **Warm-start** the next solve from the shifted solution → few
   iterations.
6. Write output metadata: `stabilize` → correction `P_t` (2×3 or 3×3 warp) +
   resulting crop box; `reframe` → viewport crop box. Reuse
   `viewport_px_fit_height` for box sizing.

### DOF / staging

- `reframe`: x only first (primary DOF); optional later `cy` from `ty`, plus the
  Apple centering objective.
- `stabilize`: affine/similarity first (LP, rigid). **Log-homography** (QP,
  perspective) is a distinct, larger step layered on the same windowing + solver
  core once affine is validated. Residual-wobble / rolling-shutter suppression
  (Grundmann §3 homography key-frame trick) is out of scope.

---

## Solver in C++ (no cvxpy, no Python)

- **Affine `stabilize` and `reframe`** reduce to a sparse **LP**: abs terms
  epigraph into `min cᵀe s.t. −e ≤ D·x ≤ e`, plus box / proximity / inclusion /
  3-frame-init equality constraints.
- **Homography `stabilize` and any ℓ₂ term** (fidelity, centering, distortion)
  make it a sparse **QP**. Same constraint machinery.

Sizes are tiny at `H ≤ 15`: `reframe` ~15 vars; affine `stabilize` ~15×4–6;
homography `stabilize` ~15×9 + slacks. Inclusion/FoV/saliency constraints are
Jacobian-linearized (linear in `p_t`), so the problem stays LP/QP.

**Recommended: OSQP** (pure C, single static lib, no BLAS, MIT, warm-start,
sub-ms). It solves **both LP and QP** (QP is its native form; LP = QP with zero
quadratic term), so one dependency covers all modes. Alternative: **Clarabel-C**
(interior-point, closest to SRS/Apple numerics, but drags Rust into the build).
Grundmann used COIN CLP simplex; OSQP's ADMM is a fine, easily-vendored
substitute. Expect small numeric deviation from the offline paths — acceptable.

---

## Build integration (avplumber only)

Third-party libs are vendored under `deps/` and static-linked into `DEPS_LIBS`
(pattern: `cpr`, `avcpp`, `librdkafka`). To add OSQP:

1. Vendor OSQP under `deps/osqp` (submodule or source drop).
2. Add a `deps/osqp/build/...libosqp.a` target to the `Makefile` (CMake, mirror
   the `deps/cpr/build/lib/libcpr.a:` recipe), append to `DEPS_LIBS`, add
   `-Ideps/osqp/include` to `CXXFLAGS`. Gate behind `HAVE_OSQP=1` (mirror
   `HAVE_KAFKA` / `HAVE_NVOF`). Log-homography mode also needs a matrix
   exp/log — Eigen (already available, `-I/usr/include/eigen3`) provides
   `MatrixBase::exp`/unsupported `MatrixLogarithm`; no new dep.
3. avplumber **`Dockerfile`** (plain, not any combined image): pass `HAVE_OSQP=1`
   to the `python_module`/binary make; build OSQP in the deps stage; also needs
   `HAVE_NVOF=1` for `CudaCameraMotion`. Static link → no runtime `.so` copy, no
   CodeArtifact / private repo.

---

## New node parameters

| param | type | default | meaning |
|-------|------|---------|---------|
| `src`, `dst` | edge | — | video in/out (SISO) |
| `mode` | str | `reframe` | `stabilize` (full-transform) or `reframe` (scalar-x crop) |
| `camera_motion_metadata_key` | str | `camera_motion` | key with per-frame `{tx,ty,affine_2x3,status}` (= `F_t`) |
| `metadata_key_out` | str | `windowed_l1_v1` | output warp/viewport metadata key |
| `max_latency_s` | float | `0.3` | window `H = round(max_latency_s·fps)`; latency knob |
| `stride_frames` | int | `1` | frames committed per solve (`l_s`); 3-frame Markov init always kept |
| `l1_w1` / `l1_w2` / `l1_w3` | float | `10 / 1 / 100` | 1st/2nd/3rd-derivative weights |
| `l2_w0` | float | `0` | ℓ₂ fidelity weight (>0 ⇒ QP, strict convexity, strength knob) |
| `debug_log_every_n` | int | `0` | periodic solve-cost / latency logging |
| **stabilize-only** | | | |
| `motion_model` | str | `affine` | `similarity` / `affine` (LP) or `homography` (log-homography QP) |
| `crop_scale` | float | `0.9` | crop-window fraction (valid-pixel + FoV constraints) |
| `fov_constraint` | bool | `true` | enable area+sidelength FoV preservation (Apple) |
| `rigidity_mode` | str | `penalty` | `bounds` (Grundmann hard) or `penalty` (Apple `‖a−d‖²`,`‖b+c‖²`) |
| `keystone_translation_ratio` | float | — | if set, add `‖k−Rt‖²` distortion term (homography) |
| `affine_translation_weight_ratio` | float | `100` | affine:translation objective weighting |
| **reframe-only** | | | |
| `metadata_key_ins` / `metadata_key_in` | str[] | `yolo_detections` | salient-target source(s) |
| `viewport_dst_width` / `viewport_dst_height` | int | — | crop size (even) |
| `proximity_bound_norm` | float | (SRS) | camera-following bound; `·vw` → px |
| `static_threshold_norm` | float | (SRS) | mean-\|tx\|/vw below which segment is static |
| `max_tx_px` | float | `min(bound, vw·0.05)` | per-frame camera-shift cap |
| `focus_mode` / `allowed_*` / `min_conf` | — | — | reuse SmoothCropViewport target-selection |
| **saliency (both, optional)** | | | |
| `saliency_metadata_key` | str | — | soft inclusion constraint (`P⁻¹s ∈ crop`) + optional centering objective `‖p−p_target‖²` |

---

## Testing

Solver / algorithm:

- **`reframe` unit**: static target → viewport still (diffs ≈ 0); linear target →
  constant-velocity track; step → eased S-curve, no overshoot past bound. Windowed
  vs SRS full-scene on a synthetic segment (close, not identical).
- **`stabilize` affine unit**: synthetic shaky `F_t` (sinusoidal jitter + low-freq
  bounce) → output path has D2/D3 ≈ 0 over segments (static/linear/parabolic
  decomposition, Grundmann fig. 8); inclusion never violated; rigidity respected.
- **`stabilize` homography unit**: keystone-wobble input → keystone removed;
  log/exp round-trips within tolerance; trace(p_t)=0 held; FoV constraint keeps
  area/sidelength above budget.
- **3-frame-Markov continuity**: windowed solve matches the global batch solve to
  tolerance on a fixed clip (validates the Markov claim); no position/velocity
  discontinuity across window boundaries; visible jump if the 3-frame init is
  disabled (guards the mechanism).
- **Camera compensation (`reframe`)**: synthetic pan (`tx` ramp), world-stationary
  target → viewport pans with the camera, doesn't fight it.
- **Latency**: output for input frame `t` appears exactly `H` frames later;
  `max_latency_s`→`H` verified at 25/30/50 fps; both modes.
- **Lookahead-quality sweep** (see caveat): vary `H` from 5 to 60 frames on a
  clip with a known static→pan transition; measure residual D2/D3 energy and the
  quality knee. Records where short-horizon planning degrades.
- **Scene cut**: `metadataRequestsReset` clears the 3-frame init + camera path.
- **Mode switch**: both modes run from the same node; params gate correctly.

Demo (avplumber standalone): run each mode on a clip with real pans/shake (e.g.
bbl.mp4, ~58 px pans), produce the mp4, eyeball: `stabilize` removes shake and
holds steady; `reframe` rides the pan and stays on the action.

---

## Work breakdown

1. **Vendor + build-wire OSQP** (`deps/osqp`, `Makefile` `HAVE_OSQP`, avplumber
   `Dockerfile`). Verify: `libosqp.a` builds, trivial LP+QP solve links.
2. **L1 window solver core** — sparse LP/QP builder + OSQP wrapper, warm-start,
   3-frame-init equality support. Mode-agnostic given a variable/constraint spec.
3. **`reframe` mode** — scalar formulation + windowed `compose_camera_path_x` /
   `segment_is_static` (causal ports from SRS `camera_motion.py`). Unit-tested vs
   SRS full-scene.
4. **`stabilize` affine mode** — 4/6-DOF `P_t`, D1/D2/D3 residuals, rigidity
   (bounds or penalty), valid-pixel inclusion (Jacobian). Unit-tested vs Grundmann
   behavior. Validate the 3-frame-Markov windowing against a batch solve here.
5. **`stabilize` homography mode** — log-homography (Eigen exp/log), QP with
   ℓ₂ fidelity, FoV + distortion terms, `trace=0`. Layered on the same core.
6. **`windowed_l1` node** — buffer/latency + metadata parse + mode dispatch +
   windowed solve + 3-frame init + warm-start + output metadata. `DECLNODE`.
7. **Demo graph configs** (per mode) — decode → [detector] → `CudaCameraMotion` →
   `windowed_l1` → warp/crop → encode.
8. **Demo + lookahead-sweep validation** on a clip with real pans/shake.

Ordering: 1-2-3 give a working `reframe` demo first (matches SRS); 4 adds affine
`stabilize` + validates windowing; 5 adds the most capable homography mode.

---

## Lookahead caveat (the real risk of our low-latency target)

The Apple 3-frame-Markov init makes the **past** exact regardless of window size,
so **continuity is guaranteed** at any `H`. But the L1 *planner* still needs
enough **future** lookahead to see an upcoming static/linear/parabolic segment
and ease into it. Apple validates windowing with `l_w = 1800` frames (60 s);
Grundmann and CineConvex likewise assume generous horizons. At **9–15 frames
(0.3–0.5 s)** we are ~2 orders of magnitude below that regime. As `H` shrinks the
result degrades gracefully toward the greedy/reactive behavior of the existing
slew limiter — you cannot plan a smooth ease-out into a tripod shot you can only
see 0.3 s ahead. This is unproven at our latency and is the main technical risk.
Mitigation: the lookahead-quality sweep test above measures the knee early;
`max_latency_s` is the exposed tradeoff knob; and the ℓ₂ fidelity term keeps
short-horizon solutions well-behaved.

---

## Explicitly NOT needed / out of scope

- No cvxpy / clarabel-Python / scipy — all C++ + OSQP (+ Eigen for exp/log).
- No changes to `CudaCameraMotion` — it already emits `F_t`.
- No motion-estimation front-end (Lucas-Kanade / grid-RANSAC) — the transform is
  given.
- No residual-wobble / rolling-shutter suppression (Grundmann §3) — v2.
- No full-scene / non-causal solve path; no runtime `.so` for the solver.
