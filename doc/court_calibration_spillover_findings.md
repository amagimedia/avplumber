# Court Calibration — 3-pt Magenta-Dot Accuracy: Spillover Findings

Status: **investigation, not shipped.** Date: 2026-06-11. Branch: `fable`.

This documents an attempt to fix the magenta 3-pt-arc calibration overlay in
`pyplumber/court_calibration.py` (drawn by `tactical_view.py`). Two segmentation
spillover problems were targeted. Neither attempt produced a *visually* good
result, even where aggregate counters improved. The key takeaway is at the
bottom: **the per-frame fail counters do not track perceived overlay quality —
future work must measure the overlay itself.**

## The two problems (as stated by the reviewer)

1. **First half — court floods INTO the arc.** The `basketball-court` seg class
   bleeds into the 3-pt region interior. Inside the painted arc only the 3-pt
   class is physically possible, so this is a model artifact. When severe, the
   `three point line` blob collapses and the fit bails (`region_small`),
   leaving a stale held homography.
2. **Second half — the 3-pt region bulges OUT past the arc in one spot.**
   ~90 % of the arc is segmented cleanly; one local lump extends beyond the
   true arc. Because the 3-pt line is always arc-shaped, those bulge boundary
   points are outliers that drag the homography fit.

## What was implemented (all opt-in, default behaviour preserved)

### A. Spillover reconciliation — `_reconcile_spillover` + `_inside_region_inset`
Project the court mask forward through a prior homography into court feet;
where court-class activation lands inside the analytic 3-pt region
(`_inside_tpl_region`) **and** the 3-pt class did NOT fire there
(`tpl < threshold`), reclassify it as 3-pt region. Refinements added during the
investigation:
- **Inset (`reconcile_margin_ft`, default 2.0):** reclaim only the deep
  interior; leave a band next to every painted line untouched so the genuine
  3-pt boundary (which the arc-fit uses) is never overwritten by a
  prior-H-shaped edge. Without this, the reclaimed region's outer edge becomes
  a fake prior-shaped arc.
- **`tpl < threshold` gate:** court+3pt legitimately co-fire on the same floor,
  so reclaim ONLY where the 3-pt class failed. Without this gate the step
  mutated good frames and regressed the second half.

Prior H = held fit when same-side and fresh (`age <= hold_frames`), else a
coarse template-IoU match (robust to spillover). No prior → no-op.

### B. Arc-shape outlier trim — in `_refine` (`arc_outlier_ft`, default 2.5)
After the existing two-pass segment-labeled LM fit, refit once dropping
arc-labeled boundary points whose radius under the current fit deviates from
`ARC_R` by more than `arc_outlier_ft`. The 3-pt line is arc-shaped, so the
bulge is not the painted line. Adopt the trimmed fit **only if** it lowers the
all-points median residual (guard against regression). `_refine`'s fit logic
was refactored into a single nested `fit_pass()` helper (no copy-paste) so both
the normal passes and the trim pass share it.

## Measured results — 4-arm isolation, matched frame 500, bbl.mp4

| arm                | published_valid | relative_unavailable | relative_refined | refine_reject |
|--------------------|-----------------|----------------------|------------------|---------------|
| base (neither)     | 449             | 34                   | 153              | 168           |
| reconcile only     | **474**         | **16**               | 143              | 193           |
| arc-trim only      | 449             | 34                   | **111** ⚠️       | 210           |
| both               | 474             | 17                   | 111              | 225           |

`relative_refined` = frames that passed the precise point-to-arc LM (the count
most correlated with dots-on-the-line); `published_valid` = frames with any
calibration; `relative_unavailable` = template fit failed entirely.

### Reading the numbers
- **Reconcile (gated + inset)** improves coverage cleanly: +25 calibrated
  frames, half the no-fit failures, precision essentially held (153→143). The
  `tpl<threshold` gate fixed an earlier second-half regression seen when the
  step mutated healthy frames.
- **Arc-trim is a net negative** on `relative_refined` (153→111) even with the
  adopt-if-better guard. Hypothesis: dropping the bulge lowers the *local* arc
  residual while removing depth-anchoring points, so a trimmed fit is adopted
  on the median test yet is *globally* worse and fails the downstream gate.
  **Median residual is the wrong yardstick for this guard.**

## The decisive finding (visual review)

**Reconcile-only (`tactical_view_iso_recon.mp4`) still looked bad to the
reviewer, despite the best aggregate counters.** This is the headline result:

> The per-frame fail counters (`relative_refined`, `published_valid`, etc.) did
> NOT predict perceived overlay quality. A config can post better numbers and
> still look worse on the painted arc.

Likely reasons the counters mislead:
- `relative_refined` only asks whether the LM *passed its gate*, not whether the
  dots sit ON the painted arc. A fit can pass the median-residual gate while
  carrying a systematic depth/perspective bias (the magenta arc parallel to but
  offset from the painted one).
- Reconcile changes *which* frames get a fresh fit, so it can trade a stable
  held-H look for more frequent but jitterier fresh fits — worse to the eye,
  better on coverage counts.
- The EMA blends H in element space (mixes perspective terms nonlinearly), so
  more fresh fits can increase visible wobble.

## Recommendations / next steps

1. **Do not ship A or B as-is.** Neither is a visual win. Reconcile is the more
   promising of the two and is correctly gated, but it must be validated
   against an overlay-quality metric, not the fail counters.
2. **Build an overlay-quality metric before more tuning.** Candidates:
   held-out-curve cross-validation in feet (fit without the arc, measure arc
   reprojection error — already proposed in
   `doc/tactical_court_calibration_plan.md`), and/or luma-line distance: sample
   the source luma along the reprojected arc normal and measure pixel offset to
   the brightest ridge. This directly scores "dots on the painted line".
3. **Re-evaluate arc-trim with the right guard:** require the trimmed fit to
   improve the metric in (2) on the FULL point set (including held-out bulge),
   or cap (don't drop) bulge residuals.
4. **Consider the line-snap (Stage 2) path instead** — 1-D luma ridge search
   along reprojected curve normals removes constant segmentation-boundary bias
   directly, which is closer to what the eye is judging. Needs the `luma_export`
   C++ node (CUDA frames can't be read from Python). See
   `doc/tactical_court_calibration_plan.md` lines 127-129.

## Reproduction

All runs in the `avp-builder:latest` container on the remote (172.17.44.114).
Pure-Python edits to `court_calibration.py` need NO rebuild, but the container's
`PYTHONPATH` puts a stale baked `/opt/avplumber/pyplumber` ahead of the live
checkout — overlay-mount the edited file:

```bash
sudo docker run --rm --gpus all \
  -v <repo>:/avp \
  -v <repo>/pyplumber/court_calibration.py:/opt/avplumber/pyplumber/court_calibration.py:ro \
  -v <repo>/pyplumber/examples/tactical_view.py:/opt/avplumber/pyplumber/examples/tactical_view.py:ro \
  -v <models>:/models -v <input.mp4>:/media/input.mp4 \
  --entrypoint bash avp-builder:latest -lc '<run script>'
```

Env overrides exposed by `tactical_view.py` for A/B:
`AVP_RECONCILE_SPILLOVER` (0/1), `AVP_RECONCILE_MARGIN_FT`, `AVP_ARC_OUTLIER_FT`.
scipy/scikit-image are not in the image — `python3 -m pip install scipy
scikit-image` each run (ephemeral `--rm` container).

## Code state on branch `fable`

`pyplumber/court_calibration.py` carries `_reconcile_spillover`,
`_inside_region_inset`, the `fit_pass` refactor of `_refine`, and the arc-trim,
all behind params (`reconcile_spillover`, `reconcile_margin_ft`,
`arc_outlier_ft`). `tactical_view.py` enables them with env overrides. **These
are experimental and not validated as a visual improvement** — see
recommendation 1 before relying on them.
