# Court Calibration Simplification Design

Date: 2026-06-12

## Goal

Preserve the accurate source-to-court homography path from the current tactical
view branch, but simplify the calibration logic enough that projected court
lines can be debugged and improved directly.

Target behavior:

- Projected broadcast-frame court lines are the first priority.
- Runtime should sustain roughly 30 fps in the tactical-view graph.
- Wide-shot line overlay coverage should reach 85-90% without publishing
  obviously wrong lines.
- Accuracy should remain better than the `basketball-demo` branch's
  direction/scale fallback when useful line evidence exists.

## Context

`basketball-demo` is a useful stability baseline but not the accuracy target.
Its `draw_tactical_court` path builds a mapper from hoop position, line samples,
and optional pose direction. That mapper is simpler and can look stable because
it holds/caches aggressively, but it is not a true projected-line homography.

The current branch has the better core contract:

- `court_calibration` publishes `court_calib` metadata containing a
  source-normalized image-to-court homography.
- `draw_tactical_court` consumes `court_calib` and no longer owns calibration.
- `court_proj` debug metadata makes projected lines visible on the source video.
- The PTZ camera model, template match, bounded LM refine, and luma snap are
  useful pieces.

The problem is the behavior stack around that core: source arbitration,
spillover reconciliation, sector consensus, degraded promotion, rescue paths,
rig learning, async stamp smoothing, hold gates, and optional lock tracking all
can move, reject, or publish a homography. That makes failures hard to explain.

## Proposed Architecture

Replace the calibration runtime with one explicit state machine:

1. `ACQUIRE`
2. `TRACK`
3. `INVALID`

The calibration output remains the existing `court_calib` metadata. The draw
node remains a pure consumer.

### ACQUIRE

`ACQUIRE` searches for a reliable camera when no lock exists or the previous
lock was invalidated.

Inputs:

- Court segmentation masks.
- Three-point-region segmentation mask when present.
- Hoop detection when present.
- Luma plane when available.
- Optional pose only for mid-court pan anchoring, not for end-zone line fitting.

Algorithm:

- Build one candidate with the existing PTZ template match.
- Refine once with bounded PTZ LM against the best available line evidence.
- Apply one luma snap to the refined candidate.
- Accept only if simple gates pass:
  - homography projects a sane court quadrilateral,
  - projected active three-point line agrees with current line evidence or
    luma ridge within a configured pixel tolerance,
  - hoop side is consistent when a hoop exists,
  - close-up/shot classifier says the frame is usable.

No source arbitration, sector promotion, rig update, or multi-stage rescue runs
inside `ACQUIRE`.

### TRACK

`TRACK` owns coverage. Once a lock exists, line projection should not require a
fresh strong segmentation detection every frame.

Per frame:

- Predict PTZ with the previous velocity.
- Snap within tight bounds to current evidence:
  - prefer clean boundary points if a reasonable line mask is present,
  - otherwise use luma ridge targets,
  - otherwise short-coast on decaying velocity.
- Publish the tracked homography while the lock is healthy.

Break the lock when:

- tracking fails for more than a small frame budget,
- a cut/close-up/non-wide shot is detected,
- a fresh `ACQUIRE` candidate disagrees for several consecutive solve frames,
- the projected line is contradicted by strong current evidence.

The tracker may coast briefly, but it must not synthesize large camera motion
without evidence.

### INVALID

`INVALID` publishes `valid: false` and clears any source projection overlay.
It is entered on shot cuts, close-ups, impossible quadrilaterals, persistent
tracking starvation, or source evidence that strongly contradicts the lock.

The system returns to `ACQUIRE` when a later wide frame has enough evidence.

## Components

Keep:

- `pyplumber/court_calibration.py` node API and `court_calib` payload shape.
- `pyplumber/court_segm/geometry.py` physical court and PTZ projection helpers.
- `pyplumber/court_segm/matching.py` template cold start.
- One bounded PTZ LM refine from `pyplumber/court_segm/fitting.py`.
- Luma snap and boundary snap, simplified to be side-effect-light helpers.
- `src/nodes/neural_net/draw/draw_tactical_court.cpp` as a `court_calib`
  consumer.
- `court_proj` and `court_evidence` overlays for visual debugging.

Remove or park behind disabled experimental flags:

- sector-consensus filtering,
- yellow/hole arbitration policy,
- spillover reconciliation,
- degraded promotion pass,
- per-venue rig learning,
- multi-branch rescue publishes,
- async stamp-level smoothing/extrapolation separate from tracker state.

## Data Flow

Graph flow stays the same at the interface level:

1. `cuda_infer_yolo` emits segmentation metadata and GPU masks.
2. `court_seg_evidence_cuda` emits CPU mask evidence and optional luma.
3. `AsyncCourtCalibrationNode` publishes `court_calib`, `court_proj`, and
   `court_evidence`.
4. `draw_tactical_court` reads `court_calib` and renders the tactical panel.

Internally, `AsyncCourtCalibrationNode` should own the state machine. The
worker may produce acquisition candidates, but the live frame path owns the
current lock and publishes the tracked camera. There should be one publish
funnel so every valid `court_calib` payload has the same final sanity checks.

## Error Handling

- Any exception in calibration publishes invalid metadata for that frame and
  does not stall the graph.
- A stale lock is invalidated rather than silently held forever.
- Strong contradiction from current evidence clears the lock.
- Missing weak evidence does not immediately invalidate a healthy lock; it
  allows short tracked/coasted coverage.
- Debug counters should use state names and rejection reasons that identify
  the state transition, not internal optimizer branches.

## Testing And Evaluation

Minimum checks before calling this rescued:

- Run the tactical-view clip with `AVP_CALIB_LOG` enabled.
- Inspect projected source lines (`court_proj`) against the broadcast video.
- Run `tools/line_provenance_eval.py` on the stamped calibration log.
- Compare coverage and line error against the current branch and
  `basketball-demo`.

Acceptance signals:

- Most valid frames come from `TRACK`, not repeated global acquisition.
- Invalid periods align with close-ups, cuts, or genuinely missing line view.
- Projected active three-point line does not visibly jump under weak masks.
- Debug logs are understandable without reading multiple nested rescue paths.

## Out Of Scope

- Player-dot accuracy and foot-point extraction, except as consumers of the
  resulting homography.
- Rewriting framework graph management or control protocol code.
- Changing model weights.
- Porting calibration fully to C++ in this simplification pass.
- Adding league presets beyond the existing NBA court model.
