"""Court calibration from segmentation masks (basketball).

Fits a physical PTZ broadcast camera (pan/tilt/zoom over a learned venue
rig) to the court segmentation model output each wide frame and publishes
per-frame `court_calib` metadata consumed by draw_tactical_court.

Convention: court y=0 is the far sideline (top of the tactical view), hoop
centers at (5.25, 25) / (88.75, 25) ft on a 94x50 court.

The implementation is split across pyplumber/court_segm/ (geometry,
evidence, matching, fitting, temporal — see that package's docstring);
this file owns the node lifecycle and the per-frame orchestration in
`process()`. The legacy pose-mode/flow architecture was removed 2026-06
(it never executed in relative mode); it lives in git history before the
court_segm split.
"""

import json

import numpy as np
from scipy.spatial import cKDTree

from pyplumber.node import PythonNode
from pyplumber.court_segm.geometry import (
    COURT_W, COURT_H, SEG_NEAR_STRAIGHT, _MODEL_LEFT, _mirror_x,
    GeometryMixin,
)
from pyplumber.court_segm.evidence import EvidenceMixin
from pyplumber.court_segm.matching import MatchingMixin
from pyplumber.court_segm.fitting import FittingMixin
from pyplumber.court_segm.temporal import TemporalMixin


class CourtCalibrationNode(GeometryMixin, EvidenceMixin, MatchingMixin,
                           FittingMixin, TemporalMixin, PythonNode):
    """SISO video node: reads court-seg CPU masks, writes `court_calib` metadata."""

    def __init__(self, args: dict):
        super().__init__(args)
        p = self.parameters
        self.metadata_key_out = p.get("metadata_key_out", "court_calib")
        self.seg_metadata_key = p.get("seg_metadata_key", "yolo_seg")
        self.player_metadata_key = p.get("player_metadata_key", "yolo_players")
        self.pose_metadata_key = p.get("pose_metadata_key", "yolo_pose")
        self.template_min_iou = p.get("template_min_iou", 0.30)
        # Polish the template homography with the segment-labeled point-to-arc
        # LM and run the acceptance gates on the result. On failure (or no
        # improvement) the raw template H is kept, so this never regresses the
        # robust template path — it only removes the coarse-grid depth offset.
        self.relative_refine = p.get("relative_refine", True)
        # Inside the 3-pt arc only the 3-pt region class is physically possible;
        # the court class spilling there is a model artifact. Fill enclosed
        # holes and keep the largest component before taking the boundary, so
        # the contour is the painted arc, not a ragged bbox-cropped blob edge.
        self.clean_tpl_mask = p.get("clean_tpl_mask", True)
        # Spillover reconciliation: when the court class floods the arc interior
        # the 3-pt blob collapses and the fit bails (region_small). Using a prior
        # homography (held fit, else the spillover-robust template IoU match),
        # reproject the court model and move any court-class activation that
        # lands inside the analytic 3-pt region back into the 3-pt mask. Unlike
        # clean_tpl_mask (which only fills enclosed holes), this recovers area
        # the court class stole, restoring a whole region for the boundary fit.
        # No-op when no prior is available, so it can never regress a frame.
        self.reconcile_spillover = p.get("reconcile_spillover", True)
        # Reclaim only the deep interior: keep a safety band (court feet) next
        # to every painted line out of the reclaim, so the genuine 3-pt mask
        # boundary near the arc/baseline/straights is never overwritten by a
        # prior-H-shaped edge. Fills the flooded interior (passes the area gate)
        # without corrupting the boundary the arc refine fits to.
        self.reconcile_margin_ft = p.get("reconcile_margin_ft", 2.0)
        # Reject fits whose arc support spans too small an angular sweep (e.g.
        # baseline-only or near-side-only degenerate solutions).
        self.min_arc_coverage_deg = p.get("min_arc_coverage_deg", 0.0)
        # Sector-consensus spillover rejection: bin boundary points along each
        # model segment under the template prior and drop whole bins whose
        # median offset deviates from the segment consensus by more than this
        # (ft). Spillover is spatially contiguous, so bin voting under a
        # fit-independent prior beats per-point trimming under the fit being
        # made. 0 disables.
        self.sector_outlier_ft = p.get("sector_outlier_ft", 2.0)
        # Distrust the frame (skip the refine, keep template+EMA) when the
        # consensus filter drops more than this fraction of arc points.
        self.sector_max_drop = p.get("sector_max_drop", 0.45)
        self.shot_metadata_key = p.get("camera_shot_metadata_key", "camera_shot_info")
        self.seg_slot = p.get("seg_slot", 0)
        self.mask_threshold = p.get("mask_threshold", 0.5)
        # CPU seg masks cover the PADDED model frame (pad_cuda 960:544:0:2 ->
        # 540 content rows at y-offset 2 of 544). Mapping mask rows straight
        # onto the source warps y by up to ~4 px at 1080p (same bug class as
        # the known metadata_dump pad bug); these describe the pad so all
        # mask<->source mappings undo it.
        self.mask_model_w = p.get("mask_model_w", 960.0)
        self.mask_model_h = p.get("mask_model_h", 544.0)
        self.mask_pad_x = p.get("mask_pad_x", 0.0)
        self.mask_pad_y = p.get("mask_pad_y", 2.0)
        self.mask_content_w = p.get("mask_content_w", 960.0)
        self.mask_content_h = p.get("mask_content_h", 540.0)
        self.gate_median_ft = p.get("gate_median_ft", 1.5)
        # Loose: the rim projects onto the floor displaced from the hoop
        # ground point (it is 10 ft up), so even perfect fits show several ft.
        self.hoop_gate_ft = p.get("hoop_gate_ft", 12.0)
        self.sideline_gate_ft = p.get("sideline_gate_ft", 5.0)
        self.min_region_area = p.get("min_region_area", 0.04)
        self.min_coverage = p.get("min_coverage", 0.45)
        self.hold_frames = p.get("hold_frames", 18)
        self.use_far_sideline = p.get("use_far_sideline", True)
        self.far_sideline_weight = p.get("far_sideline_weight", 1.0)
        self.border_margin_px = p.get("border_margin_px", 2.0)
        self.debug_log_every_n = p.get("debug_log_every_n", 0)
        # Per-frame published-calibration ndjson (for offline overlay-quality
        # evaluation against the clean source video). Empty disables.
        self.calib_log = p.get("calib_log", "")
        self._calib_log_fh = None
        self.require_wide_shot = p.get("require_wide_shot", True)
        # The shot classifier sometimes calls a close-up "wide" (its player
        #-count override); a player taller than this fraction of the frame is
        # decisive evidence to the contrary — no calibration is published.
        self.closeup_player_frac = p.get("closeup_player_frac", 0.5)

        self._last_h = None
        self._last_hoop_on_left = True
        self._age = 10 ** 9
        self._frame = 0
        self._match_prm = None
        self._match_iou = 0.0
        self._ptz_state = None
        # Per-venue camera rig estimate (cx fixed; cy/cz learned online from
        # anchored arc fits). A wrong rig is STRUCTURAL overlay error that
        # pan/tilt/zoom cannot remove — offline probing showed the nominal
        # (110, 35) costs ~1-2 ft of curve residual on the test venue whose
        # true rig is ~(148, 37). Venue-constant: survives shot resets.
        self._rig = np.array([47.0, 110.0, 35.0])
        self._rig_n = 0
        self.rig_update_every = p.get("rig_update_every", 12)
        # last STRONG (curve- or rim-anchored) camera, for the jump gate on
        # weak template-only publishes
        self._conf_prm = None
        self._conf_frame = -10 ** 9
        self._was_wide = None
        self._n_ok = 0
        self._n_published_valid = 0
        self._fail_reasons = {}
        self._model_trees = {
            True: cKDTree(_MODEL_LEFT),
            False: cKDTree(_mirror_x(_MODEL_LEFT)),
        }
        self._model_pts = {True: _MODEL_LEFT, False: _mirror_x(_MODEL_LEFT)}

    def process(self):
        frame = self._src.get()
        if not frame:
            return
        self._frame += 1
        w, hgt = frame.width, frame.height

        try:
            if self.require_wide_shot:
                shot = self._read_json(frame, self.shot_metadata_key)
                wide = shot is None or "wide" in json.dumps(shot)
                if self._was_wide is not None and wide != self._was_wide:
                    # shot transition: held homography belongs to the previous
                    # camera framing — drop all calibration state
                    self._last_h = None
                    self._age = 10 ** 9
                    self._match_prm = None
                    self._ptz_state = None
                    self._conf_prm = None
                    self._conf_frame = -10 ** 9
                    self._fail("shot_transition_reset")
                self._was_wide = wide
                if not wide:
                    self._fail("not_wide")
                    self._publish(frame, False, None, w, hgt)
                    self._dst.enqueue(frame)
                    return
                if self.closeup_player_frac > 0 and \
                        self._max_player_height_frac(frame) > self.closeup_player_frac:
                    # misclassified close-up: garbage masks here fit smooth
                    # blobs the LM loves (warm-started near the old camera, so
                    # the continuity gate cannot catch them either) — refuse
                    # to publish anything
                    self._fail("closeup_guard")
                    self._age += 1
                    self._publish(frame, False, None, w, hgt)
                    self._dst.enqueue(frame)
                    return

            masks = self._read_masks(frame)
            if masks is None:
                self._fail("no_masks")
                self._hold_or_invalid(frame, w, hgt)
                return
            court, tpls = self._class_planes(frame, masks)
            hoop = self._hoop_source_px(frame)
            # FLOOR = union of all floor-class activations (blue + yellow
            # legitimately co-fire): outline evidence (far sideline, baseline)
            # that survives either class dropping out.
            floor = court
            for m in tpls:
                floor = m if floor is None else np.maximum(floor, m)
            # Dual-source 3-pt evidence — EITHER alone is sufficient:
            #  - yellow: the 3-pt class mask, when its detection is confident
            #  - hole: the un-fired region enclosed by the floor (when the
            #    3-pt class misses, the floor classes still exclude the
            #    painted region — the hole boundary IS the arc)
            # Two substantial 3-pt regions visible at once = mid-court view
            # (the regions exist only at the two ends; both fit in frame only
            # when the camera is centered). Their midpoint maps to the court
            # center x=47 — a genuine pan anchor for the mid fit.
            self._two_region_mid = None
            cents = []
            for m in tpls:
                binm = m >= self.mask_threshold
                if float(binm.mean()) >= 0.01:
                    ys, xs = np.nonzero(binm)
                    xn, yn = self._mask_idx_to_norm(xs.mean(), ys.mean(),
                                                    m.shape[1], m.shape[0])
                    cents.append((float(xn), float(yn)))
            two_regions = (len(cents) >= 2
                           and max(c[0] for c in cents)
                           - min(c[0] for c in cents) > 0.45)
            if two_regions:
                cents.sort(key=lambda c: c[0])
                self._two_region_mid = (
                    0.5 * (cents[0][0] + cents[-1][0]),
                    0.5 * (cents[0][1] + cents[-1][1]))
            tpl_yellow = self._pick_tpl(tpls, hoop, w, hgt) if tpls else None
            if tpl_yellow is not None and getattr(self, "_tpl_conf", 0.0) < 0.4:
                tpl_yellow = None
            tpl_hole = self._synth_tpl_from_court(floor)
            if tpl_hole is not None:
                self._fail("tpl_from_hole")
            if tpl_yellow is not None and tpl_hole is not None:
                tpls = [np.maximum(tpl_yellow, tpl_hole)]
            elif tpl_yellow is not None or tpl_hole is not None:
                tpls = [tpl_yellow if tpl_yellow is not None else tpl_hole]
            else:
                tpls = []

            # Hard-cut detector on the BLUE court mask (the temporally stable
            # class — the union flickers with yellow dropouts and produced
            # 300 false cuts in t60): the shot classifier misses some cuts,
            # but the court outline cannot jump frame-to-frame under real
            # camera motion. A cut breaks the chain of trust — drop all
            # calibration state; only a rim-anchored fit can re-seed it.
            if court is not None:
                cur_ds = self._downsample_mask(court)
                prev_ds = getattr(self, "_prev_court_ds", None)
                if prev_ds is not None and cur_ds.sum() >= 50 \
                        and prev_ds.sum() >= 50:
                    inter = float(cur_ds @ prev_ds)
                    iou = inter / max(1.0, float(cur_ds.sum() + prev_ds.sum())
                                      - inter)
                    # real camera motion keeps consecutive court masks above
                    # this at template resolution (pans alias ~0.8-0.9; do
                    # NOT raise further — t54 showed 0.85 trips constantly on
                    # legit pans while close-up floor masks are stable);
                    # genuine cuts fall well below
                    if iou < 0.72:
                        self._last_h = None
                        self._age = 10 ** 9
                        self._match_prm = None
                        self._ptz_state = None
                        self._conf_prm = None
                        self._conf_frame = -10 ** 9
                        self._fail("mask_cut_reset")
                self._prev_court_ds = cur_ds
            tpl = self._pick_tpl(tpls, hoop, w, hgt) if tpls else None
            # Hoop-side hysteresis: a single false hoop detection on the
            # wrong side mirrors the whole court model — a several-hundred-px
            # one-frame bounce. Flip only after several consecutive frames
            # agree on the new side.
            if hoop:
                self._hoop_seen_frame = self._frame
                obs_left = hoop[0] < w * 0.5
                if obs_left == self._last_hoop_on_left:
                    self._side_votes = 0
                else:
                    self._side_votes = getattr(self, "_side_votes", 0) + 1
                hoop_on_left = (not self._last_hoop_on_left) \
                    if self._side_votes >= 5 else self._last_hoop_on_left
            else:
                hoop_on_left = self._last_hoop_on_left
            hoop_norm = (hoop[0] / w, hoop[1] / hgt) if hoop else None

            # Mid-court pan mode (the HOOP class is the zone detector): no
            # rim for ~1s means we are not at a court end. 3-pt evidence
            # seen mid-pan is usually a GENUINE but frame-clipped region
            # tail (not hallucination) — and a clipped sliver cannot pin
            # pan/depth, which is exactly how the arc ends up drawn
            # hundreds of px from the visible region. So arcs are not even
            # attempted here. Keep tilt/zoom glued to the floor outline;
            # court-pose keypoints (when present) re-anchor pan and earn the
            # center line. (Future: the sliver's boundary is legitimate pan
            # evidence for THIS fit, like the pose points.)
            no_hoop_long = self._frame - getattr(
                self, "_hoop_seen_frame", -10 ** 9) > 25
            pan_mid = self._ptz_state is not None \
                and abs(float(self._ptz_state[0])) < 0.18
            if two_regions \
                    or (no_hoop_long
                        and (pan_mid or (tpl_yellow is None
                                         and tpl_hole is None))):
                self._mid_pose_pts = self._parse_mid_pose(frame)
                if self._mid_pan_relative(frame, floor, w, hgt):
                    return
                self._hold_or_invalid(frame, w, hgt, floor)
                return

            # Reclaim 3-pt area lost to court-class spillover before the area
            # gate, so flooded frames can still fit instead of collapsing to
            # region_small. Prior H: the held fit when same-side and fresh,
            # else a coarse template match (IoU is robust to spillover). No
            # prior -> skip, never regressing. The reconciled mask feeds ONLY
            # the area gate and the template IoU match; boundary points for
            # the curve fit come from the pristine cleaned mask (tpl_fit), so
            # a prior-H-shaped reclaimed edge can never enter the fit — the
            # sector consensus drops the flooded sector instead.
            tpl_fit = tpl
            if self.reconcile_spillover and tpl is not None and court is not None:
                if self._last_h is not None \
                        and hoop_on_left == self._last_hoop_on_left \
                        and self._age <= self.hold_frames:
                    prior_h = self._last_h
                else:
                    # Auxiliary template match for the prior only; snapshot the
                    # cross-frame param EMA so the real fit below owns the state.
                    saved_prm = getattr(self, "_match_prm", None)
                    saved_iou = getattr(self, "_match_iou", 0.0)
                    prior_h = self._lines_h(tpl, floor, hoop, w, hgt,
                                            hoop_on_left)
                    self._match_prm = saved_prm
                    self._match_iou = saved_iou
                if prior_h is not None:
                    tpl, court = self._reconcile_spillover(
                        tpl, court, prior_h, w, hgt)
                    self._fail("reconcile_applied")
                else:
                    self._fail("reconcile_no_prior")

            if tpl is None or float((tpl >= self.mask_threshold).mean()) < self.min_region_area:
                self._fail("region_small")
                if court is None:
                    self._hold_or_invalid(frame, w, hgt, floor)
                    return
                # no usable 3-pt mask, but the court contour itself carries
                # arc evidence — proceed on floor-only matching + court-
                # contour points (appended below)
                tpl = None

            # boundary points from EACH passing source independently (the
            # union mask's merged edge would cross where the sources meet)
            pts_list = []
            for src in (tpl_yellow, tpl_hole):
                if src is None:
                    continue
                p = self._tpl_boundary_points(src, w, hgt)
                if p is not None and len(p):
                    pts_list.append(p)
            tpl_pts = np.vstack(pts_list) if pts_list else None
            pts_pristine = True
            if (tpl_pts is None or len(tpl_pts) < 50) and tpl is not tpl_fit:
                # pristine masks too collapsed for a boundary; the reconciled
                # mask can still drive the template match, but its reclaimed
                # prior-shaped edge must not enter the curve refine
                tpl_pts = self._tpl_boundary_points(tpl, w, hgt)
                pts_pristine = False
            if (tpl_pts is None or len(tpl_pts) < 50) and court is None:
                self._fail("few_boundary_pts")
                if self._rescue_relative(
                        frame, floor, hoop_norm, hoop_on_left, w, hgt):
                    return
                self._hold_or_invalid(frame, w, hgt, floor)
                return
            top_pts = self._court_top_edge(floor, w, hgt) if floor is not None else None
            bot_pts = self._court_bottom_edge(court, w, hgt) if court is not None else None

            if self.debug_log_every_n and self._frame % self.debug_log_every_n == 0:
                print(f"court_calibration: diag frame={self._frame} "
                      f"fits_ok={self._n_ok} published_valid={self._n_published_valid} "
                      f"tplarea={float((tpl >= self.mask_threshold).mean()):.3f} "
                      f"pts={0 if tpl_pts is None else len(tpl_pts)} "
                      f"hoop={hoop is not None} fails={self._fail_reasons}",
                      flush=True)

            h8 = self._lines_h(tpl, floor, hoop, w, hgt, hoop_on_left)
            if h8 is None:
                self._fail("relative_unavailable")
                if self._rescue_relative(frame, floor, hoop_norm,
                                         hoop_on_left, w, hgt):
                    return
                self._hold_or_invalid(frame, w, hgt, floor)
                return
            # The court mask's OWN contour carries arc evidence even when
            # it spills inside the 3-pt region (no enclosed hole): the
            # spill front deviates from the arc while the on-line parts
            # agree — the sector consensus separates them. The near-side
            # court edge is the courtside crowd, never the painted near
            # straight: those labels are dropped unconditionally.
            if court is not None and pts_pristine:
                cpts = self._tpl_boundary_points(court, w, hgt)
                if cpts is not None and len(cpts):
                    cpts = cpts[:: max(1, len(cpts) // 200)]
                    norm_wh = np.array([w, hgt], dtype=float)
                    sel, lab = self._assign_segments(
                        h8, cpts / norm_wh, 12.0, hoop_on_left)
                    if sel is not None and len(sel):
                        ok_lab = lab != SEG_NEAR_STRAIGHT
                        if ok_lab.any():
                            extra = sel[ok_lab] * norm_wh
                            tpl_pts = extra if tpl_pts is None \
                                else np.vstack([tpl_pts, extra])
            if tpl_pts is None or len(tpl_pts) < 50:
                self._fail("few_boundary_pts")
                if self._rescue_relative(frame, floor, hoop_norm,
                                         hoop_on_left, w, hgt):
                    return
                self._hold_or_invalid(frame, w, hgt, floor)
                return
            # Polish the robust template camera against the painted arc in
            # PTZ space (pan/tilt/zoom only): removes the template grid's
            # ~1-2 ft depth quantization without the failure modes of a
            # free 8-DOF polish — a localized segmentation error (court
            # flooding the arc interior, a bulge past the arc) cannot
            # collapse or inflate the arc when the fit has no perspective
            # DOF to bend. Gate the result; on any failure keep the raw
            # template H so this can't regress.
            err = coverage = None
            hoop_err = hoop_err_t = None
            prm = self._match_prm
            if prm is not None:
                # the template grid runs on the nominal rig; everything
                # downstream uses the learned venue rig
                prm = np.concatenate([np.asarray(prm, dtype=float)[:3],
                                      self._rig])
            # Sector-consensus spillover rejection before the refine:
            # classify boundary bins under the fit-independent template
            # prior; a frame whose arc evidence is mostly contaminated is
            # not refined at all (template+EMA coast instead).
            fit_pts = tpl_pts if pts_pristine else None
            degraded = False
            if fit_pts is not None and self.sector_outlier_ft > 0:
                fit_pts, _arc_drop, degraded = self._consensus_filter(
                    h8, tpl_pts, w, hgt, hoop_on_left)
                if fit_pts is None:
                    self._fail("sector_distrust")
                elif degraded:
                    self._fail("sector_degraded")
            # A degraded (arc-free) fit has no absolute scale without the
            # rim; don't attempt it on no-hoop frames.
            if degraded and hoop_norm is None:
                fit_pts = None
            top_n = top_pts / np.array([w, hgt], dtype=float) \
                if top_pts is not None else None
            if self.relative_refine and prm is not None and fit_pts is not None:
                prm_ref, err_r, cov_r = self._refine_ptz(
                    prm, fit_pts, top_pts, w, hgt, hoop_on_left, hoop_norm,
                    anchor_repeats=12 if degraded else 4, bot_pts=bot_pts)
                # Promotion: a depth-warped template prior makes the TRUE
                # arc bins deviate non-constantly and can misjudge a clean
                # arc as majority-contaminated (degraded). Re-classify the
                # boundary under the rim-anchored degraded fit — a far
                # better prior. If the arc now forms a clean majority, fit
                # it fully; the standard gates below still decide. True
                # floods stay degraded: their bins disagree with any sane
                # prior. This recovers exact arc depth from the painted
                # line instead of leaving it to the rim anchor's few-ft
                # rig tolerance.
                if degraded and prm_ref is not None:
                    h_deg = self._ptz_to_h8(tuple(prm_ref), w, hgt)
                    if h_deg is not None:
                        fp2, _ad2, deg2 = self._consensus_filter(
                            h_deg, tpl_pts, w, hgt, hoop_on_left)
                        if fp2 is not None and not deg2:
                            prm2, err2, cov2 = self._refine_ptz(
                                prm_ref, fp2, top_pts, w, hgt,
                                hoop_on_left, hoop_norm, anchor_repeats=4,
                                bot_pts=bot_pts)
                            h2 = self._ptz_to_h8(tuple(prm2), w, hgt) \
                                if prm2 is not None else None
                            he2 = self._hoop_error_ptz_ft(
                                h2, hoop_norm, hoop_on_left,
                                tuple(prm2[3:])) if h2 is not None else None
                            # adopt the promotion only on its own merits;
                            # otherwise keep the degraded result
                            if h2 is not None and err2 is not None \
                                    and err2 <= self.gate_median_ft \
                                    and (he2 is None
                                         or he2 <= self.hoop_gate_ft) \
                                    and self._arc_coverage_deg(
                                        h2, fp2, w, hgt, hoop_on_left) \
                                    >= self.min_arc_coverage_deg:
                                prm_ref, err_r, cov_r = prm2, err2, cov2
                                fit_pts = fp2
                                degraded = False
                                self._fail("sector_promoted")
                h_ref = self._ptz_to_h8(tuple(prm_ref), w, hgt) \
                    if prm_ref is not None else None
                hoop_err = self._hoop_error_ptz_ft(
                    h_ref, hoop_norm, hoop_on_left, tuple(prm_ref[3:])) \
                    if h_ref is not None else None
                side_r = self._sideline_error_ft(h_ref, top_n) \
                    if h_ref is not None else None
                # A solution pinned to a pan/tilt/zoom bound is the
                # optimizer escaping toward a corner chasing a flooded
                # mask (real broadcast params sit well inside); reject it.
                at_bound = prm_ref is not None and bool(
                    np.any(prm_ref[:3] <= np.array(self._PTZ_LO) + 1e-3)
                    or np.any(prm_ref[:3] >= np.array(self._PTZ_HI) - 1e-3))
                # Degraded fits MUST be rim-anchored; normal fits accept a
                # missing hoop (the arc supplies scale).
                hoop_ok = hoop_err <= self.hoop_gate_ft \
                    if hoop_err is not None else not degraded
                if h_ref is not None and err_r is not None \
                        and err_r <= self.gate_median_ft \
                        and cov_r is not None and cov_r >= self.min_coverage \
                        and not at_bound and hoop_ok \
                        and (side_r is None or side_r <= self.sideline_gate_ft) \
                        and self._quad_sane(h_ref, w, hgt):
                    arc_deg = self._arc_coverage_deg(
                        h_ref, fit_pts, w, hgt, hoop_on_left) \
                        if not degraded else None
                    if degraded or arc_deg >= self.min_arc_coverage_deg:
                        prm = prm_ref
                        # the template matcher's cross-frame EMA continues
                        # from the refined camera, not the raw grid winner
                        self._match_prm = prm_ref
                        err, coverage = err_r, cov_r
                        self._fail("relative_refined_degraded" if degraded
                                   else "relative_refined")
                        # learn the venue rig from rim-anchored arc fits:
                        # free (cy, cz) and EMA when the solution is good
                        # and interior (fast at first, then slow)
                        if hoop_err is not None and not degraded \
                                and self._frame % self.rig_update_every == 0:
                            prm5, err5, _c5 = self._refine_ptz(
                                prm_ref, fit_pts, top_pts, w, hgt,
                                hoop_on_left, hoop_norm,
                                anchor_repeats=4, fit_rig=True)
                            if prm5 is not None and err5 is not None \
                                    and err5 <= 1.5 \
                                    and 85.0 < prm5[4] < 255.0 \
                                    and 19.0 < prm5[5] < 59.0:
                                a = 0.25 if self._rig_n < 8 else 0.05
                                self._rig[1:] = (1.0 - a) * self._rig[1:] \
                                    + a * prm5[4:6]
                                self._rig_n += 1
                    else:
                        self._fail("relative_arc_cov")
                else:
                    self._fail("relative_refine_reject")
            # When the refine was NOT adopted the raw template H is about
            # to be published — it matched a possibly-flooded mask by IoU
            # with no absolute scale check. The rim anchor is that check:
            # a template whose rim projection is far from the analytic
            # ground point is calibrated to the flood, not the court.
            if err is None and prm is not None:
                # weak path needs a confident template: a garbage mask
                # (misclassified close-up) can scrape past min_iou 0.30
                if getattr(self, "_match_iou", 0.0) \
                        < max(0.45, self.template_min_iou):
                    self._fail("relative_lowiou")
                    self._hold_or_invalid(frame, w, hgt, floor)
                    return
                hoop_err_t = self._hoop_error_ptz_ft(
                    h8, hoop_norm, hoop_on_left, tuple(prm[3:])) \
                    if hoop_norm is not None else None
                if hoop_err_t is not None and hoop_err_t > self.hoop_gate_ft:
                    # A single contradicted frame is bad EVIDENCE (e.g. a
                    # bad hole synthesis or seg glitch) — try the
                    # rim-anchored rescue, else hold. Only a PERSISTENT
                    # contradiction means the state itself is poisoned
                    # and must be dropped.
                    self._contra_streak = getattr(
                        self, "_contra_streak", 0) + 1
                    if self._contra_streak < 5:
                        self._fail("relative_hoop_skip")
                        if self._rescue_relative(frame, floor, hoop_norm,
                                                 hoop_on_left, w, hgt):
                            return
                        self._hold_or_invalid(frame, w, hgt, floor)
                        return
                    self._fail("relative_hoop_reset")
                    self._contra_streak = 0
                    self._last_h = None
                    self._ptz_state = None
                    self._match_prm = None
                    self._conf_prm = None
                    self._conf_frame = -10 ** 9
                    self._age = 10 ** 9
                    self._hold_or_invalid(frame, w, hgt)
                    return
                side_t = self._sideline_error_ft(h8, top_n)
                if side_t is not None and side_t > 2.0 * self.sideline_gate_ft:
                    self._fail("relative_side_gate")
                    self._hold_or_invalid(frame, w, hgt, floor)
                    return
            anchored = (err is not None and hoop_err is not None
                        and hoop_err <= self.hoop_gate_ft) \
                or (err is None and hoop_err_t is not None)
            # Without the rim, the only evidence is the 3-pt mask itself —
            # require the seg model to be confident in it. Misclassified
            # close-ups hallucinate low-confidence 3-pt blobs that even
            # the curve refine fits beautifully (probe finding).
            if not anchored and getattr(self, "_tpl_conf", 0.0) < 0.4:
                self._fail("relative_lowconf")
                self._hold_or_invalid(frame, w, hgt, floor)
                return
            if not self._adopt_relative(frame, prm, h8, hoop_on_left, w,
                                        hgt, err, coverage, len(tpl_pts),
                                        strong=err is not None,
                                        anchored=anchored):
                self._hold_or_invalid(frame, w, hgt, floor)
            return
        except Exception as exc:  # never stall the pipeline
            print(f"court_calibration: frame={self._frame} error: {exc!r}")
            self._hold_or_invalid(frame, w, hgt)
            return
