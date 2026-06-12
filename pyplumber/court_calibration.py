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
    COURT_W, COURT_H, SEG_ARC, _MODEL_LEFT, _mirror_x,
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
        # Reject fits whose arc support spans too small an angular sweep (e.g.
        # baseline-only or near-side-only degenerate solutions).
        self.min_arc_coverage_deg = p.get("min_arc_coverage_deg", 0.0)
        # Reject obviously non-physical 3-pt candidates. Large synthetic
        # holes are usually floor/crowd/background gaps, not the painted arc.
        # <=0 disables the upper area gate.
        self.tpl_max_source_area = float(p.get("tpl_max_source_area", 0.0))
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
        self.use_far_sideline = p.get("use_far_sideline", True)
        self.far_sideline_weight = p.get("far_sideline_weight", 1.0)
        # THE publish invariant: median distance (source px) of the projected
        # model arc to the FULL 3-pt boundary evidence. The LM's err_ft is
        # measured on consensus-FILTERED points and can look perfect while
        # the consensus dropped the true arc (observed: err 1.0 ft with the
        # arc 74 px off its own mask). This gate is computed against
        # everything the mask shows, so a published line can never disagree
        # with visible evidence by more than this.
        self.evidence_gate_px = float(p.get("evidence_gate_px", 26.0))
        self.visual_evidence_p90_px = float(p.get("visual_evidence_p90_px", 90.0))
        self.hold_frames = int(p.get("hold_frames", 50))
        self.court_proj_smooth_alpha = float(
            p.get("court_proj_smooth_alpha", 0.45))
        self.court_proj_reset_px = float(p.get("court_proj_reset_px", 90.0))
        self.court_proj_max_gap_frames = int(
            p.get("court_proj_max_gap_frames", 4))
        self.border_margin_px = p.get("border_margin_px", 2.0)
        self.debug_log_every_n = p.get("debug_log_every_n", 0)
        self.calibrate_every_n = max(1, int(p.get("calibrate_every_n", 1)))
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
        self._ptz_vel = np.zeros(3)
        self._project_arc = False
        self._last_good_h = None
        self._last_good_prm = None
        self._last_good_left = True
        self._last_good_frame = -10 ** 9
        self._last_valid_h = None
        self._last_valid_prm = None
        self._last_valid_left = True
        self._last_valid_frame = -10 ** 9
        self._court_proj_xy = None
        self._court_proj_key = None
        self._court_proj_frame = -10 ** 9
        # Nominal physical camera rig. Keep it fixed; live tracking only moves
        # pan/tilt/focal so segmentation noise cannot rewrite venue geometry.
        self._rig = np.array([47.0, 110.0, 35.0])
        self._was_wide = None
        self._n_ok = 0
        self._n_published_valid = 0
        self._fail_reasons = {}
        self._model_trees = {
            True: cKDTree(_MODEL_LEFT),
            False: cKDTree(_mirror_x(_MODEL_LEFT)),
        }
        self._model_pts = {True: _MODEL_LEFT, False: _mirror_x(_MODEL_LEFT)}

    def _reset_solver_state(self):
        self._last_h = None
        self._age = 10 ** 9
        self._match_prm = None
        self._ptz_state = None
        self._ptz_vel = np.zeros(3)
        self._project_arc = False
        self._last_good_h = None
        self._last_good_prm = None
        self._last_good_frame = -10 ** 9
        self._last_valid_h = None
        self._last_valid_prm = None
        self._last_valid_frame = -10 ** 9
        self._court_proj_xy = None
        self._court_proj_key = None
        self._court_proj_frame = -10 ** 9

    def _stable_hoop_side(self, hoop, w):
        if not hoop:
            return self._last_hoop_on_left
        self._hoop_seen_frame = self._frame
        obs_left = hoop[0] < w * 0.5
        if obs_left == self._last_hoop_on_left:
            self._side_votes = 0
        else:
            self._side_votes = getattr(self, "_side_votes", 0) \
                + max(1, getattr(self, "_dt", 1))
        if self._side_votes >= 5:
            return not self._last_hoop_on_left
        return self._last_hoop_on_left

    def _invalid(self, frame, w, h, reason):
        self._fail(reason)
        self._age = min(self._age + max(1, getattr(self, "_dt", 1)), 10 ** 6)
        if self._publish_hold(frame, w, h, reason):
            self._dst.enqueue(frame)
            return
        self._publish(frame, False, None, w, h, src=reason)
        self._dst.enqueue(frame)

    def _arc_conf_ok(self, conf):
        arc = (conf or {}).get("arc")
        if not arc or len(arc) < 2:
            return False
        return arc[0] <= self.evidence_gate_px \
            and arc[1] <= self.visual_evidence_p90_px

    def _publish_hold(self, frame, w, h, reason):
        hold_h = hold_prm = None
        hold_left = True
        project_arc = False
        age = 10 ** 9
        if self._last_good_h is not None and self._last_good_prm is not None:
            good_age = self._frame - self._last_good_frame
            if 0 <= good_age <= self.hold_frames:
                conf = self._seg_conformance(
                    self._last_good_h, w, h, self._last_good_left)
                if self._arc_conf_ok(conf):
                    hold_h = self._last_good_h
                    hold_prm = self._last_good_prm
                    hold_left = self._last_good_left
                    project_arc = True
                    age = good_age
        if hold_h is None and self._last_valid_h is not None \
                and self._last_valid_prm is not None:
            valid_age = self._frame - self._last_valid_frame
            if 0 <= valid_age <= self.hold_frames:
                hold_h = self._last_valid_h
                hold_prm = self._last_valid_prm
                hold_left = self._last_valid_left
                project_arc = False
                age = valid_age
        if hold_h is None:
            return False
        self._last_h = hold_h
        self._ptz_state = hold_prm.copy()
        self._last_hoop_on_left = hold_left
        self._project_arc = project_arc
        self._age = age
        self._publish(frame, True, hold_h, w, h,
                      zone="full", src=f"hold_{reason}")
        return True

    def _remember_valid(self, h8, prm, hoop_on_left):
        self._last_valid_h = list(h8)
        self._last_valid_prm = np.asarray(prm, dtype=float).copy()
        self._last_valid_left = bool(hoop_on_left)
        self._last_valid_frame = self._frame

    def _remember_good_visual(self, h8, prm, hoop_on_left):
        self._last_good_h = list(h8)
        self._last_good_prm = np.asarray(prm, dtype=float).copy()
        self._last_good_left = bool(hoop_on_left)
        self._last_good_frame = self._frame

    def _arc_boundary_points_for_snap(self, h8, court_pts, w, h, hoop_on_left):
        if h8 is None or court_pts is None or len(court_pts) < 30:
            return None
        norm = np.array([w, h], dtype=float)
        sel, lab = self._assign_segments(
            h8, np.asarray(court_pts, dtype=float) / norm, 10.0,
            hoop_on_left)
        if sel is None:
            return None
        arc = sel[lab == SEG_ARC]
        if len(arc) < 30:
            return None
        return arc * norm

    def _try_track_current(self, frame, w, h, hoop_on_left, hoop_norm,
                           top_pts, court_pts, tpl_pts, reason):
        """Use current-frame evidence to update the last valid PTZ before
        falling back to a blind hold. This intentionally does not force the
        full arc overlay on; the arc is still gated by source-pixel evidence."""
        if self._last_valid_h is None or self._last_valid_prm is None:
            return False
        if bool(hoop_on_left) != bool(self._last_valid_left):
            self._fail(f"track_{reason}_side")
            return False
        age = self._frame - self._last_valid_frame
        if age < 0 or age > self.hold_frames:
            return False

        prm0 = self._last_valid_prm.copy()
        prm = prm0.copy()
        vel = getattr(self, "_ptz_vel", np.zeros(3))
        prm[:3] += np.clip(vel, [-0.012, -0.008, -0.05],
                           [0.012, 0.008, 0.05])
        pred_h = self._ptz_to_h8(tuple(prm), w, h)
        if pred_h is None:
            pred_h = self._last_valid_h

        track_pts = None
        if tpl_pts is not None and len(tpl_pts) >= 50:
            track_pts = tpl_pts
        elif court_pts is not None:
            track_pts = self._arc_boundary_points_for_snap(
                pred_h, court_pts, w, h, hoop_on_left)

        n_boundary = 0
        if track_pts is not None and len(track_pts) >= 30:
            snapped, n_boundary = self._boundary_snap(
                prm, w, h, hoop_on_left, track_pts, span_scale=1.5)
            if n_boundary:
                prm = np.asarray(snapped, dtype=float)

        n_luma = 0
        if n_boundary and self._luma is not None:
            snapped, n_luma = self._luma_snap(
                prm, w, h, hoop_on_left, self._luma, span_scale=0.5)
            if n_luma:
                prm = np.asarray(snapped, dtype=float)

        if not n_boundary and not n_luma:
            self._fail(f"track_{reason}_no_evidence")
            return False

        h8 = self._ptz_to_h8(tuple(prm), w, h)
        if h8 is None or not self._quad_sane(h8, w, h):
            self._fail(f"track_{reason}_bad_quad")
            return False
        if hoop_norm is not None:
            he = self._hoop_error_ptz_ft(
                h8, hoop_norm, hoop_on_left, tuple(prm[3:]))
            if he is not None and he > self.hoop_gate_ft:
                self._fail(f"track_{reason}_hoop_gate")
                return False
        if top_pts is not None:
            se = self._sideline_error_ft(
                h8, top_pts / np.array([w, h], dtype=float))
            if se is not None and se > self.sideline_gate_ft:
                self._fail(f"track_{reason}_sideline_gate")
                return False

        conf = self._seg_conformance(h8, w, h, hoop_on_left) or {}
        project_arc = self._arc_conf_ok(conf)
        if project_arc:
            self._remember_good_visual(h8, prm, hoop_on_left)
        else:
            self._fail(f"track_{reason}_visual_gate")

        self._ptz_state = prm
        delta = prm[:3] - prm0[:3]
        self._ptz_vel = 0.7 * getattr(self, "_ptz_vel", np.zeros(3)) \
            + 0.3 * delta
        self._ptz_vel = np.clip(self._ptz_vel,
                                [-0.012, -0.008, -0.05],
                                [0.012, 0.008, 0.05])
        self._last_h = h8
        self._last_hoop_on_left = hoop_on_left
        self._project_arc = project_arc
        self._remember_valid(h8, prm, hoop_on_left)
        self._age = 0
        self._n_ok += 1
        npts = 0 if track_pts is None else len(track_pts)
        self._publish(frame, True, h8, w, h,
                      npts=npts, zone="full", src=f"track_{reason}")
        return True

    def _tpl_candidate_area(self, tpl, reason):
        if tpl is None:
            return None
        area = float((tpl >= self.mask_threshold).mean())
        if area < self.min_region_area:
            self._fail(f"{reason}_small")
            return None
        if self.tpl_max_source_area > 0.0 and area > self.tpl_max_source_area:
            self._fail(f"{reason}_large")
            return None
        if area > 0.40:
            self._fail(f"{reason}_flood")
            return None
        return area

    def _acquire_tpl(self, tpls, court, hoop):
        # Prefer the raw court-mask hole when it exists: the court class often
        # outlines the 3pt region cleanly while the explicit 3pt instance is
        # partially stolen by court spillover or bbox cropping.
        hole = self._synth_tpl_from_court(court) if court is not None else None
        hole_area = self._tpl_candidate_area(hole, "acquire_hole_tpl")
        if hole_area is not None:
            self._fail("acquire_hole_tpl")
            return hole

        tpl = self._pick_tpl(tpls, hoop, 1, 1) if tpls else None
        if tpl is not None and getattr(self, "_tpl_conf", 0.0) < 0.35:
            tpl = None
        if self._tpl_candidate_area(tpl, "acquire_tpl") is None:
            return None
        return tpl

    def process(self):
        frame = self._src.get()
        if not frame:
            return
        self._frame += 1
        self._dt = max(1, self._frame - getattr(self, "_prev_frame",
                                                self._frame - 1))
        self._prev_frame = self._frame
        self._ev_pts = None
        self._luma = None
        w, hgt = frame.width, frame.height

        try:
            if self.require_wide_shot:
                shot = self._read_json(frame, self.shot_metadata_key)
                wide = shot is None or "wide" in json.dumps(shot)
                if self._was_wide is not None and wide != self._was_wide:
                    self._reset_solver_state()
                    self._fail("shot_transition_reset")
                self._was_wide = wide
                if not wide:
                    self._reset_solver_state()
                    self._invalid(frame, w, hgt, "not_wide")
                    return
                if self.closeup_player_frac > 0 and \
                        self._max_player_height_frac(frame) > self.closeup_player_frac:
                    self._reset_solver_state()
                    self._invalid(frame, w, hgt, "closeup_guard")
                    return

            if self.calibrate_every_n > 1 \
                    and (self._frame - 1) % self.calibrate_every_n != 0:
                self._invalid(frame, w, hgt, "cadence_skip")
                return

            masks = self._read_masks(frame)
            if masks is None:
                self._invalid(frame, w, hgt, "no_masks")
                return
            court, tpls = self._class_planes(frame, masks)
            self._luma = self._read_luma(frame)
            hoop = self._hoop_source_px(frame)
            hoop_on_left = self._stable_hoop_side(hoop, w)
            hoop_norm = (hoop[0] / w, hoop[1] / hgt) if hoop else None

            floor = court
            for m in tpls:
                floor = m if floor is None else np.maximum(floor, m)
            if floor is None:
                self._invalid(frame, w, hgt, "no_floor")
                return

            tpl = self._acquire_tpl(tpls, court, hoop)
            tpl_pts = self._tpl_boundary_points(tpl, w, hgt) \
                if tpl is not None else None
            court_pts = self._court_boundary_points(court, w, hgt) \
                if court is not None else None
            top_pts = self._court_top_edge(floor, w, hgt)
            bot_pts = self._court_bottom_edge(court, w, hgt) \
                if court is not None else None
            self._ev_pts = {
                "arc": tpl_pts,
                "court": court_pts,
                "far": top_pts,
                "bot": bot_pts,
            }
            line_pts = tpl_pts if tpl_pts is not None and len(tpl_pts) >= 50 \
                else court_pts

            h8 = self._lines_h(tpl, floor, hoop, w, hgt, hoop_on_left)
            prm = self._match_prm
            if h8 is None or prm is None:
                if self._try_track_current(
                        frame, w, hgt, hoop_on_left, hoop_norm,
                        top_pts, court_pts, tpl_pts, "acquire_template_failed"):
                    self._dst.enqueue(frame)
                    return
                self._invalid(frame, w, hgt, "acquire_template_failed")
                return
            prm = np.concatenate([np.asarray(prm, dtype=float)[:3],
                                  self._rig])

            err = coverage = None
            have_line_pts = line_pts is not None and len(line_pts) >= 50
            if have_line_pts and self.relative_refine:
                prm_ref, err_r, cov_r = self._refine_ptz(
                    prm, line_pts, top_pts, w, hgt, hoop_on_left, hoop_norm,
                    anchor_repeats=4, bot_pts=bot_pts)
                h_ref = self._ptz_to_h8(tuple(prm_ref), w, hgt) \
                    if prm_ref is not None else None
                if h_ref is not None and err_r is not None \
                        and err_r <= self.gate_median_ft \
                        and (cov_r is None or cov_r >= self.min_coverage) \
                        and self._quad_sane(h_ref, w, hgt):
                    he = self._hoop_error_ptz_ft(
                        h_ref, hoop_norm, hoop_on_left, tuple(prm_ref[3:])) \
                        if hoop_norm is not None else None
                    se = self._sideline_error_ft(
                        h_ref, top_pts / np.array([w, hgt], dtype=float)) \
                        if top_pts is not None else None
                    arc_cov = self._arc_coverage_deg(
                        h_ref, line_pts, w, hgt, hoop_on_left)
                    if (he is None or he <= self.hoop_gate_ft) \
                            and (se is None or se <= self.sideline_gate_ft) \
                            and arc_cov >= self.min_arc_coverage_deg:
                        prm, h8, err, coverage = prm_ref, h_ref, err_r, cov_r
                        self._fail("acquire_refined")
                    else:
                        self._fail("acquire_refine_gate")
                else:
                    self._fail("acquire_refine_reject")

            snapped, n_luma = self._luma_snap(
                prm, w, hgt, hoop_on_left, self._luma) \
                if self._luma is not None else (prm, 0)
            if n_luma:
                prm = np.asarray(snapped, dtype=float)
                h8s = self._ptz_to_h8(tuple(prm), w, hgt)
                if h8s is not None:
                    h8 = h8s
                    self._fail("acquire_luma_snap")
            n_boundary = 0
            if court_pts is not None:
                arc_pts = self._arc_boundary_points_for_snap(
                    h8, court_pts, w, hgt, hoop_on_left)
                snapped, n_try = self._boundary_snap(
                    prm, w, hgt, hoop_on_left, arc_pts, span_scale=1.5)
                if n_try:
                    cand_prm = np.asarray(snapped, dtype=float)
                    h8s = self._ptz_to_h8(tuple(cand_prm), w, hgt)
                    conf = self._seg_conformance(
                        h8s, w, hgt, hoop_on_left) or {}
                    arc_med = (conf.get("arc") or [None])[0]
                    if arc_med is not None and arc_med <= self.evidence_gate_px:
                        prm = cand_prm
                        h8 = h8s
                        n_boundary = n_try
                        self._fail("acquire_boundary_snap")
                    else:
                        self._fail("acquire_boundary_snap_reject")

            if not self._quad_sane(h8, w, hgt):
                if self._try_track_current(
                        frame, w, hgt, hoop_on_left, hoop_norm,
                        top_pts, court_pts, tpl_pts, "acquire_bad_quad"):
                    self._dst.enqueue(frame)
                    return
                self._invalid(frame, w, hgt, "acquire_bad_quad")
                return
            if hoop_norm is not None:
                he = self._hoop_error_ptz_ft(
                    h8, hoop_norm, hoop_on_left, tuple(prm[3:]))
                if he is not None and he > self.hoop_gate_ft:
                    if self._try_track_current(
                            frame, w, hgt, hoop_on_left, hoop_norm,
                            top_pts, court_pts, tpl_pts, "acquire_hoop_gate"):
                        self._dst.enqueue(frame)
                        return
                    self._invalid(frame, w, hgt, "acquire_hoop_gate")
                    return
            if top_pts is not None:
                se = self._sideline_error_ft(
                    h8, top_pts / np.array([w, hgt], dtype=float))
                if se is not None and se > self.sideline_gate_ft:
                    if self._try_track_current(
                            frame, w, hgt, hoop_on_left, hoop_norm,
                            top_pts, court_pts, tpl_pts,
                            "acquire_sideline_gate"):
                        self._dst.enqueue(frame)
                        return
                    self._invalid(frame, w, hgt, "acquire_sideline_gate")
                    return
            project_arc = False
            if have_line_pts or n_boundary:
                conf = self._seg_conformance(h8, w, hgt, hoop_on_left) or {}
                arc = conf.get("arc")
                if self._arc_conf_ok(conf):
                    project_arc = True
                elif arc:
                    self._fail("acquire_visual_gate")
                else:
                    self._fail("acquire_no_arc_conformance")
            elif n_luma == 0:
                if self._try_track_current(
                        frame, w, hgt, hoop_on_left, hoop_norm,
                        top_pts, court_pts, tpl_pts, "acquire_no_line_evidence"):
                    self._dst.enqueue(frame)
                    return
                self._invalid(frame, w, hgt, "acquire_no_line_evidence")
                return
            else:
                self._fail("acquire_luma_only_no_arc")

            self._ptz_state = np.asarray(prm, dtype=float)
            self._ptz_vel = np.zeros(3)
            self._last_h = h8
            self._last_hoop_on_left = hoop_on_left
            self._project_arc = project_arc
            self._remember_valid(h8, prm, hoop_on_left)
            if project_arc:
                self._remember_good_visual(h8, prm, hoop_on_left)
            self._age = 0
            self._n_ok += 1
            self._publish(frame, True, h8, w, hgt, err, coverage,
                          0 if line_pts is None else len(line_pts),
                          zone="full", src="acquire")
            self._dst.enqueue(frame)
            return
        except Exception as exc:
            print(f"court_calibration: frame={self._frame} error: {exc!r}",
                  flush=True)
            self._invalid(frame, w, hgt, "exception")
            return
