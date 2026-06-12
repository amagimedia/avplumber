"""Calibration output helpers.

The live calibration state machine lives in `CourtCalibrationNode`.
This mixin publishes metadata and debug overlays so every valid homography
goes through one output format.
"""

import json

import numpy as np
from scipy.spatial import cKDTree

from pyplumber.court_segm.geometry import COURT_W, COURT_H, SEG_ARC


class TemporalMixin:
    """Publishing helpers for CourtCalibrationNode."""

    def _arc_evidence_points(self, h8, w, h, hoop_on_left):
        ev = getattr(self, "_ev_pts", None)
        if not ev:
            return None
        arc = ev.get("arc")
        if arc is not None and len(arc) >= 10:
            return np.asarray(arc, dtype=float)
        court = ev.get("court")
        if h8 is None or court is None or len(court) < 30:
            return None
        norm = np.array([w, h], dtype=float)
        sel, lab = self._assign_segments(
            h8, np.asarray(court, dtype=float) / norm, 10.0, hoop_on_left)
        if sel is None:
            return None
        arc = sel[lab == SEG_ARC] * norm
        return arc if len(arc) >= 10 else None

    def _smooth_court_projection(self, xy, key, frame_index):
        alpha = float(getattr(self, "court_proj_smooth_alpha", 1.0))
        if alpha <= 0.0 or alpha >= 1.0:
            self._court_proj_xy = xy.copy()
            self._court_proj_key = key
            self._court_proj_frame = frame_index
            return xy
        prev = getattr(self, "_court_proj_xy", None)
        prev_key = getattr(self, "_court_proj_key", None)
        prev_frame = getattr(self, "_court_proj_frame", -10 ** 9)
        gap = frame_index - prev_frame
        reset = prev is None or prev_key != key \
            or prev.shape != xy.shape \
            or gap < 0 \
            or gap > int(getattr(self, "court_proj_max_gap_frames", 4))
        ok = np.isfinite(xy).all(axis=1)
        if not reset:
            disp = np.hypot(xy[:, 0] - prev[:, 0], xy[:, 1] - prev[:, 1])
            disp = disp[ok & np.isfinite(prev).all(axis=1)]
            if len(disp):
                gate = float(getattr(self, "court_proj_reset_px", 90.0))
                reset = float(np.median(disp)) > gate \
                    or float(np.percentile(disp, 90)) > gate * 2.0
            else:
                reset = True
        if reset:
            out = xy.copy()
        else:
            out = xy.copy()
            both = ok & np.isfinite(prev).all(axis=1)
            out[both] = alpha * xy[both] + (1.0 - alpha) * prev[both]
        self._court_proj_xy = out.copy()
        self._court_proj_key = key
        self._court_proj_frame = frame_index
        return out

    def _emit_court_projection(self, frame, h8, w, h, hoop_on_left,
                               mid=False, center=False, include_arc=True,
                               frame_index=None):
        """Publish the projected court model as pose-style keypoints.

        `draw_keypoints` overlays this on the source frame. This is a debug
        projection of the published homography, so it must stay in lockstep
        with `court_calib.h`; `include_arc` is kept for old callers but no
        longer hides the arc.
        """
        try:
            hm = np.array([[h8[0], h8[1], h8[2]],
                           [h8[3], h8[4], h8[5]],
                           [h8[6], h8[7], 1.0]])
            hi = np.linalg.inv(hm)
        except np.linalg.LinAlgError:
            return

        curves = []
        if not mid:
            curves.append(self._model_pts[hoop_on_left][::3])
        line = np.linspace(0.0, COURT_W, 60)
        curves.append(np.stack([line, np.zeros_like(line)], axis=1))
        curves.append(np.stack([line, np.full_like(line, COURT_H)], axis=1))
        if not mid or center:
            cl = np.linspace(0.0, COURT_H, 30)
            curves.append(np.stack([np.full_like(cl, 47.0), cl], axis=1))

        pts = np.vstack(curves) / np.array([COURT_W, COURT_H])
        q = (hi @ np.hstack([pts, np.ones((len(pts), 1))]).T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            xy = q[:, :2] / q[:, 2:3] * np.array([w, h])
        if frame_index is None:
            frame_index = int(getattr(self, "_frame", 0))
        key = (bool(hoop_on_left), bool(mid), bool(center), len(xy), int(w), int(h))
        xy = self._smooth_court_projection(xy, key, int(frame_index))
        ok = np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) \
            & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        xy = xy[ok]
        if not len(xy):
            return

        kpts = []
        for x, y in xy:
            kpts += [round(float(x), 1), round(float(y), 1), 1.0]
        frame.metadata["court_proj"] = json.dumps(
            {"model_width": w, "model_height": h,
             "num_keypoints": len(xy),
             "poses": [{"conf": 1.0, "keypoints": kpts}]})

    def _seg_conformance(self, h8, w, h, hoop_on_left=None):
        """Distance in source pixels from projected model curves to the
        current segmentation evidence cached by the acquire step."""
        ev = getattr(self, "_ev_pts", None)
        if not ev or h8 is None:
            return None
        try:
            hm = np.array([[h8[0], h8[1], h8[2]],
                           [h8[3], h8[4], h8[5]],
                           [h8[6], h8[7], 1.0]])
            hi = np.linalg.inv(hm)
        except np.linalg.LinAlgError:
            return None

        line = np.linspace(0.0, COURT_W, 60)
        side = self._last_hoop_on_left if hoop_on_left is None \
            else bool(hoop_on_left)
        model = {
            "arc": self._model_pts[side][::3],
            "far": np.stack([line, np.zeros_like(line)], axis=1),
            "bot": np.stack([line, np.full_like(line, COURT_H)], axis=1),
        }
        out = {}
        for key, curve in model.items():
            evp = ev.get(key)
            if key == "arc" and (evp is None or len(evp) < 10):
                evp = self._arc_evidence_points(h8, w, h, side)
            if evp is None or len(evp) < 10:
                continue
            pts = np.asarray(curve, dtype=float) / np.array([COURT_W, COURT_H])
            q = (hi @ np.hstack([pts, np.ones((len(pts), 1))]).T).T
            with np.errstate(divide="ignore", invalid="ignore"):
                xy = q[:, :2] / q[:, 2:3] * np.array([w, h])
            ok = np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) \
                & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
            xy = xy[ok]
            if len(xy) < 5:
                continue
            d, _ = cKDTree(np.asarray(evp, dtype=float)).query(xy)
            out[key] = [round(float(np.median(d)), 1),
                        round(float(np.percentile(d, 90)), 1), int(len(xy))]
        return out or None

    def _publish(self, frame, valid, h8, w, h, err=None, coverage=None,
                 npts=0, zone="full", center=False, src=None):
        out = {
            "valid": bool(valid),
            "source_w": w,
            "source_h": h,
            "hoop_on_left": bool(self._last_hoop_on_left),
            "age": min(getattr(self, "_age", 10 ** 6), 10 ** 6),
            "zone": zone,
            "points": int(npts),
        }
        if src:
            out["src"] = src
        if h8 is not None:
            out["h"] = [float(v) for v in h8] + [1.0]
        if valid:
            self._n_published_valid += 1
            if h8 is not None:
                self._emit_court_projection(
                    frame, h8, w, h, self._last_hoop_on_left,
                    mid=(zone == "mid"), center=center,
                    include_arc=True, frame_index=self._frame)
            arc_ev = self._arc_evidence_points(
                h8, w, h, bool(self._last_hoop_on_left))
            if arc_ev is not None and len(arc_ev):
                pts = np.asarray(arc_ev, dtype=float)
                if len(pts) > 80:
                    pts = pts[:: max(1, len(pts) // 80)]
                kp = []
                for x, y in pts:
                    kp += [round(float(x), 1), round(float(y), 1), 1.0]
                frame.metadata["court_evidence"] = json.dumps(
                    {"model_width": w, "model_height": h,
                     "num_keypoints": len(pts),
                     "poses": [{"conf": 1.0, "keypoints": kp}]})
                out["ev"] = [[round(float(x), 1), round(float(y), 1)]
                             for x, y in pts]
            if getattr(self, "_ptz_state", None) is not None:
                out["ptz"] = [float(v) for v in self._ptz_state]
                out["ptz_vel"] = [float(v) for v in
                                  getattr(self, "_ptz_vel", np.zeros(3))]
                out["rig"] = [float(v) for v in self._rig]
            out["project_arc"] = bool(getattr(self, "_project_arc", True))
        if err is not None:
            out["err_ft"] = round(float(err), 3)
        if coverage is not None:
            out["coverage"] = round(float(coverage), 3)

        frame.metadata[self.metadata_key_out] = json.dumps(out)
        self._log_calibration(out, h8, w, h)

    def _log_calibration(self, payload, h8, w, h):
        if not self.calib_log:
            return
        try:
            if self._calib_log_fh is None:
                self._calib_log_fh = open(self.calib_log, "w")
            rec = {
                "frame": self._frame,
                "valid": bool(payload.get("valid")),
                "left": bool(payload.get("hoop_on_left", True)),
                "age": int(payload.get("age", 10 ** 6)),
                "zone": payload.get("zone", "full"),
                "src": payload.get("src", ""),
                "rig": [round(float(v), 1) for v in self._rig],
            }
            if h8 is not None:
                rec["h"] = [float(v) for v in h8]
            if payload.get("err_ft") is not None:
                rec["err_ft"] = payload["err_ft"]
            if payload.get("coverage") is not None:
                rec["coverage"] = payload["coverage"]
            if payload.get("project_arc") is not None:
                rec["project_arc"] = bool(payload.get("project_arc"))
            if payload.get("valid") and h8 is not None:
                conf = self._seg_conformance(
                    h8, w, h, payload.get("hoop_on_left",
                                          self._last_hoop_on_left))
                if conf:
                    rec["seg_conf"] = conf
            self._calib_log_fh.write(json.dumps(rec) + "\n")
            self._calib_log_fh.flush()
        except Exception:
            pass

    def _fail(self, reason):
        self._fail_reasons[reason] = self._fail_reasons.get(reason, 0) + 1
