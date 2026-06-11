"""Temporal filtering, zone routing, and the publish/hold/rescue paths.

The ByteTrack-style PTZ filter (innovation gating, frozen holds, capped
convergence), the chain-of-trust adoption logic, the rim-decisive zone
model (arcs only at court ends), the mid-court pan mode, and metadata
output. This is the newest, behavior-defining layer — it stays in Python
in the realtime port.
"""

import json

import numpy as np
from scipy.optimize import least_squares

from pyplumber.court_segm.geometry import COURT_W, COURT_H, _apply_h


class TemporalMixin:
    """State machine + publish paths for CourtCalibrationNode."""

    # innovation gates per frame: pan, tilt (rad), F (relative)
    _INNOV_GATE = (0.010, 0.008, 0.04)

    def _ptz_filter(self, meas):
        """ByteTrack-style temporal filter on (pan, tilt, F): constant-
        velocity prediction + innovation gating + alpha-beta update. A real
        broadcast camera moves smoothly — a single-frame geometry jump is a
        misdetection, so the prediction is published instead (flicker
        suppressed); a large innovation must persist with consistent sign
        for several frames before being accepted as a real cut/whip."""
        meas = np.asarray(meas, dtype=float)
        st = self._ptz_state
        # keep continuity across short invalid gaps too (a broadcast camera
        # cannot teleport in 2s); only a long outage or a side-switch-scale
        # pan justifies a hard re-init
        if st is None or self._age > 50 \
                or abs(meas[0] - st[0]) >= 0.35:
            self._ptz_state = meas.copy()
            self._ptz_vel = np.zeros(3)
            self._innov_streak = 0
            return meas
        v = getattr(self, "_ptz_vel", np.zeros(3))
        pred3 = st[:3] + v
        innov = meas[:3] - pred3
        gates = np.array([self._INNOV_GATE[0], self._INNOV_GATE[1],
                          self._INNOV_GATE[2] * max(0.8, st[2])])
        out = np.abs(innov) > gates
        if out.any():
            sgn = np.sign(innov)
            prev_sgn = getattr(self, "_innov_sgn", np.zeros(3))
            if self._innov_streak > 0 and np.all((sgn * prev_sgn)[out] >= 0):
                self._innov_streak += 1
            else:
                self._innov_streak = 1
            self._innov_sgn = sgn
            if self._innov_streak < 4:
                # outlier: publish the smooth prediction, ignore the jump
                self._ptz_vel = v * 0.9
                self._ptz_state = np.concatenate([pred3, meas[3:]])
                return self._ptz_state.copy()
            # persisted consistently: it is real — converge toward the
            # measurement at a HARD-CAPPED rate (≈ one gate per frame, i.e.
            # never faster than a plausible pan); the streak stays alive so
            # we keep stepping every frame until back inside the gate. No
            # catch-up velocity: motion is never synthesized.
            step = np.clip(innov, -gates, gates)
            self._ptz_state = np.concatenate([pred3 + step, meas[3:]])
            self._ptz_vel = v * 0.5
            return self._ptz_state.copy()
        self._innov_streak = 0
        new3 = pred3 + 0.35 * innov
        self._ptz_vel = v * 0.85 + 0.12 * innov
        self._ptz_state = np.concatenate([new3, meas[3:]])
        return self._ptz_state.copy()

    def _mid_pan_relative(self, frame, floor, w, h):
        """Mid-court pan mode: no rim and no 3-pt evidence in view. Pan is
        unobservable from the masks (aperture problem along the sideline)
        but tilt/zoom stay anchored by the floor outline — refit those with
        pan frozen at its chained value, UNLESS confident court-pose
        keypoints are visible (the pose model's mid-court points make pan
        observable again; pose is used ONLY here, never at the well-served
        court ends). Publishes zone="mid": sidelines only, plus the center
        line when pose anchored it. Returns True if published."""
        if self._ptz_state is None or floor is None:
            return False
        top = self._court_top_edge(floor, w, h)
        if top is None:
            return False
        bot = self._court_bottom_edge(floor, w, h)
        norm = np.array([w, h], dtype=float)
        scale = np.array([COURT_W, COURT_H])
        sp = top / norm
        if len(sp) > 60:
            sp = sp[:: len(sp) // 60]
        bp = bot / norm if bot is not None else None
        if bp is not None and len(bp) > 40:
            bp = bp[:: len(bp) // 40]
        pose_img = pose_court = None
        if hasattr(self, "_mid_pose_pts"):
            pose_img, pose_court = self._mid_pose_pts
        have_pose = pose_img is not None and len(pose_img) >= 1
        two_mid = getattr(self, "_two_region_mid", None)
        free_pan = have_pose or two_mid is not None
        x0 = np.asarray(self._ptz_state, dtype=float).copy()

        def residuals(v):
            if free_pan:
                p6 = (v[0], v[1], v[2], x0[3], x0[4], x0[5])
            else:
                p6 = (x0[0], v[0], v[1], x0[3], x0[4], x0[5])
            h8v = self._ptz_to_h8(p6, w, h)
            n = len(sp) + (len(bp) if bp is not None else 0) \
                + (8 * len(pose_img) if have_pose else 0) \
                + (8 if two_mid is not None else 0)
            if h8v is None:
                return np.full(n, 50.0)
            pr = _apply_h(h8v, sp) * scale
            sy = np.where(np.isfinite(pr[:, 1]), pr[:, 1], 50.0)
            res = [np.abs(sy)]
            if bp is not None:
                pb = _apply_h(h8v, bp) * scale
                by = np.where(np.isfinite(pb[:, 1]), pb[:, 1], 70.0)
                res.append(0.5 * (np.maximum(0.0, 50.5 - by)
                                  + np.maximum(0.0, by - 58.0)))
            if have_pose:
                pp = _apply_h(h8v, pose_img) * scale
                pd = np.hypot(pp[:, 0] - pose_court[:, 0] * COURT_W,
                              pp[:, 1] - pose_court[:, 1] * COURT_H)
                pd = np.where(np.isfinite(pd), np.minimum(pd, 30.0), 30.0)
                res.append(np.repeat(pd, 8))
            if two_mid is not None:
                # midpoint between the two visible 3-pt regions maps to the
                # court center x=47 — the strongest mid-court pan anchor
                pm = _apply_h(h8v, np.array([two_mid]))[0] * scale
                dx = pm[0] - 47.0 if np.isfinite(pm[0]) else 30.0
                res.append(np.repeat(np.clip(dx, -30.0, 30.0), 8))
            return np.concatenate(res)

        try:
            if free_pan:
                sol = least_squares(
                    residuals, np.array([x0[0], x0[1], x0[2]]),
                    loss="soft_l1", f_scale=1.0, max_nfev=25, method="trf",
                    bounds=([-0.9, 0.10, 0.8], [0.9, 0.80, 3.0]))
                prm = np.array([sol.x[0], sol.x[1], sol.x[2],
                                x0[3], x0[4], x0[5]])
            else:
                sol = least_squares(
                    residuals, np.array([x0[1], x0[2]]),
                    loss="soft_l1", f_scale=1.0, max_nfev=25, method="trf",
                    bounds=([0.10, 0.8], [0.80, 3.0]))
                prm = np.array([x0[0], sol.x[0], sol.x[1],
                                x0[3], x0[4], x0[5]])
        except Exception:
            return False
        h8c = self._ptz_to_h8(tuple(prm), w, h)
        if h8c is None or not self._quad_sane(h8c, w, h):
            return False
        se = self._sideline_error_ft(h8c, sp)
        if se is not None and se > self.sideline_gate_ft:
            return False
        sm = self._ptz_filter(prm)
        h8s = self._ptz_to_h8(tuple(sm), w, h)
        if h8s is None:
            return False
        self._last_h = h8s
        self._age = 0
        self._fail("mid_pan_published")
        self._publish(frame, True, h8s, w, h, zone="mid", center=free_pan)
        self._dst.enqueue(frame)
        return True

    def _adopt_relative(self, frame, prm, h8, hoop_on_left, w, h,
                        err=None, coverage=None, npts=0, strong=False,
                        anchored=False):
        """Param-space EMA + state update + publish for the relative path.
        Lerping H elements mixes the perspective terms nonlinearly (visible
        wobble); pan/tilt/zoom blend cleanly. Resets across large pans (side
        switch) and stale state.

        anchored = rim-verified (absolute evidence, survives cuts). Anything
        else — INCLUDING a low-residual curve refine — is relative evidence:
        a misclassified close-up's garbage mask can produce a smooth blob the
        LM fits beautifully. Non-anchored cameras that JUMP versus a recent
        confident one (pan or zoom leap a real broadcast camera cannot do)
        are rejected; returns False. strong = curve-refined; chains the
        confident state forward when adopted."""
        if prm is not None:
            prm = np.asarray(prm, dtype=float)
            if not anchored:
                # chain of trust: the rim (absolute evidence) starts a chain;
                # curve/template fits (relative evidence) may only extend a
                # live one. Without this, a post-cut garbage mask can seed
                # its own self-consistent state.
                if self._conf_prm is None \
                        or self._frame - self._conf_frame > 150:
                    self._fail("relative_unbootstrapped")
                    return False
                zr = max(prm[2], self._conf_prm[2]) \
                    / max(1e-6, min(prm[2], self._conf_prm[2]))
                if abs(prm[0] - self._conf_prm[0]) > 0.25 or zr > 1.35:
                    self._fail("relative_jump_gate")
                    return False
            # physical scale check on the RAW measurement (before it can
            # touch filter state): every publish must size people correctly.
            # Catches wrong-scale fits (misclassified close-ups) on ANY
            # evidence path — blob residuals, IoU, even a stray rim detection
            # can be fooled; a frame full of 2x-too-large humans cannot.
            sc = self._players_scale_ratio(frame, prm, w, h)
            if sc is not None and not (0.55 <= sc <= 1.8):
                self._fail("relative_scale_gate")
                return False
            sm = self._ptz_filter(prm)
            h8s = self._ptz_to_h8(tuple(sm), w, h)
            if h8s is not None:
                h8 = h8s
            if strong or anchored:
                self._conf_prm = np.asarray(self._ptz_state
                                            if self._ptz_state is not None
                                            else prm, dtype=float)
                self._conf_frame = self._frame
        self._last_h = h8
        self._last_hoop_on_left = hoop_on_left
        self._age = 0
        self._n_ok += 1
        self._contra_streak = 0
        # Arc visibility: rim within 0.5s, or within 1s when THIS publish is
        # a fresh curve-fitted result — while panning OUT of an end zone the
        # exiting region clips away and only a live strong fit may keep the
        # arcs for the extra half second; holds/weak publishes may not.
        gap = self._frame - getattr(self, "_hoop_seen_frame", -10 ** 9)
        zone = "full" if (gap <= 12 or (gap <= 25 and err is not None)) \
            else "mid"
        self._publish(frame, True, h8, w, h, err, coverage, npts, zone=zone)
        self._dst.enqueue(frame)
        return True

    def _rescue_relative(self, frame, court, hoop_norm, hoop_on_left, w, h):
        """Relative-mode rescue when the 3-pt mask has collapsed (the big
        region_small bucket): the court mask still provides the baseline edge
        and far sideline, and the rim provides absolute scale — enough to pin
        the 3-DOF camera. Fully gated (rim + sideline + quad), so coverage is
        gained only on evidence-verified fits. Returns True if published."""
        if court is None or hoop_norm is None:
            return False
        prm = self._ptz_state if self._ptz_state is not None else \
            (np.asarray(self._match_prm, dtype=float)
             if self._match_prm is not None else None)
        if prm is None:
            # no warm state (e.g. clip start with the 3-pt class absent):
            # cold-start from the court mask + rim pixel alone
            prm = self._court_only_match(court, hoop_norm, w, h, hoop_on_left)
        if prm is None:
            return False
        prm = np.concatenate([np.asarray(prm, dtype=float)[:3], self._rig])
        base_n = self._baseline_edge_points(court, w, h, hoop_on_left)
        top_pts = self._court_top_edge(court, w, h)
        if base_n is None or top_pts is None:
            return False
        base_px = base_n * np.array([w, h], dtype=float)
        prm_ref, err_r, _cov = self._refine_ptz(
            prm, base_px, top_pts, w, h, hoop_on_left, hoop_norm,
            anchor_repeats=12, bot_pts=self._court_bottom_edge(court, w, h))
        if prm_ref is None or err_r is None or err_r > self.gate_median_ft:
            return False
        h_ref = self._ptz_to_h8(tuple(prm_ref), w, h)
        if h_ref is None:
            return False
        hoop_err = self._hoop_error_ptz_ft(
            h_ref, hoop_norm, hoop_on_left, tuple(prm_ref[3:]))
        side_err = self._sideline_error_ft(
            h_ref, top_pts / np.array([w, h], dtype=float))
        if hoop_err is None or hoop_err > self.hoop_gate_ft \
                or (side_err is not None and side_err > self.sideline_gate_ft) \
                or not self._quad_sane(h_ref, w, h):
            return False
        self._fail("rescue_no_tpl")
        return self._adopt_relative(frame, prm_ref, h_ref, hoop_on_left, w, h,
                                    err_r, None, len(base_px), strong=True,
                                    anchored=True)

    def _coast_verified(self, court, w, h):
        """Evidence-checked coasting past hold_frames: keep publishing the
        held camera only while it still explains the CURRENT court mask —
        observed court projects inside the court rectangle (coverage) and the
        observed top edge still lands on the far sideline. A blind hold
        preserves wrong fits; this check makes coasting falsifiable each
        frame."""
        if court is None or self._ptz_state is None:
            return False
        h8 = self._ptz_to_h8(tuple(self._ptz_state), w, h)
        if h8 is None:
            return False
        rows, cols = self._TMPL_RES
        gy, gx = np.meshgrid((np.arange(rows) + 0.5) / rows,
                             (np.arange(cols) + 0.5) / cols, indexing="ij")
        pix = np.stack([gx.ravel(), gy.ravel()], axis=1)
        c = _apply_h(h8, pix) * np.array([COURT_W, COURT_H])
        X = np.where(np.isfinite(c[:, 0]), c[:, 0], 1e6)
        Y = np.where(np.isfinite(c[:, 1]), c[:, 1], 1e6)
        t = ((X >= 0) & (X <= COURT_W)
             & (Y >= 0) & (Y <= COURT_H)).astype(np.float32)
        obs = self._downsample_mask(court)
        if obs.sum() < 50:
            return False
        coverage = float(t @ obs) / float(obs.sum())
        if coverage < 0.85:
            return False
        # coverage alone cannot catch a wrongly zoomed-IN camera (everything
        # maps deep inside the rectangle); the sideline can
        top_pts = self._court_top_edge(court, w, h)
        if top_pts is not None:
            se = self._sideline_error_ft(
                h8, top_pts / np.array([w, h], dtype=float))
            if se is not None and se > 2.0 * self.sideline_gate_ft:
                return False
        return True

    def _hold_or_invalid(self, frame, w, h, court=None):
        self._age += 1
        if self._last_h is not None and self._age <= self.hold_frames:
            # FROZEN hold: never synthesize motion the evidence didn't show —
            # velocity coasting can sweep the overlay toward a bad fit
            # (user-confirmed "lines fly away"); a brief freeze is honest.
            # Held frames may NEVER extend arcs beyond 0.5s of rim loss: a
            # hold during a pan-out would drag end-zone arcs into mid-court.
            gap = self._frame - getattr(self, "_hoop_seen_frame", -10 ** 9)
            self._publish(frame, True, self._last_h, w, h,
                          zone="full" if gap <= 12 else "mid")
        elif self._last_h is not None and self._coast_verified(court, w, h):
            # past the blind-hold budget, but the held camera still explains
            # the current court mask — verified coast, re-checked every frame
            self._fail("coast_verified")
            gap = self._frame - getattr(self, "_hoop_seen_frame", -10 ** 9)
            self._publish(frame, True, self._last_h, w, h,
                          zone="full" if gap <= 12 else "mid")
        else:
            self._publish(frame, False, None, w, h)
        self._dst.enqueue(frame)

    # --- output --------------------------------------------------------------

    def _emit_court_projection(self, frame, h8, w, h, hoop_on_left,
                               mid=False, center=False):
        """Reproject the court model into the image and publish it as
        pose-style metadata so draw_keypoints can overlay it on the video —
        calibration error is then directly visible against the painted lines.

        mid=True (mid-court pan): draw ONLY the pan-invariant sidelines —
        pan is unobservable there and the 3-pt lines exist only at the
        court ends; anything else drawn mid-court is fiction. The center
        line is added only when pose keypoints anchored it this frame."""
        try:
            hm = np.array([[h8[0], h8[1], h8[2]],
                           [h8[3], h8[4], h8[5]],
                           [h8[6], h8[7], 1.0]])
            hi = np.linalg.inv(hm)
        except np.linalg.LinAlgError:
            return
        curves = [] if mid else [self._model_pts[hoop_on_left][::3]]
        line = np.linspace(0.0, COURT_W, 60)
        curves.append(np.stack([line, np.zeros_like(line)], axis=1))          # far sideline
        curves.append(np.stack([line, np.full_like(line, COURT_H)], axis=1))  # near sideline
        if not mid or center:
            cl = np.linspace(0.0, COURT_H, 30)
            curves.append(np.stack([np.full_like(cl, 47.0), cl], axis=1))     # center line
        pts = np.vstack(curves) / np.array([COURT_W, COURT_H])
        q = (hi @ np.hstack([pts, np.ones((len(pts), 1))]).T).T
        with np.errstate(divide="ignore", invalid="ignore"):
            xy = q[:, :2] / q[:, 2:3] * np.array([w, h])
        ok = np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 0] < w) \
            & (xy[:, 1] >= 0) & (xy[:, 1] < h)
        xy = xy[ok]
        if not len(xy):
            return
        kpts = []
        for x, y in xy:
            kpts += [round(float(x), 1), round(float(y), 1), 1.0]
        md = {"model_width": w, "model_height": h,
              "num_keypoints": len(xy), "poses": [{"conf": 1.0, "keypoints": kpts}]}
        frame.metadata["court_proj"] = json.dumps(md)

    def _publish(self, frame, valid, h8, w, h, err=None, coverage=None,
                 npts=0, zone="full", center=False):
        out = {
            "valid": bool(valid),
            "source_w": w,
            "source_h": h,
            "hoop_on_left": bool(self._last_hoop_on_left),
            "age": min(self._age, 10 ** 6),
            "zone": zone,
        }
        if valid:
            self._n_published_valid += 1
            if h8 is not None:
                self._emit_court_projection(frame, h8, w, h,
                                            self._last_hoop_on_left,
                                            mid=(zone == "mid"), center=center)
        if h8 is not None:
            out["h"] = [float(v) for v in h8] + [1.0]
        if err is not None:
            out["err_ft"] = round(err, 3)
        if coverage is not None:
            out["coverage"] = round(coverage, 3)
        out["points"] = npts
        frame.metadata[self.metadata_key_out] = json.dumps(out)
        if self.calib_log:
            try:
                if self._calib_log_fh is None:
                    self._calib_log_fh = open(self.calib_log, "w")
                rec = {"frame": self._frame, "valid": bool(valid),
                       "left": bool(self._last_hoop_on_left),
                       "age": int(min(self._age, 10 ** 6)), "zone": zone,
                       "rig": [round(float(v), 1) for v in self._rig]}
                if h8 is not None:
                    rec["h"] = [float(v) for v in h8]
                self._calib_log_fh.write(json.dumps(rec) + "\n")
                self._calib_log_fh.flush()
            except Exception:
                pass

    def _fail(self, reason):
        self._fail_reasons[reason] = self._fail_reasons.get(reason, 0) + 1
