"""Per-frame fitting: segment assignment, PTZ refine, and line snaps.

Takes boundary/edge evidence (source px or normalized) plus a camera prior
and produces a polished 3-DOF (pan, tilt, focal) physical camera. All the
runtime polish machinery lives here: bounded least-squares refine, optional
mask-boundary snap, and optional luma-ridge snap.

Point counts here are small (<=300 boundary points + <=120 edge points per
LM iteration) — this module stays on the CPU in the realtime port.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from pyplumber.court_segm.geometry import (
    COURT_W, COURT_H, HOOP_X, ARC_R, CORNER_LAT,
    SEG_ARC, _MODEL_LABELS, _apply_h,
)


class FittingMixin:
    """PTZ least-squares and snap helpers for CourtCalibrationNode."""

    # luma-snap tuning: ridge search half-width (model px), peak contrast
    # over the local median, minimum surviving samples, pre-fit sanity (px)
    _SNAP_SEARCH = 12
    _SNAP_CONTRAST = 18.0
    _SNAP_MIN_PTS = 12
    _SNAP_MAX_MED_PX = 11.0
    # internal target-curve labels for the luma snap
    _SNAP_ARC = 0
    _SNAP_SIDE = 1
    _SNAP_BASE = 2

    def _luma_ridge_targets(self, h8, hoop_on_left, luma):
        """Project the model arc + far sideline into the luma plane, find the
        nearest painted-line ridge along each local normal (same detector as
        the offline ridge_eval), and return (targets_norm, labels) in source-
        normalized coords. The luma plane is the model-input frame; content
        geometry comes from the mask_* letterbox params. Pure function of its
        arguments (called from BOTH the solver worker and the wrapper's
        per-frame tracker thread)."""
        if luma is None:
            return None, None, None
        lh, lw = luma.shape
        try:
            hm = np.array([[h8[0], h8[1], h8[2]],
                           [h8[3], h8[4], h8[5]],
                           [h8[6], h8[7], 1.0]])
            hi = np.linalg.inv(hm)
        except np.linalg.LinAlgError:
            return None, None, None
        theta = np.arcsin(CORNER_LAT / ARC_R)
        ts = np.linspace(-theta, theta, 60)
        hx = HOOP_X if hoop_on_left else COURT_W - HOOP_X
        bx = 0.0 if hoop_on_left else COURT_W
        sgn = 1.0 if hoop_on_left else -1.0
        arc = np.stack([hx + sgn * ARC_R * np.cos(ts),
                        25.0 + ARC_R * np.sin(ts)], axis=1)
        # near-side arc (court y > 38) is occluded by the courtside crowd
        arc = arc[arc[:, 1] <= 38.0]
        line = np.linspace(4.0, COURT_W - 4.0, 50)
        far = np.stack([line, np.zeros_like(line)], axis=1)
        bl = np.linspace(2.0, 40.0, 30)
        base = np.stack([np.full_like(bl, bx), bl], axis=1)
        sr = self._SNAP_SEARCH
        ss = np.arange(-sr, sr + 1, dtype=float)
        targets, labels, pre_px = [], [], []
        for pts_ft, lab in ((arc, self._SNAP_ARC), (far, self._SNAP_SIDE),
                            (base, self._SNAP_BASE)):
            pn = pts_ft / np.array([COURT_W, COURT_H])
            q = (hi @ np.hstack([pn, np.ones((len(pn), 1))]).T).T
            with np.errstate(divide="ignore", invalid="ignore"):
                xyn = q[:, :2] / q[:, 2:3]
            lx = xyn[:, 0] * self.mask_content_w + self.mask_pad_x
            ly = xyn[:, 1] * self.mask_content_h + self.mask_pad_y
            ok = (np.isfinite(lx) & np.isfinite(ly)
                  & (lx >= sr + 2) & (lx < lw - sr - 2)
                  & (ly >= sr + 2) & (ly < lh - sr - 2))
            # normals from curve neighbors (central difference)
            tx = np.gradient(lx)
            ty = np.gradient(ly)
            nrm = np.hypot(tx, ty)
            ok &= np.isfinite(nrm) & (nrm > 1e-3)
            if not ok.any():
                continue
            nx = np.where(ok, -ty / np.maximum(nrm, 1e-9), 0.0)
            ny = np.where(ok, tx / np.maximum(nrm, 1e-9), 0.0)
            idx = np.nonzero(ok)[0]
            sx = (lx[idx, None] + nx[idx, None] * ss).astype(int)
            sy = (ly[idx, None] + ny[idx, None] * ss).astype(int)
            prof = luma[sy, sx].astype(np.float32)
            base = np.median(prof, axis=1, keepdims=True)
            inner = prof[:, 1:-1]
            peaks = ((inner - base >= self._SNAP_CONTRAST)
                     & (inner >= prof[:, :-2]) & (inner >= prof[:, 2:]))
            # nearest qualifying ridge to the projected point, per sample
            offmag = np.where(peaks, np.abs(ss[1:-1])[None, :], np.inf)
            k = np.argmin(offmag, axis=1)
            rows = np.isfinite(offmag[np.arange(len(k)), k])
            if not rows.any():
                continue
            off = ss[1:-1][k[rows]]
            ri = idx[rows]
            rx = lx[ri] + nx[ri] * off
            ry = ly[ri] + ny[ri] * off
            txn = (rx - self.mask_pad_x) / self.mask_content_w
            tyn = (ry - self.mask_pad_y) / self.mask_content_h
            targets.append(np.stack([txn, tyn], axis=1))
            labels.append(np.full(len(txn), lab, dtype=int))
            pre_px.extend(np.abs(off).tolist())
        if not targets:
            return None, None, None
        return np.vstack(targets), np.concatenate(labels), pre_px

    def _boundary_snap(self, prm, w, h, hoop_on_left, pts_px, span_scale=0.5):
        """Track-mode fit of the locked camera to the SEG-MASK BOUNDARY —
        the boundary between the court and 3-pt segmentation classes, often
        pixel-perfect while luma ridges capture the wrong painted line. Bounded (pan, tilt, F)
        LM on boundary points segment-assigned under the lock prior; arc
        evidence required; never regresses."""
        if prm is None or pts_px is None or len(pts_px) < 30:
            return prm, 0
        prm = np.asarray(prm, dtype=float)
        h8 = self._ptz_to_h8(tuple(prm), w, h)
        if h8 is None:
            return prm, 0
        pn = np.asarray(pts_px, dtype=float) / np.array([w, h], dtype=float)
        sel, lab = self._assign_segments(h8, pn, 6.0, hoop_on_left)
        if sel is None or len(sel) < 30 or int((lab == SEG_ARC).sum()) < 15:
            return prm, 0
        if len(sel) > 150:
            idx = np.linspace(0, len(sel) - 1, 150).astype(int)
            sel, lab = sel[idx], lab[idx]
        scale = np.array([COURT_W, COURT_H])

        def resid(v):
            h8v = self._ptz_to_h8((v[0], v[1], v[2],
                                   prm[3], prm[4], prm[5]), w, h)
            if h8v is None:
                return np.full(len(sel), 10.0)
            proj = _apply_h(h8v, sel) * scale
            return self._curve_residuals_ft(proj, lab, hoop_on_left)

        span = np.array([0.02, 0.015, 0.04 * prm[2]]) * float(span_scale)
        lo = np.maximum(prm[:3] - span, self._PTZ_LO)
        hi_b = np.minimum(prm[:3] + span, self._PTZ_HI)
        pre = float(np.median(np.abs(resid(prm[:3]))))
        try:
            sol = least_squares(resid, prm[:3], loss="soft_l1", f_scale=0.5,
                                max_nfev=15, method="trf", bounds=(lo, hi_b))
        except Exception:
            return prm, 0
        post = float(np.median(np.abs(resid(sol.x))))
        if not np.isfinite(post) or post >= pre or post > 1.0:
            return prm, 0
        out = prm.copy()
        out[:3] = sol.x
        return out, len(sel)

    def _luma_snap(self, prm, w, h, hoop_on_left, luma, span_scale=1.0):
        """Final painted-line polish: bounded (pan, tilt, F) LM against luma
        ridge targets. The segmentation masks quantize at proto resolution
        (~8 px at 1080p); the painted lines in the luma plane do not — this
        is the step that breaks the mask-quantization accuracy floor.
        Returns (prm, n_targets_used); on any gate failure returns the input
        camera unchanged (never regresses)."""
        if prm is None or luma is None:
            return prm, 0
        prm = np.asarray(prm, dtype=float)
        h8 = self._ptz_to_h8(tuple(prm), w, h)
        if h8 is None:
            return prm, 0
        tn, labels, pre = self._luma_ridge_targets(h8, hoop_on_left, luma)
        if tn is None or len(tn) < self._SNAP_MIN_PTS:
            self._fail("luma_few")
            return prm, 0
        if len(pre) < 10 or float(np.median(pre)) > self._SNAP_MAX_MED_PX:
            # ridges too far from the projection: wrong lines / bad frame —
            # do not let the snap chase them
            self._fail("luma_far")
            return prm, 0
        # Each painted line pins specific camera DOF: the arc constrains all
        # three; the (near-horizontal) far sideline only tilt; the (near-
        # vertical) baseline only pan. Players usually occlude the arc, so
        # boundary-only frames still snap — with exactly the DOF their
        # evidence can support, never more.
        arc_m = labels == self._SNAP_ARC
        side_m = labels == self._SNAP_SIDE
        base_m = labels == self._SNAP_BASE
        na, ns, nb = int(arc_m.sum()), int(side_m.sum()), int(base_m.sum())
        if na >= 10:
            free = [0, 1, 2]
        elif nb >= 8 and ns >= 8:
            free = [0, 1]
        elif ns >= 12:
            free = [1]
        elif nb >= 12:
            free = [0]
        else:
            self._fail("luma_few")
            return prm, 0
        scale = np.array([COURT_W, COURT_H])
        hx = HOOP_X if hoop_on_left else COURT_W - HOOP_X
        bx = 0.0 if hoop_on_left else COURT_W

        def resid(v):
            p3 = prm[:3].copy()
            p3[free] = v
            h8v = self._ptz_to_h8((p3[0], p3[1], p3[2],
                                   prm[3], prm[4], prm[5]), w, h)
            if h8v is None:
                return np.full(len(tn), 10.0)
            proj = _apply_h(h8v, tn) * scale
            X = np.where(np.isfinite(proj[:, 0]), proj[:, 0], 1e6)
            Y = np.where(np.isfinite(proj[:, 1]), proj[:, 1], 1e6)
            r = np.empty(len(tn))
            r[arc_m] = np.hypot(X[arc_m] - hx, Y[arc_m] - 25.0) - ARC_R
            r[side_m] = Y[side_m]
            r[base_m] = X[base_m] - bx
            return np.clip(np.where(np.isfinite(r), r, 30.0), -30.0, 30.0)

        span = np.array([0.02, 0.015, 0.04 * prm[2]]) * float(span_scale)
        lo = np.maximum(prm[:3] - span, self._PTZ_LO)[free]
        hi_b = np.minimum(prm[:3] + span, self._PTZ_HI)[free]
        pre_med = float(np.median(np.abs(resid(prm[:3][free]))))
        try:
            sol = least_squares(resid, prm[:3][free], loss="soft_l1",
                                f_scale=0.5, max_nfev=20, method="trf",
                                bounds=(lo, hi_b))
        except Exception:
            return prm, 0
        post_med = float(np.median(np.abs(resid(sol.x))))
        if not np.isfinite(post_med) or post_med >= pre_med \
                or post_med > 1.0:
            self._fail("luma_worse")
            return prm, 0
        out = prm.copy()
        out[np.asarray(free)] = sol.x
        return out, len(tn)

    # Physical-camera bounds for the per-frame PTZ refine (pan, tilt, focal):
    # real broadcast ranges, same envelope as the template grid.
    _PTZ_LO = (-0.9, 0.10, 0.8)
    _PTZ_HI = (0.9, 0.80, 3.0)

    def _assign_segments(self, h8, tp, radius, hoop_on_left):
        """Boundary points (img-norm) assigned to model segments under h8.
        Returns (pts, labels); (None, None) when the projection is mostly
        non-finite; empty arrays when nothing lands within `radius` ft."""
        proj = _apply_h(h8, tp) * np.array([COURT_W, COURT_H])
        ok = np.isfinite(proj).all(axis=1)
        if ok.sum() < 30:
            return None, None
        d, idx = self._model_trees[hoop_on_left].query(
            np.where(ok[:, None], proj, 0.0))
        assigned = ok & (d < radius)
        if not assigned.any():
            return tp[:0], _MODEL_LABELS[:0]
        return tp[assigned], _MODEL_LABELS[idx[assigned]]

    def _fit_quality(self, h8, tp, hoop_on_left):
        """(median obs->model distance ft, model coverage) of normalized
        boundary points under h8; (None, None) if degenerate."""
        proj = _apply_h(h8, tp) * np.array([COURT_W, COURT_H])
        ok = np.isfinite(proj).all(axis=1)
        if ok.sum() < 10:
            return None, None
        d, _ = self._model_trees[hoop_on_left].query(proj[ok])
        md, _ = cKDTree(proj[ok]).query(self._model_pts[hoop_on_left][::4])
        return float(np.median(d)), float(np.mean(md < 2.0))

    def _refine_ptz(self, prm, tpl_pts, top_pts, w, h, hoop_on_left,
                    hoop_norm=None, anchor_repeats=4, fit_rig=False,
                    bot_pts=None, radii=(15.0, 6.0)):
        """Polish the template camera against the painted curves in PTZ space.

        Optimizes only (pan, tilt, focal) through `_ptz_to_h8`, rig position
        frozen at the template rig: 3 DOF instead of a free 8-DOF H. A
        localized segmentation error (court flooding the arc interior, or a
        local bulge past the arc) can then only nudge pan/tilt/zoom — where
        the clean majority of the boundary outvotes it — instead of bending
        the perspective terms into a collapsed or oversized arc. The detected
        rim, anchored at its analytic ground projection (see
        `_hoop_error_ptz_ft`), is the absolute scale reference the mask
        boundary cannot provide when it floods.
        Returns (params6, median_ft, coverage) or (None, None, None)."""
        norm = np.array([w, h], dtype=float)
        scale = np.array([COURT_W, COURT_H])
        tp = tpl_pts[:: max(1, len(tpl_pts) // 300)] / norm
        sp = top_pts / norm if (top_pts is not None and self.use_far_sideline) else None
        if sp is not None and len(sp) > 80:
            sp = sp[:: len(sp) // 80]
        # court/crowd-cut points: the painted near sideline must sit just
        # ABOVE them — hinge band [50.5, 58] ft bounds the near side that
        # nothing else constrains
        bp = bot_pts / norm if bot_pts is not None else None
        if bp is not None and len(bp) > 40:
            bp = bp[:: len(bp) // 40]
        rig = tuple(float(v) for v in prm[3:])
        hx_rim = HOOP_X if hoop_on_left else COURT_W - HOOP_X

        def params_of(xv):
            """Optimization vector -> full 6-param camera. fit_rig frees the
            rig depth/height (cy, cz) on top of pan/tilt/zoom — used for the
            slow per-venue rig estimate; the structural residual of a wrong
            rig cannot be removed by pan/tilt/zoom alone."""
            if fit_rig:
                return (xv[0], xv[1], xv[2], rig[0], xv[3], xv[4])
            return tuple(xv) + rig

        def hoop_exp_of(xv):
            p6 = params_of(xv)
            cz = p6[5]
            if hoop_norm is None or cz <= self.RIM_HEIGHT_FT + 1.0:
                return None
            f = cz / (cz - self.RIM_HEIGHT_FT)
            return np.array([p6[3] + (hx_rim - p6[3]) * f,
                             p6[4] + (25.0 - p6[4]) * f])

        if fit_rig:
            lo = np.array(list(self._PTZ_LO) + [80.0, 18.0])
            hi = np.array(list(self._PTZ_HI) + [260.0, 60.0])
            x = np.clip(np.array(list(prm[:3]) + [rig[1], rig[2]], dtype=float),
                        lo + 1e-6, hi - 1e-6)
        else:
            lo = np.array(self._PTZ_LO)
            hi = np.array(self._PTZ_HI)
            x = np.clip(np.asarray(prm[:3], dtype=float), lo + 1e-6, hi - 1e-6)
        hoop_exp = hoop_exp_of(x)

        # Two assignment passes: loose radius under the template init, tight
        # after the first fit improved the camera.
        has_anchor = hoop_exp is not None
        # warm-seeded callers pass radii=(6.0,) — the coarse pass exists to
        # absorb template-grid quantization, which a fresh previous-frame
        # camera does not have; skipping it halves the per-frame LM cost
        for radius in radii:
            h8c = self._ptz_to_h8(params_of(x), w, h)
            if h8c is None:
                return None, None, None
            pts, labels = self._assign_segments(h8c, tp, radius, hoop_on_left)
            if pts is None:
                self._fail(f"ptz_segments_r{int(radius)}")
                return None, None, None
            # Without arc support the rim anchor must supply absolute scale.
            # Require two evidence groups then; the far sideline counts as one.
            n_seg = len(np.unique(labels)) + (1 if sp is not None else 0)
            min_pts = 30 if hoop_exp is None else 10
            if len(pts) < min_pts \
                    or ((labels == SEG_ARC).sum() < 5
                        and (hoop_exp is None or n_seg < 2)):
                self._fail(f"ptz_segments_r{int(radius)}")
                return None, None, None
            # Evidence checks above run on the full set; the LM itself needs
            # far fewer points — residual evaluation cost is linear in them
            # and dominates the solve (profiled ~40 evals/solve). A 3-DOF
            # camera is overdetermined a hundredfold either way.
            # precision-first (user directive): 250 points per LM iteration
            if len(pts) > 250:
                idx = np.linspace(0, len(pts) - 1, 250).astype(int)
                pts, labels = pts[idx], labels[idx]

            def residuals(xv, pts=pts, labels=labels):
                h8v = self._ptz_to_h8(params_of(xv), w, h)
                if h8v is None:
                    n = len(pts) + (len(sp) if sp is not None else 0) \
                        + (len(bp) if bp is not None else 0) \
                        + (2 * anchor_repeats if has_anchor else 0)
                    return np.full(n, 50.0)
                proj = _apply_h(h8v, pts) * scale
                res = [self._curve_residuals_ft(proj, labels, hoop_on_left)]
                if sp is not None:
                    pr = _apply_h(h8v, sp) * scale
                    sy = np.where(np.isfinite(pr[:, 1]), pr[:, 1], 50.0)
                    res.append(self.far_sideline_weight * np.abs(sy))
                if bp is not None:
                    pr = _apply_h(h8v, bp) * scale
                    by = np.where(np.isfinite(pr[:, 1]), pr[:, 1], 70.0)
                    res.append(0.5 * (np.maximum(0.0, 50.5 - by)
                                      + np.maximum(0.0, by - 58.0)))
                if has_anchor:
                    he = hoop_exp_of(xv) if fit_rig else hoop_exp
                    hp = _apply_h(h8v, np.array([hoop_norm]))[0] * scale
                    hr = hp - he
                    hr = np.where(np.isfinite(hr), np.clip(hr, -30.0, 30.0), 30.0)
                    # repeat instead of scaling: under soft_l1 a scaled-up
                    # residual saturates and pulls LESS; repetition keeps each
                    # copy in the quadratic regime (same trick as the depth
                    # anchors in the earlier PTZ work). The caller sets the
                    # repeat count: high for arc-free fits where the rim is
                    # the only absolute scale, low when clean arc evidence
                    # exists. The analytic rim point carries a few ft of
                    # rig-height uncertainty and remains a gate either way.
                    res.append(np.repeat(hr, anchor_repeats))
                return np.concatenate(res)

            try:
                sol = least_squares(residuals, x, loss="soft_l1", f_scale=1.0,
                                    max_nfev=40 if fit_rig else 30,
                                    method="trf", bounds=(lo, hi))
            except Exception:
                self._fail("ptz_lm")
                return None, None, None
            x = sol.x
        h8f = self._ptz_to_h8(params_of(x), w, h)
        if h8f is None:
            return None, None, None
        median_ft, coverage = self._fit_quality(h8f, tp, hoop_on_left)
        if median_ft is None:
            return None, None, None
        return np.array(params_of(x), dtype=float), median_ft, coverage
