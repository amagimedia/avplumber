"""Per-frame fitting: segment assignment, sector consensus, PTZ refine.

Takes boundary/edge evidence (source px or normalized) plus a camera prior
and produces a polished 3-DOF (pan, tilt, focal) physical camera. All the
robustness machinery against segmentation spillover lives here: the
fit-independent sector-consensus vote, the degraded (arc-free, rim-anchored)
mode, and the bounded least-squares refine.

Point counts here are small (<=300 boundary points + <=120 edge points per
LM iteration) — this module stays on the CPU in the realtime port.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from pyplumber.court_segm.geometry import (
    COURT_W, COURT_H, HOOP_X, ARC_R, CORNER_LAT,
    SEG_FAR_STRAIGHT, SEG_ARC, SEG_NEAR_STRAIGHT, SEG_BASELINE,
    _MODEL_LABELS, _apply_h,
)


class FittingMixin:
    """Consensus filtering + PTZ least-squares refine for CourtCalibrationNode."""

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

    def _consensus_filter(self, h8_prior, tpl_pts, w, h, hoop_on_left):
        """Drop spatially-coherent spillover lumps from the 3-pt boundary.

        Projects ALL boundary points through the prior (template-match)
        homography, labels them by nearest model segment, and bins them
        along each segment (arc: 5 deg of hoop angle; straights/baseline:
        5 ft). A bin whose median signed offset from its analytic curve
        deviates from that segment's own consensus (median of bin medians)
        by more than sector_outlier_ft is a segmentation lump: flood edges
        and bulges are contiguous, so whole-bin voting separates them from
        the painted line far more reliably than per-point trimming, and is
        symmetric (catches inward flood edges and outward bulges alike).
        Per-segment consensus also cancels the prior's own quantization
        bias. Classification uses only the prior, never the fit being
        produced — no circular adopt-if-better guard needed.
        Returns (kept_pts, arc_drop_fraction, degraded); degraded means the
        arc was majority-contaminated and fully dropped (kept_pts then holds
        only straights/baseline; the caller must anchor scale on the rim).
        kept_pts is None when nothing trustworthy remains."""
        norm = np.array([w, h], dtype=float)
        scale = np.array([COURT_W, COURT_H])
        tp = tpl_pts / norm
        proj = _apply_h(h8_prior, tp) * scale
        ok = np.isfinite(proj).all(axis=1)
        if ok.sum() < 30:
            return None, 1.0, True
        d, idx = self._model_trees[hoop_on_left].query(
            np.where(ok[:, None], proj, 0.0))
        labels = _MODEL_LABELS[idx]
        assigned = ok & (d < 15.0)
        hx = HOOP_X if hoop_on_left else COURT_W - HOOP_X
        X = np.where(ok, proj[:, 0], 0.0)
        Y = np.where(ok, proj[:, 1], 0.0)
        dx = X - hx if hoop_on_left else hx - X  # mirrored: arc spans dx>0
        dy = Y - 25.0
        base_sign = 1.0 if hoop_on_left else -1.0
        base_x = 0.0 if hoop_on_left else COURT_W

        dropped = np.zeros(len(tp), dtype=bool)

        def vote(seg_mask, offs, coords, width):
            if seg_mask.sum() < 6:
                return
            sub = np.nonzero(seg_mask)[0]
            b = np.floor(coords[sub] / width).astype(int)
            uniq = np.unique(b)
            meds = np.array([np.median(offs[sub[b == u]]) for u in uniq])
            # Max-coverage consensus: the offset window of width 2*thr that
            # holds the most bins. A plain median-of-medians sits BETWEEN the
            # clusters at ~50/50 contamination (or under a depth-warped
            # prior) and drops both sides, true arc included; the densest
            # window finds the majority cluster itself.
            thr = self.sector_outlier_ft
            sm = np.sort(meds)
            best_n, lo_i, hi_i = 0, 0, 0
            j = 0
            for i in range(len(sm)):
                while sm[i] - sm[j] > 2.0 * thr:
                    j += 1
                if i - j + 1 > best_n:
                    best_n, lo_i, hi_i = i - j + 1, j, i
            consensus = 0.5 * (sm[lo_i] + sm[hi_i])
            bad = uniq[np.abs(meds - consensus) > thr]
            if len(bad):
                dropped[sub[np.isin(b, bad)]] = True

        arc_mask = assigned & (labels == SEG_ARC) & (dx > 0)
        vote(arc_mask, np.hypot(dx, dy) - ARC_R,
             np.degrees(np.arctan2(dy, np.maximum(dx, 1e-9))), 5.0)
        vote(assigned & (labels == SEG_FAR_STRAIGHT),
             Y - (25.0 - CORNER_LAT), X, 5.0)
        vote(assigned & (labels == SEG_NEAR_STRAIGHT),
             Y - (25.0 + CORNER_LAT), X, 5.0)
        vote(assigned & (labels == SEG_BASELINE),
             base_sign * (X - base_x), Y, 5.0)

        arc_total = int(arc_mask.sum())
        arc_drop = float((dropped & arc_mask).sum()) / max(1, arc_total)
        # Majority-contaminated arc (uniform flood): no per-bin majority vote
        # can recover the painted arc, so drop ALL arc points — the surviving
        # straights/baseline plus the rim anchor still pin a degraded 3-DOF
        # fit. This is what keeps a sane steady arc through the flood eras
        # instead of publishing a flood-shaped one.
        degraded = arc_total < 5 or arc_drop > self.sector_max_drop
        if degraded:
            dropped = dropped | arc_mask
        keep = ~dropped
        if keep.sum() < 40:
            return None, arc_drop, degraded
        return tpl_pts[keep], arc_drop, degraded

    def _refine_ptz(self, prm, tpl_pts, top_pts, w, h, hoop_on_left,
                    hoop_norm=None, anchor_repeats=4, fit_rig=False,
                    bot_pts=None):
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
        for radius in (15.0, 6.0):
            h8c = self._ptz_to_h8(params_of(x), w, h)
            if h8c is None:
                return None, None, None
            pts, labels = self._assign_segments(h8c, tp, radius, hoop_on_left)
            if pts is None:
                self._fail(f"ptz_segments_r{int(radius)}")
                return None, None, None
            # Without arc support the rim anchor must supply absolute scale;
            # with it, straights+baseline+sideline+rim still pin 3 DOF (the
            # degraded flood-era fit). Require two evidence groups then — the
            # far sideline counts as one (rescue fits run on baseline edge +
            # sideline + rim alone when the 3-pt mask has collapsed, with as
            # few as 10 boundary points).
            n_seg = len(np.unique(labels)) + (1 if sp is not None else 0)
            min_pts = 30 if hoop_exp is None else 10
            if len(pts) < min_pts \
                    or ((labels == SEG_ARC).sum() < 5
                        and (hoop_exp is None or n_seg < 2)):
                self._fail(f"ptz_segments_r{int(radius)}")
                return None, None, None

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
                    # repeat count: high for degraded (arc-free) fits where
                    # the rim is the only absolute scale, low when clean arc
                    # evidence exists — the analytic rim point carries a few
                    # ft of rig-height uncertainty and must not out-pull the
                    # painted line (it remains enforced as a GATE either way).
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
