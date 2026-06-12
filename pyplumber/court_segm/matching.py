"""Template-matching camera search.

Exhaustive search over physically valid broadcast cameras: each grid point
(pan, tilt, zoom) is pre-rendered as court + 3-pt masks at _TMPL_RES; per
frame the best IoU against the observed segmentation wins. No descent, no
local minima, no reference extraction.

The (templates x pixels) GEMM in _lines_h and the 27-render
fine grid are CUDA/CuPy port candidates (cuBLAS measured 0.42 ms
for the GEMM); see doc/court_calibration_realtime_port.md.
"""

import numpy as np

from pyplumber.court_segm.geometry import COURT_W, COURT_H, _apply_h
from pyplumber.court_segm import cuda as _gpu


class MatchingMixin:
    """Pre-rendered PTZ template grid + IoU matching for CourtCalibrationNode."""

    _TMPL_RES = (54, 96)  # rows, cols

    def _ensure_templates(self, w, h):
        if getattr(self, "_tmpl_tpl", None) is not None:
            return
        rows, cols = self._TMPL_RES
        gy, gx = np.meshgrid((np.arange(rows) + 0.5) / rows,
                             (np.arange(cols) + 0.5) / cols, indexing="ij")
        pix = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (P,2) img-norm
        pans = np.linspace(-0.85, 0.85, 35)
        tilts = np.linspace(0.15, 0.65, 11)
        zooms = np.geomspace(0.9, 2.8, 12)
        params, t_tpl, t_court = [], [], []
        rim_l, rim_r = [], []
        for th in pans:
            for ph in tilts:
                for F in zooms:
                    prm = (th, ph, F, 47.0, 110.0, 35.0)
                    h8 = self._ptz_to_h8(prm, w, h)
                    if h8 is None:
                        continue
                    c = _apply_h(h8, pix) * np.array([COURT_W, COURT_H])
                    X = np.where(np.isfinite(c[:, 0]), c[:, 0], 1e6)
                    Y = np.where(np.isfinite(c[:, 1]), c[:, 1], 1e6)
                    in_court = (X >= 0) & (X <= COURT_W) & (Y >= 0) & (Y <= COURT_H)
                    if in_court.mean() < 0.10:
                        continue  # camera barely sees the court
                    rl = self._rim_px_norm(h8, True, prm)
                    rr = self._rim_px_norm(h8, False, prm)
                    if rl is None or rr is None:
                        continue
                    params.append(prm)
                    t_court.append(in_court)
                    t_tpl.append(self._inside_tpl_region(X, Y) & in_court)
                    rim_l.append(rl)
                    rim_r.append(rr)
        self._tmpl_params = np.array(params)
        self._tmpl_tpl = np.array(t_tpl, dtype=np.float32)
        self._tmpl_court = np.array(t_court, dtype=np.float32)
        # row sums are reused every frame; computing them per call costs a
        # full pass over the 3191x5184 matrices
        self._tmpl_tpl_sum = self._tmpl_tpl.sum(axis=1)
        self._tmpl_court_sum = self._tmpl_court.sum(axis=1)
        self._tmpl_rim = {True: np.array(rim_l), False: np.array(rim_r)}
        self._tmpl_w, self._tmpl_h = w, h
        print(f"court_calibration: built {len(params)} camera templates", flush=True)

    def _downsample_mask(self, m):
        rows, cols = self._TMPL_RES
        mh, mw = m.shape
        ci, ri = self._norm_to_mask_idx((np.arange(cols) + 0.5) / cols,
                                        (np.arange(rows) + 0.5) / rows, mw, mh)
        xi = np.clip(np.round(ci).astype(int), 0, mw - 1)
        yi = np.clip(np.round(ri).astype(int), 0, mh - 1)
        return (m[np.ix_(yi, xi)] >= self.mask_threshold).astype(np.float32).ravel()

    def _lines_h(self, tpl, court, hoop, w, h, hoop_on_left):
        """Template-matching calibration; returns source-norm->court-norm h8.

        Best-of-both evidence: the 3-pt region term (yellow and/or floor-hole
        union) and the floor term each contribute when their observation is
        usable — NEITHER is required alone. A floor-only match additionally
        needs the rim pixel to pin pan/side (the floor trapezoid is
        near-symmetric)."""
        self._ensure_templates(w, h)
        obs_tpl = self._downsample_mask(tpl) if tpl is not None else None
        obs_court = self._downsample_mask(court) if court is not None else None
        has_tpl = obs_tpl is not None and obs_tpl.sum() >= 12
        has_court = obs_court is not None and obs_court.sum() >= 50
        if not has_tpl and not has_court:
            self._match_iou = 0.0
            return None
        score = np.zeros(len(self._tmpl_params))
        iou_tpl = None
        coverage = None
        if has_tpl:
            inter = _gpu.gemm(self._tmpl_tpl, obs_tpl)
            union = self._tmpl_tpl_sum + obs_tpl.sum() - inter
            iou_tpl = inter / np.maximum(union, 1.0)
            score += 2.0 * iou_tpl
        if has_court:
            cov_inter = _gpu.gemm(self._tmpl_court, obs_court)
            coverage = cov_inter / obs_court.sum()
            # penalize templates claiming court where none is observed above
            # the crowd line (false court in the stands)
            excess = (self._tmpl_court_sum - cov_inter) \
                / np.maximum(self._tmpl_court_sum, 1.0)
            score = score + coverage - 0.3 * excess
        d_rim = None
        if hoop is not None:
            rim = self._tmpl_rim[hoop_on_left]
            d_rim = np.hypot(rim[:, 0] - hoop[0] / w, rim[:, 1] - hoop[1] / h)
            score = score - 2.0 * np.minimum(d_rim, 0.3)
            if not has_tpl:
                score = np.where(d_rim > 0.10, -1e9, score)
        elif not has_tpl:
            self._match_iou = 0.0
            return None  # floor-only without a rim cannot pin pan/side
        best = int(np.argmax(score))
        if has_tpl:
            if iou_tpl[best] < self.template_min_iou:
                self._match_iou = 0.0
                return None
            self._match_iou = float(iou_tpl[best])
        else:
            if coverage[best] < 0.85 or d_rim[best] > 0.10:
                self._match_iou = 0.0
                return None
            self._match_iou = float(coverage[best])
        prm = self._tmpl_params[best].copy()
        if has_tpl:
            # fine local refinement: render a small on-the-fly grid around the
            # winner to remove coarse-grid quantization error
            rows, cols = self._TMPL_RES
            cands, h8s = [], []
            for dth in (-0.025, 0.0, 0.025):
                for dph in (-0.023, 0.0, 0.023):
                    for dF in (0.93, 1.0, 1.075):
                        cand = (prm[0] + dth, prm[1] + dph, prm[2] * dF,
                                prm[3], prm[4], prm[5])
                        h8c = self._ptz_to_h8(cand, w, h)
                        if h8c is None:
                            continue
                        cands.append(cand)
                        h8s.append(h8c)
            ious = []
            if cands:
                inter, tsum = _gpu.fine_grid_iou(h8s, obs_tpl, rows, cols)
                osum = float(obs_tpl.sum())
                ious = inter / np.maximum(tsum + osum - inter, 1.0)
            best_f, best_prm = -1.0, prm
            for cand, iou in zip(cands, ious):
                if iou > best_f:
                    best_f, best_prm = float(iou), np.array(cand)
            prm = best_prm
            self._match_iou = float(best_f)
        prev = getattr(self, "_match_prm", None)
        if prev is not None and abs(prev[0] - prm[0]) < 0.35:
            a = 0.5
            prm[:3] = prev[:3] * (1 - a) + prm[:3] * a
        self._match_prm = prm
        return self._ptz_to_h8(tuple(prm), w, h)
