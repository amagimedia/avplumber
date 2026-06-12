"""Microbenchmark for court_segm/cuda.py active kernels.

Builds realistic synthetic 272x272 masks through the real PTZ camera model
and times each ported op. (The numpy fallback was removed 2026-06-11 —
CUDA is required — so this is timing-only; the parity comparison lives in
git history.)

Run inside the avp-builder container on the T4:
    PYTHONPATH=<repo> python -m pyplumber.court_segm.cuda_bench
"""

import time

import numpy as np

from pyplumber.court_segm import cuda as _gpu
from pyplumber.court_segm.geometry import GeometryMixin, COURT_W, COURT_H, _apply_h
from pyplumber.court_segm.evidence import EvidenceMixin
from pyplumber.court_segm.matching import MatchingMixin

MASK = 272
W, H = 1920, 1080


class _Node(GeometryMixin, EvidenceMixin, MatchingMixin):
    mask_threshold = 0.5
    min_region_area = 0.04
    clean_tpl_mask = True
    template_min_iou = 0.30
    border_margin_px = 2.0
    mask_model_w = 960.0
    mask_model_h = 544.0
    mask_pad_x = 0.0
    mask_pad_y = 2.0
    mask_content_w = 960.0
    mask_content_h = 540.0

    def _fail(self, reason):
        pass


def _render_masks(node, prm):
    """Project every mask pixel through the camera: court + 3-pt masks."""
    h8 = node._ptz_to_h8(prm, W, H)
    cols, rows = np.arange(MASK), np.arange(MASK)
    xn, _ = node._mask_idx_to_norm(cols, np.zeros(MASK), MASK, MASK)
    _, yn = node._mask_idx_to_norm(np.zeros(MASK), rows, MASK, MASK)
    gy, gx = np.meshgrid(yn, xn, indexing="ij")
    pix = np.stack([gx.ravel(), gy.ravel()], axis=1)
    c = _apply_h(h8, pix) * np.array([COURT_W, COURT_H])
    X = np.where(np.isfinite(c[:, 0]), c[:, 0], 1e6).reshape(MASK, MASK)
    Y = np.where(np.isfinite(c[:, 1]), c[:, 1], 1e6).reshape(MASK, MASK)
    in_court = (X >= 0) & (X <= COURT_W) & (Y >= 0) & (Y <= COURT_H)
    in_tpl = node._inside_tpl_region(X, Y) & in_court
    court = (in_court & ~in_tpl).astype(np.float32) * 0.9
    tpl = in_tpl.astype(np.float32) * 0.9
    return court, tpl, h8


def _timeit(fn, reps=30):
    fn()  # warmup / JIT / template upload
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    return (time.perf_counter() - t0) / reps * 1e3, out


def _compare(name, fn, exact=False, tol=0.005):
    """Time fn on the GPU (name kept for the call sites below)."""
    t_gpu, out_gpu = _timeit(fn)
    shape = None if out_gpu is None else np.asarray(out_gpu).shape
    print(f"{name:28s} gpu {t_gpu:7.3f} ms   out {shape}", flush=True)


def main():
    node = _Node()
    rng = np.random.default_rng(7)
    prm = (0.5, 0.35, 1.3, 47.0, 110.0, 35.0)
    court, tpl, h8 = _render_masks(node, prm)

    # noisy 3-pt mask: specks across the floor + a hole bitten into it
    noisy = tpl.copy()
    sp = rng.random(tpl.shape) > 0.999
    noisy[sp] = 0.8
    cy, cx = np.argwhere(tpl > 0.5)[len(np.argwhere(tpl > 0.5)) // 2]
    noisy[cy - 6:cy + 6, cx - 6:cx + 6] = 0.0
    _compare("K3 clean_region_mask",
             lambda: node._clean_region_mask(noisy), exact=True)

    # floor with an enclosed un-fired hole for the synthesis: full floor
    # (court + 3-pt) with a hole cut where the floor has interior support
    floor = np.maximum(court, tpl)
    ys, xs = np.nonzero(floor > 0.5)
    hy, hx = int(np.median(ys)), int(np.median(xs))
    floor[hy - 32:hy + 32, hx - 32:hx + 32] = 0.0
    _compare("K2 synth_tpl_from_court",
             lambda: node._synth_tpl_from_court(floor.copy()))

    node._ensure_templates(W, H)
    obs_tpl = node._downsample_mask(tpl)
    obs_court = node._downsample_mask(np.maximum(court, tpl))
    _compare("K4a gemm tpl",
             lambda: _gpu.gemm(node._tmpl_tpl, obs_tpl))
    _compare("K4a gemm court",
             lambda: _gpu.gemm(node._tmpl_court, obs_court))

    def lines():
        node._match_prm = None
        h = node._lines_h(tpl, np.maximum(court, tpl), None, W, H, True)
        return np.concatenate([np.asarray(h), node._match_prm[:3]]) \
            if h is not None else None
    _compare("K4 _lines_h end-to-end", lines, tol=0.05)
    print("fitted prm:", None if node._match_prm is None
          else np.round(node._match_prm[:3], 4), " true:", prm[:3],
          flush=True)


if __name__ == "__main__":
    main()
