"""CuPy CUDA backend for court_segm: per-pixel mask kernels + template GEMM.

Implements the active mask cleanup, hole synthesis, template GEMM, and
fine-grid scoring kernels as literal CUDA C RawKernels (they port verbatim
into the future C++ node).

K5 (boundary extraction) intentionally stays on CPU (find_contours): the fit
needs <=300 sub-pixel ordered-enough points and the parity risk outweighs
~1 ms. K6 (mask downsample) stays numpy: the masks are CPU-resident today,
and a 5184-element gather is cheaper than the upload it would replace —
revisit both once pybind exposes the GPU-side mask planes.

CuPy + a CUDA device are REQUIRED (user decision 2026-06-11: the numpy
per-pixel path is far too slow for production and was removed).
"""

import numpy as np

import cupy as _cp

from pyplumber.court_segm.geometry import JUNCTION_X

_cp.cuda.runtime.getDeviceCount()  # fail fast at import if no CUDA device


_BLK = 256

# Literal CUDA C — the device helpers mirror geometry.py exactly and the
# whole source is meant to be lifted verbatim into the C++ node later.
_SRC = r"""
#define COURT_W 94.0f
#define COURT_H 50.0f
#define HOOP_X 5.25f
#define ARC_R 23.75f
#define CORNER_LAT 22.0f
#define JUNCTION_X @JUNCTION_X@

__device__ __forceinline__ bool apply_h8(const float* h, float xn, float yn,
                                         float* X, float* Y) {
    float den = h[6] * xn + h[7] * yn + 1.0f;
    if (fabsf(den) < 1e-9f) return false;
    float cx = (h[0] * xn + h[1] * yn + h[2]) / den;
    float cy = (h[3] * xn + h[4] * yn + h[5]) / den;
    if (!isfinite(cx) || !isfinite(cy)) return false;
    *X = cx * COURT_W;
    *Y = cy * COURT_H;
    return true;
}

// geometry._inside_region_inset; margin <= 0 -> _inside_tpl_region
__device__ __forceinline__ bool inside_region_inset(float X, float Y,
                                                    float margin) {
    if (margin < 0.0f) margin = 0.0f;
    float lat = fmaxf(0.0f, CORNER_LAT - margin);
    float rad = fmaxf(0.0f, ARC_R - margin);
    if (fabsf(Y - 25.0f) > lat) return false;
    if (X >= margin &&
        (X <= JUNCTION_X || hypotf(X - HOOP_X, Y - 25.0f) <= rad))
        return true;
    float Xm = COURT_W - X;
    if (Xm >= margin &&
        (Xm <= JUNCTION_X || hypotf(Xm - HOOP_X, Y - 25.0f) <= rad))
        return true;
    return false;
}

// 4-connectivity CCL by min-label propagation + path compression
// (shared by K2 and K3). Labels are linear pixel indices; the component
// root is its minimum index, matching ndimage.label's raster ordering.
extern "C" __global__ void ccl_init(const unsigned char* m, int* lbl, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) lbl[i] = m[i] ? i : -1;
}

extern "C" __global__ void ccl_sweep(const unsigned char* m, int* lbl,
                                     int mw, int mh, int* changed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= mw * mh || !m[i]) return;
    int c = i % mw, r = i / mw;
    int best = lbl[i];
    if (c > 0      && m[i - 1]  && lbl[i - 1]  < best) best = lbl[i - 1];
    if (c < mw - 1 && m[i + 1]  && lbl[i + 1]  < best) best = lbl[i + 1];
    if (r > 0      && m[i - mw] && lbl[i - mw] < best) best = lbl[i - mw];
    if (r < mh - 1 && m[i + mw] && lbl[i + mw] < best) best = lbl[i + mw];
    if (best < lbl[i]) {
        lbl[i] = best;
        *changed = 1;
    }
}

extern "C" __global__ void ccl_compress(const unsigned char* m, int* lbl,
                                        int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !m[i]) return;
    int l = lbl[i];
    while (lbl[l] < l) l = lbl[l];
    lbl[i] = l;
}

extern "C" __global__ void comp_area(const int* lbl, unsigned int* area,
                                     int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int l = lbl[i];
    if (l >= 0) atomicAdd(&area[l], 1u);
}

// roots of components that touch the frame border (K3 hole filling:
// inverse-mask components NOT touching the border are the holes)
extern "C" __global__ void mark_border(const int* lbl, int mw, int mh,
                                       unsigned char* flag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= mw * mh) return;
    int c = i % mw, r = i / mw;
    if (c != 0 && c != mw - 1 && r != 0 && r != mh - 1) return;
    int l = lbl[i];
    if (l >= 0) flag[l] = 1;
}

// K2: per-hole-component area + dilation-ring stats. A ring pixel is a
// non-hole pixel 4-adjacent to the component; it is counted once per
// component (neighbor labels deduped), exactly like
// binary_dilation(comp) & ~comp in the numpy original.
extern "C" __global__ void hole_stats(
        const int* lbl, const unsigned char* holes,
        const unsigned char* court, int mw, int mh,
        unsigned int* area, unsigned int* ring_tot,
        unsigned int* ring_court) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= mw * mh) return;
    if (holes[i]) {
        atomicAdd(&area[lbl[i]], 1u);
        return;
    }
    int c = i % mw, r = i / mw;
    int ls[4];
    int nl = 0;
    if (c > 0      && holes[i - 1])  ls[nl++] = lbl[i - 1];
    if (c < mw - 1 && holes[i + 1])  ls[nl++] = lbl[i + 1];
    if (r > 0      && holes[i - mw]) ls[nl++] = lbl[i - mw];
    if (r < mh - 1 && holes[i + mw]) ls[nl++] = lbl[i + mw];
    bool isc = court[i] != 0;
    for (int a = 0; a < nl; ++a) {
        bool dup = false;
        for (int b = 0; b < a; ++b)
            if (ls[b] == ls[a]) dup = true;
        if (dup) continue;
        atomicAdd(&ring_tot[ls[a]], 1u);
        if (isc) atomicAdd(&ring_court[ls[a]], 1u);
    }
}

extern "C" __global__ void hole_write(
        const int* lbl, const unsigned char* holes,
        const unsigned int* area, const unsigned int* ring_tot,
        const unsigned int* ring_court, float min_px, float val,
        float* out, int n, unsigned int* n_found) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !holes[i]) return;
    int l = lbl[i];
    unsigned int rt = ring_tot[l];
    if ((float)area[l] >= min_px && rt > 0u && 2u * ring_court[l] >= rt) {
        out[i] = val;
        atomicAdd(n_found, 1u);
    }
}

// K4b: the 27-candidate fine grid of matching._lines_h — one block per
// candidate camera, IoU of its rendered 3-pt region vs the observed mask.
extern "C" __global__ void fine_iou(const float* h8s, const float* obs,
                                    int rows, int cols,
                                    float* inter, float* tsum) {
    int cand = blockIdx.x;
    const float* h = h8s + cand * 8;
    float it = 0.0f, ts = 0.0f;
    int P = rows * cols;
    for (int i = threadIdx.x; i < P; i += blockDim.x) {
        float xn = ((i % cols) + 0.5f) / cols;
        float yn = ((i / cols) + 0.5f) / rows;
        float X, Y;
        float t = 0.0f;
        if (apply_h8(h, xn, yn, &X, &Y) &&
            X >= 0.0f && X <= COURT_W && Y >= 0.0f && Y <= COURT_H &&
            inside_region_inset(X, Y, 0.0f))
            t = 1.0f;
        it += t * obs[i];
        ts += t;
    }
    __shared__ float s_it[256];
    __shared__ float s_ts[256];
    int tid = threadIdx.x;
    s_it[tid] = it;
    s_ts[tid] = ts;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_it[tid] += s_it[tid + s];
            s_ts[tid] += s_ts[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        inter[cand] = s_it[0];
        tsum[cand] = s_ts[0];
    }
}
""".replace("@JUNCTION_X@", "%.9ff" % JUNCTION_X)

_mod = None
_kern = {}


def _k(name):
    global _mod
    if _mod is None:
        _mod = _cp.RawModule(code=_SRC)
    f = _kern.get(name)
    if f is None:
        f = _mod.get_function(name)
        _kern[name] = f
    return f


def _launch(name, n, args):
    _k(name)(((n + _BLK - 1) // _BLK,), (_BLK,), args)


def _ccl(mask_u8, mw, mh):
    """4-conn connected components; returns int32 labels (root = min linear
    index of the component, -1 for background)."""
    n = mw * mh
    lbl = _cp.empty(n, dtype=_cp.int32)
    _launch("ccl_init", n, (mask_u8, lbl, np.int32(n)))
    changed = _cp.zeros(1, dtype=_cp.int32)
    # The convergence check (`int(changed[0])`) is a device sync — at one
    # check per sweep it dominated the whole solve (~5-8 CCL calls/frame x
    # tens of syncs each). Sweeps are micro-kernels: run them in blocks of
    # 8 between checks; a wasted block costs microseconds, a sync costs
    # milliseconds in a busy pipeline. Same 64-sweep cap, same labels.
    for _ in range(8):
        changed.fill(0)
        for _ in range(8):
            _launch("ccl_sweep", n, (mask_u8, lbl, np.int32(mw),
                                     np.int32(mh), changed))
            _launch("ccl_compress", n, (mask_u8, lbl, np.int32(n)))
        if int(changed[0]) == 0:
            break
    return lbl


# --- Region mask cleanup ------------------------------------------------


def _fill_holes(largest_u8, mw, mh):
    """binary_fill_holes: components of the inverse not touching the border
    are holes. Returns filled mask (cupy bool, raveled)."""
    inv = (largest_u8 == 0).astype(_cp.uint8)
    lbl = _ccl(inv, mw, mh)
    flag = _cp.zeros(mw * mh, dtype=_cp.uint8)
    _launch("mark_border", mw * mh, (lbl, np.int32(mw), np.int32(mh), flag))
    holes = (inv != 0) & (flag[lbl] == 0)
    return (largest_u8 != 0) | holes


def clean_region_mask(m, thr):
    """GPU evidence._clean_region_mask: keep largest component, fill holes,
    preserve sub-threshold boundary values."""
    mh, mw = m.shape
    g = _cp.asarray(m, dtype=_cp.float32).ravel()
    binm = (g >= thr).astype(_cp.uint8)
    if int(binm.any()) == 0:
        return m
    lbl = _ccl(binm, mw, mh)
    area = _cp.zeros(mw * mh, dtype=_cp.uint32)
    _launch("comp_area", mw * mh, (lbl, area, np.int32(mw * mh)))
    best = int(_cp.argmax(area))
    largest = (lbl == best).astype(_cp.uint8)
    filled = _fill_holes(largest, mw, mh)
    out = _cp.where((g >= thr) & ~filled, _cp.float32(0.0), g)
    out = _cp.where(filled & (g < thr), _cp.float32(thr + 0.25), out)
    return _cp.asnumpy(out).reshape(mh, mw)


# --- Floor-hole synthesis -----------------------------------------------


def synth_holes(court, thr, min_region_area):
    """GPU core of evidence._synth_tpl_from_court (before the final
    clean_region_mask): enclosed un-fired components of the floor mask that
    are ring-surrounded by court pixels. Returns float mask or None."""
    mh, mw = court.shape
    n = mw * mh
    g = _cp.asarray(court, dtype=_cp.float32).ravel()
    binm = (g >= thr).astype(_cp.uint8)
    if float(binm.sum()) / n < 0.08:
        return None
    # holes = ~(binm | border): with the border closed, fill_holes(closed)
    # is all-ones, so the numpy chain reduces to exactly this.
    holes = (binm == 0).reshape(mh, mw)
    holes[0, :] = False
    holes[-1, :] = False
    holes[:, 0] = False
    holes[:, -1] = False
    holes = holes.ravel().astype(_cp.uint8)
    if int(holes.any()) == 0:
        return None
    lbl = _ccl(holes, mw, mh)
    area = _cp.zeros(n, dtype=_cp.uint32)
    ring_tot = _cp.zeros(n, dtype=_cp.uint32)
    ring_court = _cp.zeros(n, dtype=_cp.uint32)
    _launch("hole_stats", n, (lbl, holes, binm, np.int32(mw), np.int32(mh),
                              area, ring_tot, ring_court))
    out = _cp.zeros(n, dtype=_cp.float32)
    found = _cp.zeros(1, dtype=_cp.uint32)
    _launch("hole_write", n,
            (lbl, holes, area, ring_tot, ring_court,
             np.float32(min_region_area * n), np.float32(thr + 0.25),
             out, np.int32(n), found))
    if int(found[0]) == 0:
        return None
    return _cp.asnumpy(out).reshape(mh, mw)


# --- Template GEMM -------------------------------------------------------

_mat_cache = {}


def gemm(mat, vec):
    """mat @ vec with `mat` cached on the GPU across calls (the template
    matrices live for the node's lifetime, so caching by id is safe)."""
    key = id(mat)
    g = _mat_cache.get(key)
    if g is None or g.shape != mat.shape:
        g = _cp.asarray(mat)
        _mat_cache[key] = g
    return _cp.asnumpy(g @ _cp.asarray(vec, dtype=g.dtype))


# --- Fine grid -----------------------------------------------------------


def fine_grid_iou(h8s, obs, rows, cols):
    """Render each candidate h8 over the template grid and IoU it against
    the observed downsampled 3-pt mask. Returns (inter, tsum) arrays."""
    nc = len(h8s)
    h = _cp.asarray(np.asarray(h8s, dtype=np.float32).reshape(nc * 8))
    o = _cp.asarray(obs, dtype=_cp.float32)
    inter = _cp.empty(nc, dtype=_cp.float32)
    tsum = _cp.empty(nc, dtype=_cp.float32)
    _k("fine_iou")((nc,), (_BLK,),
                   (h, o, np.int32(rows), np.int32(cols), inter, tsum))
    return _cp.asnumpy(inter), _cp.asnumpy(tsum)
