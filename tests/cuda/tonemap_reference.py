"""NumPy reference (oracle) for tonemap_cuda, ported verbatim from FFmpeg n8.1
libavfilter/opencl/tonemap.cl + colorspace_common.cl.

HDR (HLG or PQ, BT.2020) -> SDR (BT.709). Used to validate the CUDA kernel
offline: the CUDA output must match this within rounding. Static peak (no
per-frame detection); operators direct/linear/gamma/clip/reinhard/hable/mobius,
matching the n8.1 kernel exactly.
"""

import numpy as np

REFERENCE_WHITE = 100.0
ST2084_MAX_LUMINANCE = 10000.0
ST2084_M1 = 0.1593017578125
ST2084_M2 = 78.84375
ST2084_C1 = 0.8359375
ST2084_C2 = 18.8515625
ST2084_C3 = 18.6875
HLG_A, HLG_B, HLG_C = 0.17883277, 0.28466892, 0.55991073
SDR_AVG = 0.25

# BT.2020 and BT.709 luma weights.
LUMA_2020 = np.array([0.2627, 0.6780, 0.0593])
LUMA_709 = np.array([0.2126, 0.7152, 0.0722])

# Linear-light BT.2020 -> BT.709 primaries (rgb2rgb), standard NPM product.
RGB2020_TO_709 = np.array([
    [1.660491, -0.587641, -0.072850],
    [-0.124550, 1.132900, -0.008349],
    [-0.018151, -0.100579, 1.118730],
])


def eotf_st2084(x):
    p = np.power(np.maximum(x, 0.0), 1.0 / ST2084_M2)
    a = np.maximum(p - ST2084_C1, 0.0)
    b = np.maximum(ST2084_C2 - ST2084_C3 * p, 1e-6)
    c = np.power(a / b, 1.0 / ST2084_M1)
    return np.where(x > 0.0, c * ST2084_MAX_LUMINANCE / REFERENCE_WHITE, 0.0)


def inverse_oetf_hlg(x):
    a = 4.0 * x * x
    b = np.exp((x - HLG_C) / HLG_A) + HLG_B
    return np.where(x < 0.5, a, b)


def oetf_bt709(c):
    c = np.maximum(c, 0.0)
    return np.where(c < 0.018, 4.5 * c, 1.099 * np.power(c, 0.45) - 0.099)


def ootf_hlg(rgb, peak):
    luma = rgb @ LUMA_2020
    gamma = max(1.0, 1.2 + 0.42 * np.log10(peak * REFERENCE_WHITE / 1000.0))
    factor = peak * np.power(np.maximum(luma, 1e-6), gamma - 1.0) / np.power(12.0, gamma)
    return rgb * factor[..., None]


def hable_f(x):
    a, b, c, d, e, f = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
    return (x * (x * a + b * c) + d * e) / (x * (x * a + b) + d * f) - e / f


TONE = {
    "direct": lambda s, peak, p: s,
    "linear": lambda s, peak, p: s * p / peak,
    "clip": lambda s, peak, p: np.clip(s * p, 0.0, 1.0),
    "reinhard": lambda s, peak, p: s / (s + p) * (peak + p) / peak,
    "hable": lambda s, peak, p: hable_f(s) / hable_f(peak),
    "mobius": lambda s, peak, p: _mobius(s, peak, p),
    "gamma": lambda s, peak, p: np.where(s > 0.05, (s / peak) ** (1 / p),
                                          s * (0.05 / peak) ** (1 / p) / 0.05),
}


def _mobius(s, peak, j):
    a = -j * j * (peak - 1.0) / (j * j - 2.0 * j + peak)
    b = (j * j - 2.0 * j * peak + peak) / max(peak - 1.0, 1e-6)
    out = (b * b + 2.0 * b * j + j * j) / (b - a) * (s + a) / (s + b)
    return np.where(s <= j, s, out)


DEFAULT_PARAM = {"linear": 1.0, "gamma": 1.8, "clip": 1.0, "reinhard": 0.3,
                 "hable": 1.0, "mobius": 0.3, "direct": 1.0}


def map_one_pixel_rgb(rgb, peak, average, tonemap, param, desat, target_peak=1.0):
    sig = np.maximum(np.maximum(rgb[..., 0], np.maximum(rgb[..., 1], rgb[..., 2])), 1e-6)
    if target_peak > 1.0:
        sig = sig / target_peak
        peak = peak / target_peak
    sig_old = sig.copy()
    slope = min(1.0, SDR_AVG / average)
    sig = sig * slope
    peak = peak * slope
    if desat > 0.0:
        luma = rgb @ LUMA_709
        coeff = np.maximum(sig - 0.18, 1e-6) / np.maximum(sig, 1e-6)
        coeff = np.power(coeff, 10.0 / desat)
        rgb = rgb * (1 - coeff[..., None]) + luma[..., None] * coeff[..., None]
        sig = sig * (1 - coeff) + (luma * slope) * coeff
    sig = TONE[tonemap](sig, peak, param)
    sig = np.minimum(sig, 1.0)
    return rgb * (sig / sig_old)[..., None]


def tonemap_hdr_to_sdr(rgb_src, transfer, peak, tonemap="hable", param=None, desat=0.5):
    """rgb_src: source non-linear RGB in [0,1] (BT.2020). Returns BT.709 SDR
    non-linear RGB in [0,1]. peak in REFERENCE_WHITE units (HLG 1000 nits -> 10)."""
    if param is None:
        param = DEFAULT_PARAM[tonemap]
    if transfer == "pq":
        lin = eotf_st2084(rgb_src)
    elif transfer == "hlg":
        lin = inverse_oetf_hlg(rgb_src)
        lin = ootf_hlg(lin, peak)
    else:
        raise ValueError(transfer)
    lin = lin @ RGB2020_TO_709.T                    # lrgb2lrgb: BT.2020 -> BT.709
    lin = map_one_pixel_rgb(lin, peak, SDR_AVG, tonemap, param, desat)
    return oetf_bt709(np.clip(lin, 0.0, None))


# --- YCbCr <-> non-linear RGB, matching colorspace_common.cl (limited range) ---

def yuv2rgb_2020(y10, cb10, cr10):
    """BT.2020 limited-range 10-bit YCbCr codes -> non-linear RGB in [0,1]."""
    y = (y10 / 1023.0 * 255.0 - 16.0) / 219.0
    u = (cb10 / 1023.0 * 255.0 - 128.0) / 224.0
    v = (cr10 / 1023.0 * 255.0 - 128.0) / 224.0
    r = y + 1.4746 * v
    g = y - 0.16455 * u - 0.57135 * v
    b = y + 1.8814 * u
    return np.stack([r, g, b], axis=-1)


def rgb2yuv_709_8bit(rgb):
    """BT.709 non-linear RGB [0,1] -> limited-range 8-bit YCbCr codes."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    u = (b - y) / 1.8556
    v = (r - y) / 1.5748
    y8 = np.rint(219.0 * y + 16.0)
    u8 = np.rint(224.0 * u + 128.0)
    v8 = np.rint(224.0 * v + 128.0)
    return (np.clip(y8, 0, 255).astype(int), np.clip(u8, 0, 255).astype(int),
            np.clip(v8, 0, 255).astype(int))


def tonemap_codes(y10, cb10, cr10, transfer, peak, tonemap="hable", param=None, desat=0.5):
    """Full pipeline oracle: BT.2020 HLG/PQ 10-bit codes -> BT.709 SDR 8-bit codes."""
    rgb_src = yuv2rgb_2020(np.asarray(y10, float), np.asarray(cb10, float), np.asarray(cr10, float))
    sdr = tonemap_hdr_to_sdr(rgb_src, transfer, peak, tonemap, param, desat)
    return rgb2yuv_709_8bit(sdr)


# Explicit-transfer reference uses display light in nits (BT.1886/BT.2100).
# Derive the gamut transform from published chromaticities, independently of
# the rounded CUDA coefficient tables.
def _rgb_to_xyz(primaries):
    p = np.array(primaries)
    basis = np.stack((p[:, 0] / p[:, 1], np.ones(3),
                      (1 - p.sum(axis=1)) / p[:, 1]))
    white = np.array([0.3127 / 0.3290, 1.0, (1 - 0.3127 - 0.3290) / 0.3290])
    return basis * np.linalg.solve(basis, white)


_XYZ709 = _rgb_to_xyz([(0.640, 0.330), (0.300, 0.600), (0.150, 0.060)])
_XYZ2020 = _rgb_to_xyz([(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)])
_TO2020 = np.linalg.solve(_XYZ2020, _XYZ709)


def display_light(rgb, transfer, sdr_white=203.0, hdr_peak=1000.0):
    """Encoded RGB -> display-linear nits, ideal black, no surround adjustment."""
    rgb = np.maximum(rgb, 0.0)
    if transfer == "sdr":
        return rgb ** 2.4 * sdr_white
    if transfer == "pq":
        return eotf_st2084(rgb) * REFERENCE_WHITE
    scene = np.where(rgb <= 0.5, rgb * rgb / 3,
                     (np.exp((rgb - HLG_C) / HLG_A) + HLG_B) / 12)
    gamma = max(1.0, 1.2 + 0.42 * np.log10(hdr_peak / 1000))
    gain = hdr_peak * np.maximum(scene @ LUMA_2020, 1e-12) ** (gamma - 1)
    return scene * gain[..., None]


def encode_display_light(rgb, transfer, sdr_white=203.0, hdr_peak=1000.0):
    """Display-linear nits -> encoded RGB, inverse of display_light."""
    rgb = np.maximum(rgb, 0.0)
    if transfer == "sdr":
        return (rgb / sdr_white) ** (1 / 2.4)
    if transfer == "pq":
        p = (rgb / 10000) ** ST2084_M1
        return ((ST2084_C1 + ST2084_C2 * p) / (1 + ST2084_C3 * p)) ** ST2084_M2
    gamma = max(1.0, 1.2 + 0.42 * np.log10(hdr_peak / 1000))
    luma = rgb @ LUMA_2020
    scene = rgb * ((luma / hdr_peak) ** (1 / gamma) / np.maximum(luma, 1e-12))[..., None]
    return np.where(scene <= 1 / 12, np.sqrt(3 * scene),
                    HLG_A * np.log(np.maximum(12 * scene - HLG_B, 1e-12)) + HLG_C)


def rgb_to_codes(rgb, transfer, depth):
    weights = LUMA_709 if transfer == "sdr" else LUMA_2020
    y = rgb @ weights
    u = (rgb[..., 2] - y) / (2 * (1 - weights[2]))
    v = (rgb[..., 0] - y) / (2 * (1 - weights[0]))
    scale = 1 << (depth - 8)
    return np.stack(((219 * y + 16) * scale, (224 * u + 128) * scale,
                     (224 * v + 128) * scale), axis=-1)


def codes_to_rgb(codes, transfer, depth):
    weights = LUMA_709 if transfer == "sdr" else LUMA_2020
    codes = codes / (1 << (depth - 8))
    y, u, v = (codes[..., 0] - 16) / 219, (codes[..., 1] - 128) / 224, (codes[..., 2] - 128) / 224
    r = y + 2 * (1 - weights[0]) * v
    b = y + 2 * (1 - weights[2]) * u
    g = (y - weights[0] * r - weights[2] * b) / weights[1]
    return np.stack((r, g, b), axis=-1)


def convert_transfer_codes(codes, source, target, depth, *, sdr_white=203.0,
                           hdr_peak=1000.0, tonemap="direct", desat=0.0):
    if source == target:
        return codes.copy()
    light = display_light(codes_to_rgb(codes, source, depth), source, sdr_white, hdr_peak)
    if source == "sdr":
        light = light @ _TO2020.T
    elif target == "sdr":
        light = light @ np.linalg.inv(_TO2020).T
        light = map_one_pixel_rgb(light / sdr_white, hdr_peak / sdr_white, SDR_AVG,
                                  tonemap, DEFAULT_PARAM[tonemap], desat) * sdr_white
    rgb = np.clip(encode_display_light(light, target, sdr_white, hdr_peak), 0, 1)
    return rgb_to_codes(rgb, target, 8 if target == "sdr" else 10)


if __name__ == "__main__":
    # Reference points to lock the CUDA kernel against (peak=10 -> HLG 1000 nits).
    print("transfer y10 cb10 cr10 -> Y8 U8 V8 (hable, peak=10, desat=0.5)")
    for tr in ("hlg", "pq"):
        for (y, cb, cr) in [(64, 512, 512), (500, 512, 512), (940, 512, 512),
                            (700, 400, 700), (300, 700, 400), (843, 512, 512)]:
            Y, U, V = tonemap_codes(np.array([y]), np.array([cb]), np.array([cr]), tr, 10.0)
            print(f"{tr} {y} {cb} {cr} -> {int(Y[0])} {int(U[0])} {int(V[0])}")
