"""GPU color contracts shared by source, canvas and rendition builders.

Input metadata is resolved on every decoded frame by tonemap_cuda. A declared
contract overrides frame tags explicitly; pixel depth never selects a transfer.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

TRANSFER_TAGS = {"sdr": "bt709", "hlg": "arib-std-b67", "pq": "smpte2084"}
OPERATORS = ("none", "linear", "gamma", "clip", "reinhard", "hable", "mobius")
COLOR_KEYS = ("color_trc", "color_primaries", "colorspace", "color_range")
TEN_BIT_FORMATS = ("p010le", "p210le", "yuv420p10le", "yuv422p10le", "yuv444p10le")
SEMIPLANAR_FORMATS = ("nv12", "nv16", "p010le", "p210le")   # what tonemap_cuda reads and writes
# HDR10 static metadata for PQ outputs: the mixer masters on BT.2020 primaries with D65 white.
MASTERING_PRIMARIES = {"bt2020": {"primaries": [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
                                  "white_point": [0.3127, 0.3290]}}


def hdr_metadata(peak_nits, *, max_cll=0, max_fall=0, min_nits=0.0001):
    """Mastering display + content light level for a PQ output, in nits. MaxCLL defaults
    to the peak and MaxFALL to 40% of MaxCLL, the usual 1000/400 pairing."""
    cll = max_cll or round(peak_nits)
    return {**MASTERING_PRIMARIES["bt2020"], "max_luminance": round(peak_nits), "min_luminance": min_nits,
            "max_cll": cll, "max_fall": max_fall or round(cll * 0.4)}
YUV_FORMATS = ("nv12", "nv16", "yuv420p", "yuv422p", "yuv444p", *TEN_BIT_FORMATS)


@dataclass(frozen=True)
class Color:
    transfer: str = "sdr"

    def __post_init__(self):
        if self.transfer not in TRANSFER_TAGS:
            raise ValueError(f"unsupported color transfer {self.transfer!r}; use sdr, hlg or pq")

    @property
    def tags(self):
        hdr = self.transfer != "sdr"
        return dict(zip(COLOR_KEYS, (TRANSFER_TAGS[self.transfer],
                                    "bt2020" if hdr else "bt709",
                                    "bt2020nc" if hdr else "bt709", "tv")))

    @property
    def setparams(self):
        return "setparams=" + ":".join(f"{'range' if k == 'color_range' else k}={v}"
                                      for k, v in self.tags.items())

    def validate_format(self, pixel_format):
        if pixel_format not in SEMIPLANAR_FORMATS:
            raise ValueError(f"unsupported canvas/output pixel format {pixel_format!r}")
        if self.transfer != "sdr" and pixel_format not in TEN_BIT_FORMATS:
            raise ValueError("HLG/PQ canvas and outputs require a 10-bit pixel format")

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(value)
        if not isinstance(value, Mapping):
            raise ValueError("color must be sdr, hlg, pq or a complete color metadata object")
        missing = [k for k in COLOR_KEYS if not value.get(k)]
        if missing:
            raise ValueError(f"explicit source color setting is incomplete: missing {', '.join(missing)}")
        tags = dict(value)
        tags["color_range"] = {"limited": "tv", "mpeg": "tv"}.get(tags["color_range"], tags["color_range"])
        tags["colorspace"] = {"bt2020_ncl": "bt2020nc"}.get(tags["colorspace"], tags["colorspace"])
        for transfer in TRANSFER_TAGS:
            color = cls(transfer)
            if all(tags[k] == v for k, v in color.tags.items()):
                return color
        raise ValueError("unsupported or contradictory color metadata; supported contracts are limited-range "
                         "BT.709 SDR and BT.2020 non-constant-luminance HLG/PQ")


def declared_color(obj):
    """None means require frame metadata; an override must be complete."""
    fields = {k: obj[k] for k in COLOR_KEYS if k in obj}
    if "color" in obj:
        color = Color.parse(obj["color"])
        if fields and any(color.tags[k] != v for k, v in fields.items()):
            raise ValueError("color preset contradicts individual color metadata fields")
        return color
    return Color.parse(fields) if fields else None


def conversion_graph(target, pixel_format, *, source=None, source_format=None,
                     tonemap="clip", sdr_white=203.0, hdr_peak=1000.0, desat=0.0, param=0.0):
    """Validate and normalize CUDA YUV frames, keeping identity frames zero-copy.

    NVDEC emits NV12/P010. Callers with other CUDA storage must declare it so
    the required chroma conversion precedes tone mapping. Custom source filters
    run before this graph: their output frame metadata is authoritative.
    With pixel_format=None, tonemap_cuda preserves input chroma subsampling and
    selects depth from the target transfer. Renditions need an explicit format.
    """
    target = Color.parse(target)
    if pixel_format is not None:
        target.validate_format(pixel_format)
    if tonemap not in OPERATORS:
        raise ValueError(f"unsupported tone-map operator {tonemap!r}")
    if not all(isfinite(v) for v in (sdr_white, hdr_peak, desat, param)) or not (1 <= sdr_white <= hdr_peak <= 10000 and hdr_peak >= 100 and desat >= 0 and param >= 0):
        raise ValueError("require finite 1 <= sdr_white <= hdr_peak <= 10000, hdr_peak >= 100, desat >= 0 and param >= 0")
    if source_format and source_format not in YUV_FORMATS:
        raise ValueError(f"unsupported source pixel format {source_format!r}; color conversion requires CUDA YUV")
    if pixel_format is not None and source is not None and Color.parse(source) == target:
        # Same contract: stamp it and only change storage, so 4:2:2 (P210) content
        # never round-trips through the 4:2:0-only tone mapper.
        parts = [target.setparams]
        return ",".join(parts if source_format == pixel_format else parts + [f"scale_cuda=format={pixel_format}"])
    parts = [Color.parse(source).setparams] if source is not None else []
    if source_format and source_format not in SEMIPLANAR_FORMATS:
        # tonemap_cuda works on semiplanar storage; planar sources are re-laid out at 10 bits.
        parts.append("scale_cuda=format=p210le" if "422" in source_format else "scale_cuda=format=p010le")
    # tonemap_cuda converts color and storage in one pass, 4:2:0 or 4:2:2 in and out.
    # param is the operator knee in reference-white units (mobius/reinhard; 0 keeps the
    # filter default 0.3). mobius at 0.9 keeps 0..90% of SDR white linear and folds
    # everything brighter into the top 10% of the SDR range; 1.0 would be a plain clip.
    storage = f":format={pixel_format}" if pixel_format is not None else ""
    parts.append(f"tonemap_cuda=transfer_in=auto:transfer_out={target.transfer}{storage}"
                 f":tonemap={tonemap}:sdr_white={sdr_white:g}:hdr_peak={hdr_peak:g}:desat={desat:g}"
                 + (f":param={param:g}" if param else ""))
    return ",".join(parts)


def rendition_color(canvas, codec, requested=None, tonemap=""):
    canvas = Color.parse(canvas)
    if codec not in ("h264_nvenc", "hevc_nvenc"):
        raise ValueError(f"unsupported mixer output codec {codec!r}")
    color = Color.parse(requested) if requested is not None else (Color() if tonemap or codec == "h264_nvenc" else canvas)
    if codec == "h264_nvenc" and color.transfer != "sdr":
        raise ValueError("H.264 renditions require SDR; use HEVC Main10 for HLG/PQ")
    if tonemap and color.transfer != "sdr":
        raise ValueError("rendition.tonemap requests SDR and contradicts the output color")
    return color
