"""Node-parameter helpers shared by the graph stages (no engine import)."""

from __future__ import annotations

from typing import Any

#: Shield for nodes that need the frame format at creation (cuda_transform-family
#: nodes do not declare one; see ``assume_video_format``).
CUDA_NV12 = {"pixel_format": "cuda", "real_pixel_format": "nv12"}

_BICUBIC_ABOVE = 1.3
_MULTISAMPLE_FROM = 2.0
_SAMPLES = 4


def auto_filter() -> dict[str, Any]:
    """Per-frame filter choice for a source of unknown size (bicubic up, multisample down)."""
    return {
        "filter": "auto",
        "bicubic_above": _BICUBIC_ABOVE,
        "multisample_above": _MULTISAMPLE_FROM,
        "samples": _SAMPLES,
        "bicubic_param": 0.5,
    }


BILINEAR = {"filter": "bilinear"}


def cuda_scale(
    n: Any,
    *,
    name: str,
    src: str,
    dst: str,
    width: int,
    height: int,
    group: str,
    auto_restart: str,
    fit: str = "stretch",
    filter_: dict[str, Any] | None = None,
    crop: dict[str, int] | None = None,
    on_error: str | None = None,
    hwaccel: str = "@gpu",
) -> Any:
    """A ``cuda_transform`` with one NV12 output holding one layer: the whole input
    (or ``crop``) scaled into ``width`` x ``height``.

    ``fit="stretch"`` fills the output, ``"contain"`` letterboxes. Sizes must be
    even (NV12 chroma is 2x2).
    """
    if fit not in ("stretch", "contain"):
        raise ValueError(f"{name}: fit must be 'stretch' or 'contain', got {fit!r}")
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError(f"{name}: need a positive even size, got {width}x{height}")
    layer: dict[str, Any] = {
        "dst_x": 0,
        "dst_y": 0,
        "dst_w": width,
        "dst_h": height,
        "fit": fit,
        **(filter_ or BILINEAR),
    }
    if crop is not None:
        layer["crop"] = crop
    params: dict[str, Any] = {
        "src": src,
        "group": group,
        "name": name,
        "auto_restart": auto_restart,
        "hwaccel": hwaccel,
        "outputs": [
            {
                "dst": dst,
                "width": width,
                "height": height,
                "sw_format": "nv12",
                "layers": [layer],
            }
        ],
    }
    if on_error is not None:
        params["on_error"] = on_error
    return n.CudaTransform(params)


def center_crop(src_w: int, src_h: int, dst_w: int, dst_h: int) -> dict[str, int]:
    """Largest centred rectangle of ``src`` with the aspect of ``dst`` (even values)."""
    if src_w * dst_h >= dst_w * src_h:  # source is wider: trim the sides
        h = src_h
        w = int(src_h * dst_w / dst_h) // 2 * 2
    else:  # source is taller: trim top and bottom
        w = src_w
        h = int(src_w * dst_h / dst_w) // 2 * 2
    return {
        "x": (src_w - w) // 2 // 2 * 2,
        "y": (src_h - h) // 2 // 2 * 2,
        "w": w,
        "h": h,
    }
