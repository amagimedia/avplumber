"""Normalized placement geometry for the native CUDA compositor. No rendering."""

from __future__ import annotations

from typing import Any

DEFAULT_LAYERS = 3


def hidden_layer(canvas_width: int) -> dict[str, Any]:
    """An inactive metadata pad with its own crop dictionary."""
    return {
        "crop": {"x": 0, "y": 0, "w": 2, "h": 2},
        "dst_x": -canvas_width,
        "dst_y": 0,
        "dst_w": 2,
        "dst_h": 2,
        "fit": "stretch",
    }


def _even(value: float) -> int:
    return int(value) & ~1


def source_crop(src_norm: list[float], src_w: int, src_h: int) -> dict[str, int]:
    x1 = max(0, _even(src_norm[0] * src_w))
    y1 = max(0, _even(src_norm[1] * src_h))
    w = max(2, _even((src_norm[2] - src_norm[0]) * src_w))
    h = max(2, _even((src_norm[3] - src_norm[1]) * src_h))
    if x1 + w > src_w:
        x1 = max(0, _even(src_w - w))
        w = max(2, _even(src_w - x1))
    if y1 + h > src_h:
        y1 = max(0, _even(src_h - h))
        h = max(2, _even(src_h - y1))
    return {"x": x1, "y": y1, "w": w, "h": h}


def layers_from_plan(
    crop_plan: list[dict[str, list[float]]],
    *,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
    max_layers: int = DEFAULT_LAYERS,
) -> tuple[dict[str, dict[str, Any]], list[list[int]]]:
    """Map recipe (src,tgt) placements onto cuda_rect_overlay per-input metadata."""
    layers = {str(i): hidden_layer(dst_w) for i in range(max_layers)}
    dsts: list[list[int]] = []
    for i, entry in enumerate(crop_plan[:max_layers]):
        src = entry.get("src") or []
        tgt = entry.get("tgt") or []
        if len(src) < 4 or len(tgt) < 4:
            dsts.append([-dst_w, 0, 2, 2])
            continue
        x1 = max(0, _even(tgt[0] * dst_w))
        y1 = max(0, _even(tgt[1] * dst_h))
        w = max(2, _even((tgt[2] - tgt[0]) * dst_w))
        h = max(2, _even((tgt[3] - tgt[1]) * dst_h))
        if x1 + w > dst_w:
            w = max(2, _even(dst_w - x1))
        if y1 + h > dst_h:
            h = max(2, _even(dst_h - y1))
        layers[str(i)] = {
            "crop": source_crop(src, src_w, src_h),
            "dst_x": x1,
            "dst_y": y1,
            "dst_w": w,
            "dst_h": h,
            "fit": "stretch",
        }
        dsts.append([x1, y1, w, h])
    while len(dsts) < max_layers:
        dsts.append([-dst_w, 0, 2, 2])
    return layers, dsts
