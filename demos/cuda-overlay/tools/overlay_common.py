#!/usr/bin/env python3
"""Shared format and raw-frame helpers for the CUDA overlay demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


WIDTH = 641
HEIGHT = 360
MAX_OVERLAYS = 15


@dataclass(frozen=True)
class Mode:
    name: str
    main_format: str
    overlay_format: str


MODES = {
    "420_420": Mode("420_420", "yuv420p", "yuva420p"),
    "420_444": Mode("420_444", "yuv420p", "yuva444p"),
    "444_444": Mode("444_444", "yuv444p", "yuva444p"),
}


def plane_shapes(pixel_format: str, width: int, height: int) -> tuple[tuple[int, int], ...]:
    """Return planar (height, width) shapes in rawvideo byte order."""
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")

    chroma_420 = ((height + 1) // 2, (width + 1) // 2)
    full = (height, width)
    if pixel_format == "yuv420p":
        return full, chroma_420, chroma_420
    if pixel_format == "yuva420p":
        return full, chroma_420, chroma_420, full
    if pixel_format == "yuv444p":
        return full, full, full
    if pixel_format == "yuva444p":
        return full, full, full, full
    raise ValueError(f"unsupported planar format: {pixel_format}")


def frame_size(pixel_format: str, width: int, height: int) -> int:
    return sum(rows * columns for rows, columns in plane_shapes(pixel_format, width, height))


def read_frame(path: Path, pixel_format: str, width: int, height: int) -> list[np.ndarray]:
    data = path.read_bytes()
    expected = frame_size(pixel_format, width, height)
    if len(data) != expected:
        raise ValueError(f"{path} has {len(data)} bytes; expected exactly {expected}")

    planes: list[np.ndarray] = []
    offset = 0
    for rows, columns in plane_shapes(pixel_format, width, height):
        size = rows * columns
        plane = np.frombuffer(data, dtype=np.uint8, count=size, offset=offset)
        planes.append(plane.reshape(rows, columns).copy())
        offset += size
    return planes


def write_frame(path: Path, planes: Iterable[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        for plane in planes:
            array = np.asarray(plane)
            if array.dtype != np.uint8 or array.ndim != 2:
                raise ValueError("raw planes must be two-dimensional uint8 arrays")
            output.write(array.tobytes(order="C"))


def base_fixture_path(raw_dir: Path, pixel_format: str) -> Path:
    return raw_dir / f"base_{pixel_format}.yuv"


def overlay_fixture_path(raw_dir: Path, index: int, pixel_format: str) -> Path:
    if not 1 <= index <= MAX_OVERLAYS:
        raise ValueError(f"overlay index must be in 1..{MAX_OVERLAYS}")
    return raw_dir / f"overlay_{index:02d}_{pixel_format}.yuv"


def blend_plane(background: np.ndarray, foreground: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Blend one uint8 plane using the CUDA kernel's exact rounded /255."""
    if background.shape != foreground.shape or background.shape != alpha.shape:
        raise ValueError("background, foreground, and alpha shapes must match")
    bg = background.astype(np.uint32)
    fg = foreground.astype(np.uint32)
    a = alpha.astype(np.uint32)
    total = a * fg + (255 - a) * bg
    rounded = total + 128
    return ((rounded + (rounded >> 8)) >> 8).astype(np.uint8)


def average_2x2_valid(values: np.ndarray) -> np.ndarray:
    """Rounded 2x2 average using only samples inside an odd-sized plane."""
    if values.ndim != 2:
        raise ValueError("values must be a two-dimensional plane")
    height, width = values.shape
    chroma_height = (height + 1) // 2
    chroma_width = (width + 1) // 2
    total = values[0::2, 0::2].astype(np.uint32)
    count = np.ones((chroma_height, chroma_width), dtype=np.uint32)

    right = values[0::2, 1::2].astype(np.uint32)
    if right.size:
        total[:, : right.shape[1]] += right
        count[:, : right.shape[1]] += 1

    bottom = values[1::2, 0::2].astype(np.uint32)
    if bottom.size:
        total[: bottom.shape[0], :] += bottom
        count[: bottom.shape[0], :] += 1

    diagonal = values[1::2, 1::2].astype(np.uint32)
    if diagonal.size:
        total[: diagonal.shape[0], : diagonal.shape[1]] += diagonal
        count[: diagonal.shape[0], : diagonal.shape[1]] += 1

    return ((total + count // 2) // count).astype(np.uint8)
