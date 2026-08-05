#!/usr/bin/env python3
"""Independent CPU implementation of the overlay_many_cuda arithmetic."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from overlay_common import (
    HEIGHT,
    MAX_OVERLAYS,
    MODES,
    WIDTH,
    average_2x2_valid,
    base_fixture_path,
    blend_plane,
    overlay_fixture_path,
    read_frame,
    write_frame,
)


def compose(
    mode_name: str,
    overlay_count: int,
    raw_dir: Path,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> list[np.ndarray]:
    if mode_name not in MODES:
        raise ValueError(f"unsupported mode {mode_name!r}; choose from {', '.join(MODES)}")
    if not 1 <= overlay_count <= MAX_OVERLAYS:
        raise ValueError(f"overlay count must be in 1..{MAX_OVERLAYS}")

    mode = MODES[mode_name]
    main = read_frame(
        base_fixture_path(raw_dir, mode.main_format),
        mode.main_format,
        width,
        height,
    )
    overlays = [
        read_frame(
            overlay_fixture_path(raw_dir, index, mode.overlay_format),
            mode.overlay_format,
            width,
            height,
        )
        for index in range(1, overlay_count + 1)
    ]

    output_y = main[0]
    for overlay in overlays:
        output_y = blend_plane(output_y, overlay[0], overlay[3])

    if mode_name == "444_444":
        output_u = main[1]
        output_v = main[2]
        for overlay in overlays:
            output_u = blend_plane(output_u, overlay[1], overlay[3])
            output_v = blend_plane(output_v, overlay[2], overlay[3])
        return [output_y, output_u, output_v]

    if mode_name == "420_420":
        output_u = main[1]
        output_v = main[2]
        for overlay in overlays:
            chroma_alpha = average_2x2_valid(overlay[3])
            output_u = blend_plane(output_u, overlay[1], chroma_alpha)
            output_v = blend_plane(output_v, overlay[2], chroma_alpha)
        return [output_y, output_u, output_v]

    full_u = np.repeat(np.repeat(main[1], 2, axis=0), 2, axis=1)[:height, :width]
    full_v = np.repeat(np.repeat(main[2], 2, axis=0), 2, axis=1)[:height, :width]
    for overlay in overlays:
        full_u = blend_plane(full_u, overlay[1], overlay[3])
        full_v = blend_plane(full_v, overlay[2], overlay[3])
    return [output_y, average_2x2_valid(full_u), average_2x2_valid(full_v)]


def compare_frames(actual: list[np.ndarray], expected: list[np.ndarray]) -> list[dict[str, float | int]]:
    if len(actual) != len(expected):
        raise ValueError("actual and expected plane counts differ")
    results: list[dict[str, float | int]] = []
    for actual_plane, expected_plane in zip(actual, expected):
        if actual_plane.shape != expected_plane.shape:
            raise ValueError("actual and expected plane shapes differ")
        delta = np.abs(actual_plane.astype(np.int16) - expected_plane.astype(np.int16))
        results.append(
            {
                "mismatch_count": int(np.count_nonzero(delta)),
                "max_abs_diff": int(delta.max(initial=0)),
                "mean_abs_diff": float(delta.mean()),
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--overlays", required=True, type=int)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    args = parser.parse_args()
    write_frame(
        args.output,
        compose(args.mode, args.overlays, args.raw_dir, args.width, args.height),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
