#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from overlay_common import (  # noqa: E402
    HEIGHT,
    WIDTH,
    average_2x2_valid,
    blend_plane,
    frame_size,
    plane_shapes,
)


class OverlayCommonTest(unittest.TestCase):
    def test_odd_420_layout_uses_ceiling_chroma_dimensions(self) -> None:
        self.assertEqual(
            plane_shapes("yuv420p", WIDTH, HEIGHT),
            ((360, 641), (180, 321), (180, 321)),
        )
        self.assertEqual(frame_size("yuva420p", WIDTH, HEIGHT), 577_080)

    def test_blend_endpoints_and_rounding(self) -> None:
        backgrounds = np.arange(256, dtype=np.uint8).reshape(16, 16)
        foregrounds = np.flip(backgrounds, axis=1).copy()
        for alpha in (0, 1, 63, 127, 128, 191, 254, 255):
            alphas = np.full_like(backgrounds, alpha)
            actual = blend_plane(backgrounds, foregrounds, alphas)
            total = (
                alpha * foregrounds.astype(np.uint32)
                + (255 - alpha) * backgrounds.astype(np.uint32)
            )
            expected = ((total + 127) // 255).astype(np.uint8)
            np.testing.assert_array_equal(actual, expected)

    def test_average_2x2_uses_only_valid_edge_samples(self) -> None:
        values = np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            dtype=np.uint8,
        )
        expected = np.array([[3, 5], [8, 9]], dtype=np.uint8)
        np.testing.assert_array_equal(average_2x2_valid(values), expected)

    def test_sequential_layers_round_after_each_blend(self) -> None:
        background = np.array([[5]], dtype=np.uint8)
        first = np.array([[128]], dtype=np.uint8)
        second = np.array([[250]], dtype=np.uint8)
        alpha = np.array([[128]], dtype=np.uint8)
        result = blend_plane(blend_plane(background, first, alpha), second, alpha)
        self.assertEqual(int(result[0, 0]), 159)


if __name__ == "__main__":
    unittest.main()
