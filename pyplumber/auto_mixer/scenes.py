"""Scene definitions for the auto mixer canvas."""

from __future__ import annotations

import random

from pyplumber.mixer import MixerGraphBuilder

from .config import (
    CANVAS_H,
    CANVAS_W,
    FACE_CROP_W,
    PIP_SCENE_SAMPLE_SEED,
    SAMPLED_MANUAL_SCENE_COUNT,
    VSTACK2_SCENE_SAMPLE_SEED,
    VSTACK3_SCENE_SAMPLE_SEED,
)


def sampled_ordered_pairs(n: int, limit: int, seed: int) -> list[tuple[int, int]]:
    pairs = [(a, b) for a in range(n) for b in range(n) if a != b]
    random.Random(seed).shuffle(pairs)
    return pairs[:limit]


def sampled_ordered_triples(n: int, limit: int, seed: int) -> list[tuple[int, int, int]]:
    triples = [
        (a, b, c)
        for a in range(n)
        for b in range(n)
        for c in range(n)
        if a != b and a != c and b != c
    ]
    random.Random(seed).shuffle(triples)
    return triples[:limit]


# ------------------------------------------------------------------
# Scene definitions for a 9:16 canvas
# ------------------------------------------------------------------


def define_auto_scenes(
    mx: MixerGraphBuilder,
    n: int,
    *,
    include_full_face_scenes: bool = True,
    include_videoconf_scenes: bool = True,
) -> None:
    """Register scenes for *n* inputs on the 1080x1920 (9:16) canvas.

    Source names follow the convention used in build_input_subgraph:
        face_{i}  -- 608x1080 portrait crop from input i (face-tracked 9:16)
        orig_{i}  -- 1920x1080 landscape frame from input i

    Scenes registered
    -----------------
    full_face_{i}
        Camera *i* face-tracked 9:16 portrait scaled to fill the full canvas.

    videoconf_{i}
        Camera *i* as a top 1:1 crop in the top 1080×1080 slot, with up
        to 5 other cameras shown as portrait thumbnails in the bottom strip.

    vstack3_{a}_{b}_{c}
        Sampled manual scenes with three landscape sources stacked vertically,
        each scaled to 1080×608 (16:9).  Three tiles occupy 1824 px; the
        remaining 96 px are split evenly as top/bottom margins.

    vstack_{a}_{b}
        Sampled manual scenes with two landscape sources stacked vertically.

    pip_{i}_{j}
        Sampled manual scenes with camera *i* face portrait fullscreen +
        camera *j* landscape thumbnail in the top-right corner.

    multiviewer
        2×2 grid of face portraits (up to 4 inputs, ≥ 3 required).
    """
    W, H = CANVAS_W, CANVAS_H

    # ------------------------------------------------------------------
    # 1. Full screen: dominant speaker 9:16 (face-tracked portrait)
    # ------------------------------------------------------------------
    if include_full_face_scenes:
        for i in range(n):
            mx.add_scene(f"full_face_{i}", {
                f"face_{i}": {
                    "graph": f"scale_cuda=w={W}:h={H}:interp_algo=lanczos",
                    "dst_x": 0,
                    "dst_y": 0,
                }
            })

    # ------------------------------------------------------------------
    # 2. Videoconference: dominant speaker 1:1 (top) + others portrait below
    #
    # Top slot (1080 × 1080 — 1:1):
    #   face_{i} (608 × 1080) is cropped from the top to 608 × 608, then scaled.
    #
    # Bottom strip (1080 × 840):
    #   Up to 5 other cameras tiled horizontally as 9:16 portrait thumbnails,
    #   centred within the strip.
    # ------------------------------------------------------------------
    CONF_TOP_H = W                          # 1080 — square (1:1) top slot
    CONF_BOT_H = H - CONF_TOP_H            # 840
    CONF_MAIN_CROP_Y = 0

    if include_videoconf_scenes:
        for i in range(n):
            others = [j for j in range(n) if j != i][:5]
            n_oth = len(others)

            cell_w = (W // n_oth) & ~1
            cell_h = (cell_w * 16 // 9) & ~1
            if cell_h > CONF_BOT_H:
                # Portrait cells taller than the strip; constrain height and width.
                cell_h = CONF_BOT_H & ~1
                cell_w = (cell_h * 9 // 16) & ~1

            x_off = (W - n_oth * cell_w) // 2          # centre cells horizontally
            y_off = CONF_TOP_H + (CONF_BOT_H - cell_h) // 2  # centre in strip

            sources: dict = {
                f"face_{i}": {
                    "graph": (
                        f"crop_cuda=w={FACE_CROP_W}:h={FACE_CROP_W}:x=0:y={CONF_MAIN_CROP_Y},"
                        f"scale_cuda=w={W}:h={CONF_TOP_H}:interp_algo=lanczos"
                    ),
                    "dst_x": 0,
                    "dst_y": 0,
                }
            }
            for k, j in enumerate(others):
                sources[f"face_{j}"] = {
                    "graph": f"scale_cuda=w={cell_w}:h={cell_h}:interp_algo=lanczos",
                    "dst_x": x_off + k * cell_w,
                    "dst_y": y_off,
                }
            mx.add_scene(f"videoconf_{i}", sources)

    # ------------------------------------------------------------------
    # 3. Vertical stack: 3 × 16:9 landscape sources
    #
    # Each tile: 1080 × 608 (16:9; 1080 × 9/16 = 607.5 → 608 < 0.1 % error).
    # Three tiles: 1824 px total; top/bottom margins of 48 px each.
    # ------------------------------------------------------------------
    if n >= 3:
        tile_w3 = W           # 1080
        tile_h3 = 608
        top3 = (H - 3 * tile_h3) // 2  # 48
        for a, b, c in sampled_ordered_triples(n, SAMPLED_MANUAL_SCENE_COUNT, VSTACK3_SCENE_SAMPLE_SEED):
            mx.add_scene(f"vstack3_{a}_{b}_{c}", {
                f"orig_{a}": {
                    "graph": f"scale_cuda=w={tile_w3}:h={tile_h3}:interp_algo=lanczos",
                    "dst_x": 0,
                    "dst_y": top3,
                },
                f"orig_{b}": {
                    "graph": f"scale_cuda=w={tile_w3}:h={tile_h3}:interp_algo=lanczos",
                    "dst_x": 0,
                    "dst_y": top3 + tile_h3,
                },
                f"orig_{c}": {
                    "graph": f"scale_cuda=w={tile_w3}:h={tile_h3}:interp_algo=lanczos",
                    "dst_x": 0,
                    "dst_y": top3 + 2 * tile_h3,
                },
            })

    # ------------------------------------------------------------------
    # PiP: face i fullscreen + orig j as thumbnail in top-right corner.
    # ------------------------------------------------------------------
    if n >= 2:
        pip_w = W // 3           # 360
        pip_h = (pip_w * 9 // 16) & ~1  # 202
        pip_x = W - pip_w - 16
        pip_y = 16
        for i, j in sampled_ordered_pairs(n, SAMPLED_MANUAL_SCENE_COUNT, PIP_SCENE_SAMPLE_SEED):
            mx.add_scene(f"pip_{i}_{j}", {
                f"face_{i}": {
                    "graph": f"scale_cuda=w={W}:h={H}:interp_algo=lanczos",
                    "dst_x": 0,
                    "dst_y": 0,
                },
                f"orig_{j}": {
                    "graph": f"scale_cuda=w={pip_w}:h={pip_h}:interp_algo=lanczos",
                    "dst_x": pip_x,
                    "dst_y": pip_y,
                },
            })

    # ------------------------------------------------------------------
    # Vertical stack: 2 × 16:9 landscape sources.
    # Each tile: 1080 × 608.  Two tiles: 1216 px; margins of 352 px each.
    # ------------------------------------------------------------------
    if n >= 2:
        tile_w = W    # 1080
        tile_h = 608
        gap = (H - 2 * tile_h) // 2  # 352
        for a, b in sampled_ordered_pairs(n, SAMPLED_MANUAL_SCENE_COUNT, VSTACK2_SCENE_SAMPLE_SEED):
            mx.add_scene(f"vstack_{a}_{b}", {
                f"orig_{a}": {
                    "graph": f"scale_cuda=w={tile_w}:h={tile_h}:interp_algo=lanczos",
                    "dst_x": 0,
                    "dst_y": gap,
                },
                f"orig_{b}": {
                    "graph": f"scale_cuda=w={tile_w}:h={tile_h}:interp_algo=lanczos",
                    "dst_x": 0,
                    "dst_y": gap + tile_h,
                },
            })

    # ------------------------------------------------------------------
    # Multiviewer: 2×2 grid of face portraits (up to 4 inputs).
    # Each cell: 540 × 960 (9:16).  Two rows of two fill 1080 × 1920 exactly.
    # ------------------------------------------------------------------
    if n >= 3:
        cols = 2
        cell_w = W // cols          # 540
        cell_h = cell_w * 16 // 9  # 960
        grid_n = min(n, 4)
        sources_mv: dict = {}
        for j in range(grid_n):
            row, col = j // cols, j % cols
            sources_mv[f"face_{j}"] = {
                "graph": f"scale_cuda=w={cell_w}:h={cell_h}:interp_algo=lanczos",
                "dst_x": col * cell_w,
                "dst_y": row * cell_h,
            }
        mx.add_scene("multiviewer", sources_mv)
