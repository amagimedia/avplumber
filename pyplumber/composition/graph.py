"""Source-paced layer composition with an optional GPU backdrop.

No planner, detector, tracking state or mixer application dependency. Inputs are
SDR NV12 CUDA frames; the caller owns decode, decision metadata and encode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from .effects import white_wash_lut
from .geometry import DEFAULT_LAYERS, hidden_layer
from .nodes import MAX_INPUTS, source_compositor, source_mask_param


@dataclass(frozen=True)
class SourceCompositionConfig:
    dest_width: int
    dest_height: int
    source_width: int = 1920
    source_height: int = 1080
    fps: str = "25/1"
    background: str = "black"
    group: str = "out"
    auto_restart: str = "off"
    max_layers: int = DEFAULT_LAYERS
    hwaccel: str = "@gpu"
    name: str = "Compose"
    edge_prefix: str = "v_compose"
    dst: str = "v_canvas"
    metadata_key: str = "composition_layers"
    move_metadata_key: str = "composition_move"
    blur_fraction: float = 99 / 1080
    blur_step: int = 4
    wash: float = 0.3

    def __post_init__(self) -> None:
        for name in ("dest_width", "dest_height", "source_width", "source_height"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0 or value % 2:
                raise ValueError(f"{name} must be a positive even integer for NV12")
        if Fraction(self.fps) <= 0:
            raise ValueError("fps must be positive")
        if self.background not in ("black", "blur"):
            raise ValueError("background must be 'black' or 'blur'")
        if type(
            self.max_layers
        ) is not int or not 1 <= self.max_layers <= MAX_INPUTS - int(
            self.background == "blur"
        ):
            raise ValueError(
                f"max_layers and backdrop must fit in {MAX_INPUTS} input pads"
            )
        if not math.isfinite(self.blur_fraction) or self.blur_fraction <= 0:
            raise ValueError("blur_fraction must be finite and positive")
        if type(self.blur_step) is not int or self.blur_step <= 1:
            raise ValueError("blur_step must be an integer greater than one")
        if not math.isfinite(self.wash) or not 0 <= self.wash <= 1:
            raise ValueError("wash must be in 0..1")


class SourceCompositionGraph:
    def __init__(self, cfg: SourceCompositionConfig, nodes_mod: Any) -> None:
        self._cfg, self._n = cfg, nodes_mod

    @property
    def _restart(self) -> str:
        return self._cfg.auto_restart

    def default_layers(self) -> list[dict[str, Any]]:
        cfg = self._cfg
        blur = cfg.background == "blur"
        default_layers = [
            {
                "input": 0,
                "dst_x": 0,
                "dst_y": 0,
                "dst_w": cfg.dest_width,
                "dst_h": cfg.dest_height,
                "fit": "contain",
            }
        ] + [
            {**hidden_layer(cfg.dest_width), "input": i}
            for i in range(1, cfg.max_layers)
        ]
        if blur:
            # Listed last (metadata addresses content layers by position) but
            # drawn first (z=-1). Covers the whole canvas; the content layers hide it.
            default_layers.append(
                {
                    "input": cfg.max_layers,
                    "dst_x": 0,
                    "dst_y": 0,
                    "dst_w": cfg.dest_width,
                    "dst_h": cfg.dest_height,
                    "fit": "stretch",
                    "z": -1,
                    "filter": "bicubic",
                    "bicubic_param": 0.5,
                    "yuv_lut": white_wash_lut(cfg.wash),
                }
            )
        return default_layers

    def build(self, nodes: list, src: str) -> str:
        cfg, n = self._cfg, self._n
        r = self._restart
        blur = cfg.background == "blur"
        pads = cfg.max_layers + int(blur)
        fanout = [f"{cfg.edge_prefix}_{i}" for i in range(cfg.max_layers)] + (
            [f"{cfg.edge_prefix}_bg"] if blur else []
        )
        default_layers = self.default_layers()
        nodes += [
            # filter_video needs a declared frame rate, including after metadata nodes.
            n.ForceFPS(
                {
                    "fps": cfg.fps,
                    "group": cfg.group,
                    "src": src,
                    "dst": f"{cfg.edge_prefix}_paced",
                    "name": f"{cfg.name}_Pace",
                    "auto_restart": r,
                }
            ),
            # cuda_rect_overlay with color=sdr requires BT.709 MPEG tags on its source;
            # NVDEC frames often leave them unspecified.
            n.FilterVideo(
                {
                    "src": f"{cfg.edge_prefix}_paced",
                    "dst": f"{cfg.edge_prefix}_color",
                    "group": cfg.group,
                    "name": f"{cfg.name}_Color",
                    "auto_restart": r,
                    "hwaccel": cfg.hwaccel,
                    "graph": "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv",
                    "dst_pixel_format": "cuda",
                }
            ),
            # Every pad is permanently active. Split also forwards EOF to every
            # branch, so the background and compositor drain on finite inputs.
            n.Split(
                {
                    "src": f"{cfg.edge_prefix}_color",
                    "dst": fanout,
                    "group": cfg.group,
                    "name": f"{cfg.name}_Fanout",
                    "auto_restart": r,
                }
            ),
        ]
        compose_src = fanout
        if blur:
            nodes += self._background_nodes(fanout[-1])
            compose_src = [*fanout[:-1], f"{cfg.edge_prefix}_bg_small"]
        nodes += [
            source_compositor(
                {
                    "src": compose_src,
                    "dst": cfg.dst,
                    "group": cfg.group,
                    "name": cfg.name,
                    "auto_restart": r,
                    "hwaccel": cfg.hwaccel,
                    "width": cfg.dest_width,
                    "height": cfg.dest_height,
                    "sw_format": "nv12",
                    "color": "sdr",
                    "metadata_key": cfg.metadata_key,
                    "move_metadata_key": cfg.move_metadata_key,
                    "move_metadata_input": 0,
                    "active_inputs": source_mask_param((1 << pads) - 1),
                    "layers": default_layers,
                },
                api=n,
            ),
        ]
        return cfg.dst

    def _background_nodes(self, src: str) -> list[Any]:
        """Area-downsampled centre crop, washed before bicubic enlargement.

        Defaults preserve the existing 99/1080-height blur approximation and
        30% white wash. Shrinking in stages avoids aliasing sharp text into blocks.
        """
        from .scaling import auto_filter, center_crop, cuda_scale

        cfg, n = self._cfg, self._n
        texel = max(2.0, cfg.dest_height * cfg.blur_fraction / 2)
        w = max(2, round(cfg.dest_width / texel) // 2 * 2)
        h = max(2, round(cfg.dest_height / texel) // 2 * 2)
        crop = center_crop(cfg.source_width, cfg.source_height, w, h)
        ratio = max(crop["w"] / w, crop["h"] / h)
        steps = max(1, math.ceil(math.log(ratio, cfg.blur_step))) if ratio > 1 else 1
        nodes: list[Any] = []
        cur = src
        for i in range(steps):
            last = i == steps - 1
            f = (i + 1) / steps
            sw = w if last else max(w, int(crop["w"] * (w / crop["w"]) ** f) // 2 * 2)
            sh = h if last else max(h, int(crop["h"] * (h / crop["h"]) ** f) // 2 * 2)
            dst = f"{cfg.edge_prefix}_bg_small" if last else f"{cfg.edge_prefix}_bg_{i}"
            nodes.append(
                cuda_scale(
                    n,
                    name=f"{cfg.name}_Bg" if i == 0 else f"{cfg.name}_Bg{i}",
                    src=cur,
                    dst=dst,
                    width=sw,
                    height=sh,
                    group=cfg.group,
                    auto_restart=self._restart,
                    fit="stretch",
                    filter_=auto_filter(),
                    crop=crop if i == 0 else None,
                    hwaccel=cfg.hwaccel,
                )
            )
            cur = dst
        return nodes
