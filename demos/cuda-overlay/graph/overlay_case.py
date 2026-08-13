#!/usr/bin/env python3
"""Run one overlay_many_cuda mode/count case through a PyPlumber graph."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_DIR / "tools"))
sys.path.insert(0, os.environ.get("PYPLUMBER_PATH", str(DEMO_DIR.parent)))

from overlay_common import (  # noqa: E402
    HEIGHT,
    MAX_OVERLAYS,
    MODES,
    WIDTH,
    base_fixture_path,
    frame_size,
    overlay_fixture_path,
)

import pyplumber  # noqa: E402
from pyplumber.node import (  # noqa: E402
    DecVideo,
    Demux,
    FilterVideo,
    InputRec,
    InternalNode,
)


OUTPUT_GROUP = "cuda_overlay_case"
SOURCE_GROUP = "cuda_overlay_sources"
COMPOSE_GROUP = "cuda_overlay_compose"
GPU_DEVICE = "cuda_overlay_gpu"


class RawVideoOutput(InternalNode):
    TYPE = "raw_output"


def _source_nodes(
    index: int,
    source: Path,
    pixel_format: str,
    width: int,
    height: int,
) -> tuple[list[object], str]:
    prefix = f"source_{index:02d}"
    packet_input = f"{prefix}_input"
    packet = f"{prefix}_packet"
    decoded = f"{prefix}_decoded"
    nodes: list[object] = [
        InputRec(
            {
                "name": f"{prefix}_reader",
                "url": str(source),
                "format": "rawvideo",
                "options": {
                    "video_size": f"{width}x{height}",
                    "pixel_format": pixel_format,
                    "framerate": "1/1",
                },
                "dst": packet_input,
                "loop": False,
                "timeout": -1,
                "group": SOURCE_GROUP,
                "auto_restart": "off",
            }
        ),
        Demux(
            {
                "name": f"{prefix}_demux",
                "src": packet_input,
                "routing": {"v:0": packet},
                "wait_for_keyframe": False,
                "group": SOURCE_GROUP,
                "auto_restart": "off",
            }
        ),
        DecVideo(
            {
                "name": f"{prefix}_decode",
                "src": packet,
                "dst": decoded,
                "group": SOURCE_GROUP,
                "auto_restart": "off",
            }
        ),
    ]
    return nodes, decoded


def run_case(
    mode_name: str,
    overlay_count: int,
    raw_dir: Path,
    output: Path,
    timeout: float,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> None:
    if mode_name not in MODES:
        raise ValueError(f"unsupported mode: {mode_name}")
    if not 1 <= overlay_count <= MAX_OVERLAYS:
        raise ValueError(f"overlay count must be in 1..{MAX_OVERLAYS}")
    if width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    mode = MODES[mode_name]
    sources = [base_fixture_path(raw_dir, mode.main_format)] + [
        overlay_fixture_path(raw_dir, index, mode.overlay_format)
        for index in range(1, overlay_count + 1)
    ]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    avp = pyplumber.AVPlumber()
    avp.executeCommandsFromString(
        f'hwaccel.init {{ "name": "{GPU_DEVICE}", "type": "cuda" }}'
    )

    software_edges: list[str] = []
    for index, source in enumerate(sources):
        pixel_format = mode.main_format if index == 0 else mode.overlay_format
        nodes, software_edge = _source_nodes(index, source, pixel_format, width, height)
        for node in nodes:
            avp.addNode(node, early_create=True)
        software_edges.append(software_edge)
    del node

    downloaded = "downloaded_yuv"
    total_inputs = overlay_count + 1
    upload_chains = [
        f"[source_{index:02d}]hwupload[uploaded_{index:02d}]"
        for index in range(total_inputs)
    ]
    uploaded_inputs = "".join(
        f"[uploaded_{index:02d}]" for index in range(total_inputs)
    )
    compose_chain = (
        f"{uploaded_inputs}overlay_many_cuda=inputs={total_inputs}:"
        f"shortest=1:repeatlast=0,hwdownload,format={mode.main_format}"
    )
    filter_graph = ";".join([*upload_chains, compose_chain])
    tail_nodes = [
        FilterVideo(
            {
                "name": "overlay_many_cuda",
                "src": software_edges,
                "dst": downloaded,
                "graph": filter_graph,
                "hwaccel": GPU_DEVICE,
                "dst_width": width,
                "dst_height": height,
                "dst_pixel_format": mode.main_format,
                "group": COMPOSE_GROUP,
                "auto_restart": "off",
            }
        ),
        RawVideoOutput(
            {
                "name": "rawvideo_output",
                "src": downloaded,
                "path": str(output),
                "output_group": OUTPUT_GROUP,
                "group": COMPOSE_GROUP,
                "auto_restart": "off",
            }
        ),
    ]
    avp.addNode(tail_nodes[0], early_create=True)
    avp.addNode(tail_nodes[1], early_create=True)

    expected_size = frame_size(mode.main_format, width, height)
    started = time.monotonic()
    produced_frame = False
    try:
        avp.group(SOURCE_GROUP).startNodes()
        while time.monotonic() - started < timeout:
            if all(avp.getEdge(edge, "VideoFrame").occupied for edge in software_edges):
                break
            time.sleep(0.01)
        else:
            missing = [
                edge
                for edge in software_edges
                if not avp.getEdge(edge, "VideoFrame").occupied
            ]
            raise TimeoutError(f"source frames not ready: {', '.join(missing)}")

        avp.executeCommandsFromString(f"output.start {OUTPUT_GROUP}")
        avp.group(COMPOSE_GROUP).startNodes()
        while time.monotonic() - started < timeout:
            avp.heartbeat()
            if output.exists():
                actual_size = output.stat().st_size
                if actual_size >= expected_size:
                    produced_frame = True
                    break
            time.sleep(0.05)
        if not produced_frame:
            actual_size = output.stat().st_size if output.exists() else 0
            raise TimeoutError(
                f"case timed out after {timeout:.1f}s with {actual_size}/{expected_size} output bytes"
            )
    finally:
        avp.executeCommandsFromString(f"output.stop {OUTPUT_GROUP}")
        avp.group(COMPOSE_GROUP).stopNodes()
        avp.group(SOURCE_GROUP).stopNodes()
        node_names = [
            *(f"source_{index:02d}_{suffix}" for index in range(total_inputs) for suffix in ("reader", "demux", "decode")),
            "overlay_many_cuda",
            "rawvideo_output",
        ]
        stop_deadline = time.monotonic() + 5.0
        while time.monotonic() < stop_deadline:
            if not any(avp.node(name).isWorking for name in node_names):
                break
            time.sleep(0.01)
        nodes.clear()
        tail_nodes.clear()
        avp.shutdown()

    with output.open("r+b") as stream:
        stream.truncate(expected_size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--overlays", required=True, type=int)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    args = parser.parse_args()
    run_case(
        args.mode,
        args.overlays,
        args.raw_dir,
        args.output,
        args.timeout,
        args.width,
        args.height,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
