"""N-input, video-only manual mixer demonstration.

The graph exposes fullscreen and paged 2/4/8/16-box scenes through the generic
AVPlumber mixer control protocol. It contains no audio or automatic selection
path; use ``tui.py`` to preview and take scenes manually.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

try:
    from .layouts import (
        CANONICAL_SOURCE_HEIGHT,
        CANONICAL_SOURCE_WIDTH,
        CANVAS_HEIGHT,
        CANVAS_WIDTH,
        PREHEAT_CAPACITIES,
        all_scenes,
        layout_filter_graph,
        layout_source_name,
    )
except ImportError:
    from layouts import (  # type: ignore[no-redef]
        CANONICAL_SOURCE_HEIGHT,
        CANONICAL_SOURCE_WIDTH,
        CANVAS_HEIGHT,
        CANVAS_WIDTH,
        PREHEAT_CAPACITIES,
        all_scenes,
        layout_filter_graph,
        layout_source_name,
    )


FPS_NUM = 30
FPS_DEN = 1
HWACCEL = "@gpu"
MIXER_NAME = "mixer"
ROUTER_GROUP = "mixer_preheat_router"
GEOMETRY_GROUP = "mixer_preheated_geometry"
OUTPUT_GROUP = "output"


@dataclass(frozen=True)
class GraphOptions:
    inputs: tuple[str, ...]
    output: str
    output_format: str | None = None
    remote_control_port: int = 7777
    codec: str = "h264_nvenc"
    bitrate: str = "8M"
    loop_inputs: bool = False
    enable_wipe: bool = True

    def validate(self) -> None:
        if not self.inputs:
            raise ValueError("at least one input is required")
        if not self.output:
            raise ValueError("output is required")
        if not 0 <= self.remote_control_port <= 65535:
            raise ValueError("remote_control_port must be between 0 and 65535")


@dataclass
class MixerApplication:
    avp: object
    mixer: object
    input_groups: tuple[str, ...]

    def start(self) -> None:
        for group in self.input_groups:
            self.avp.group(group).startNodes()
        self.avp.group(ROUTER_GROUP).startNodes()
        self.avp.group(GEOMETRY_GROUP).startNodes()
        self.mixer.start_groups()
        self.avp.group(OUTPUT_GROUP).startNodes()

    def stop(self) -> None:
        groups = (
            OUTPUT_GROUP,
            f"{MIXER_NAME}_b",
            f"{MIXER_NAME}_a",
            MIXER_NAME,
            GEOMETRY_GROUP,
            ROUTER_GROUP,
            *reversed(self.input_groups),
        )
        for group in groups:
            try:
                self.avp.group(group).stopNodes()
            except Exception:
                pass


def load_avp_api():
    from pyplumber import AVPlumber
    from pyplumber.mixer import MixerGraphBuilder
    from pyplumber.node import (
        AssumeVideoFormat,
        DecVideo,
        Demux,
        EncVideo,
        FilterVideo,
        ForceFPS,
        InputRec,
        Mux,
        Output,
        PreheatVideoRouter,
        Realtime,
    )

    return SimpleNamespace(
        AVPlumber=AVPlumber,
        MixerGraphBuilder=MixerGraphBuilder,
        AssumeVideoFormat=AssumeVideoFormat,
        DecVideo=DecVideo,
        Demux=Demux,
        EncVideo=EncVideo,
        FilterVideo=FilterVideo,
        ForceFPS=ForceFPS,
        InputRec=InputRec,
        Mux=Mux,
        Output=Output,
        PreheatVideoRouter=PreheatVideoRouter,
        Realtime=Realtime,
    )


def infer_output_format(output: str, explicit_format: str | None = None) -> str:
    if explicit_format:
        return explicit_format
    lowered = output.lower()
    if lowered.startswith("rtmp://") or lowered.endswith(".flv"):
        return "flv"
    if lowered.startswith("srt://") or lowered.endswith(".ts"):
        return "mpegts"
    if lowered.endswith(".mp4"):
        return "mp4"
    if lowered.endswith((".mkv", ".webm")):
        return "matroska"
    raise ValueError("cannot infer output format; pass --output-format")


def _input_group(index: int) -> str:
    return f"input_{index}"


def _build_input(avp, api, index: int, url: str, *, loop: bool) -> str:
    group = _input_group(index)
    packet_edge = f"input_{index}_packets"
    video_packet_edge = f"input_{index}_video_packets"
    decoded_edge = f"input_{index}_decoded"
    realtime_edge = f"input_{index}_realtime"
    fps_edge = f"input_{index}_fps"
    normalized_edge = f"input_{index}_normalized"

    avp.addNode(api.InputRec({
        "name": f"input_{index}",
        "url": url,
        "dst": packet_edge,
        "loop": loop,
        "initial_timeout": 20,
        "timeout": 3_942_000_000,
        "group": group,
    }))
    avp.addNode(api.Demux({
        "name": f"demux_{index}",
        "src": packet_edge,
        "routing": {"?v:0": video_packet_edge},
        "wait_for_keyframe": False,
        "auto_restart": "group",
        "group": group,
    }))
    avp.addNode(api.DecVideo({
        "name": f"decode_{index}",
        "src": video_packet_edge,
        "dst": decoded_edge,
        "pixel_format": "?cuda",
        "hwaccel": HWACCEL,
        "auto_restart": "group",
        "group": group,
    }))
    avp.addNode(api.Realtime({
        "name": f"realtime_{index}",
        "src": decoded_edge,
        "dst": realtime_edge,
        "set_pts": True,
        "group": group,
    }))
    avp.addNode(api.ForceFPS({
        "name": f"fps_{index}",
        "src": realtime_edge,
        "dst": fps_edge,
        "fps": f"{FPS_NUM}/{FPS_DEN}",
        "group": group,
    }))
    avp.addNode(api.FilterVideo({
        "name": f"normalize_{index}",
        "src": fps_edge,
        "dst": normalized_edge,
        "graph": (
            f"scale_cuda=w={CANONICAL_SOURCE_WIDTH}:h={CANONICAL_SOURCE_HEIGHT}:"
            "force_original_aspect_ratio=decrease:force_divisible_by=2,"
            f"pad_cuda=w={CANONICAL_SOURCE_WIDTH}:h={CANONICAL_SOURCE_HEIGHT}:"
            "x=(ow-iw)/2:y=(oh-ih)/2:color=black"
        ),
        "dst_width": CANONICAL_SOURCE_WIDTH,
        "dst_height": CANONICAL_SOURCE_HEIGHT,
        "dst_pixel_format": "cuda",
        "hwaccel": HWACCEL,
        "auto_restart": "group",
        "group": group,
    }))
    return normalized_edge


def _route_edge(capacity: int, slot: int, mixer_slot: str) -> str:
    return f"route_{capacity}_{slot}_{mixer_slot.lower()}"


def _route_label(capacity: int, slot: int, mixer_slot: str) -> str:
    return f"capacity_{capacity}_slot_{slot}_{mixer_slot.lower()}"


def _build_preheated_layouts(avp, api, mixer, input_edges: list[str]) -> None:
    outputs = []
    labels = []
    for capacity in PREHEAT_CAPACITIES:
        for slot in range(capacity):
            for mixer_slot in ("A", "B"):
                outputs.append(_route_edge(capacity, slot, mixer_slot))
                labels.append(_route_label(capacity, slot, mixer_slot))

    avp.addNode(api.PreheatVideoRouter({
        "name": "layout_preheat_router",
        "src": input_edges,
        "dst": outputs,
        "routes": [-1] * len(outputs),
        "labels": labels,
        "width": CANONICAL_SOURCE_WIDTH,
        "height": CANONICAL_SOURCE_HEIGHT,
        "pixel_format": "cuda",
        "real_pixel_format": "nv12",
        "frame_rate": f"{FPS_NUM}/{FPS_DEN}",
        "timebase": f"{FPS_DEN}/{FPS_NUM}",
        "timeline": mixer.timeline,
        "auto_restart": "group",
        "group": ROUTER_GROUP,
    }))

    for capacity in PREHEAT_CAPACITIES:
        graph = layout_filter_graph(capacity)
        for slot in range(capacity):
            mixer.add_routed_source(
                layout_source_name(capacity, slot),
                pre_filter_edge_a=_route_edge(capacity, slot, "A"),
                pre_filter_edge_b=_route_edge(capacity, slot, "B"),
                input_group=GEOMETRY_GROUP,
                route_router="layout_preheat_router",
                route_output_label_a=_route_label(capacity, slot, "A"),
                route_output_label_b=_route_label(capacity, slot, "B"),
                default_graph=graph,
            )


def _define_scenes(mixer, input_count: int) -> None:
    for scene in all_scenes(input_count):
        graph = layout_filter_graph(scene.capacity)
        sources = {}
        routes = {}
        for placement in scene.placements:
            source = layout_source_name(scene.capacity, placement.slot_index)
            sources[source] = {
                "graph": graph,
                "dst_x": placement.x,
                "dst_y": placement.y,
            }
            routes[source] = placement.source_index
        mixer.add_scene(scene.name, sources, routes=routes)


def _build_output(avp, api, options: GraphOptions, mixer_edge: str) -> None:
    fps_edge = "program_fps"
    assumed_edge = "program_video"
    encoded_edge = "program_encoded"
    muxed_edge = "program_muxed"
    avp.addNode(api.ForceFPS({
        "name": "program_fps",
        "src": mixer_edge,
        "dst": fps_edge,
        "fps": f"{FPS_NUM}/{FPS_DEN}",
        "group": OUTPUT_GROUP,
    }))
    avp.addNode(api.AssumeVideoFormat({
        "name": "program_format",
        "src": fps_edge,
        "dst": assumed_edge,
        "width": CANVAS_WIDTH,
        "height": CANVAS_HEIGHT,
        "pixel_format": "cuda",
        "real_pixel_format": "nv12",
        "group": OUTPUT_GROUP,
    }))
    avp.addNode(api.EncVideo({
        "name": "program_encoder",
        "src": assumed_edge,
        "dst": encoded_edge,
        "codec": options.codec,
        "hwaccel": HWACCEL,
        "options": {
            "b": options.bitrate,
            "maxrate": options.bitrate,
            "bufsize": options.bitrate,
            "g": FPS_NUM * 2,
            "bf": 0,
            "preset": "p3",
            "tune": "ll",
            "profile": "high",
        },
        "group": OUTPUT_GROUP,
    }))
    avp.addNode(api.Mux({
        "name": "program_mux",
        "src": [encoded_edge],
        "dst": muxed_edge,
        "ts_sort_wait": 0,
        "group": OUTPUT_GROUP,
    }))
    avp.addNode(api.Output({
        "name": "program_output",
        "src": muxed_edge,
        "url": options.output,
        "format": infer_output_format(options.output, options.output_format),
        "auto_restart": "panic",
        "group": OUTPUT_GROUP,
    }))


def build_application(options: GraphOptions, api=None) -> MixerApplication:
    options.validate()
    api = api or load_avp_api()
    avp = api.AVPlumber()
    if options.remote_control_port:
        avp.enableControlServer(options.remote_control_port)
    avp.executeCommandsFromString(
        f'hwaccel.init {{ "name": "{HWACCEL}", "type": "cuda" }}'
    )
    avp.edges.planCapacity("*", 4)

    input_edges = [
        _build_input(avp, api, index, url, loop=options.loop_inputs)
        for index, url in enumerate(options.inputs)
    ]
    mixer = api.MixerGraphBuilder(
        avp,
        name=MIXER_NAME,
        canvas=(CANVAS_WIDTH, CANVAS_HEIGHT),
        fps=(FPS_NUM, FPS_DEN),
        hwaccel=HWACCEL,
        enable_wipe=options.enable_wipe,
    )
    _build_preheated_layouts(avp, api, mixer, input_edges)
    _define_scenes(mixer, len(input_edges))
    mixer.set_initial_scene("fullscreen_0", slot="A")
    mixer_edge = mixer.build()
    _build_output(avp, api, options, mixer_edge)
    return MixerApplication(
        avp=avp,
        mixer=mixer,
        input_groups=tuple(_input_group(index) for index in range(len(input_edges))),
    )


def parse_args(argv: list[str] | None = None) -> GraphOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Input URLs or paths")
    parser.add_argument("--output", required=True, help="Video-only output URL or path")
    parser.add_argument(
        "--output-format",
        help="Muxer format when it cannot be inferred from the output",
    )
    parser.add_argument("--remote-control-port", type=int, default=7777)
    parser.add_argument("--codec", default="h264_nvenc")
    parser.add_argument("--bitrate", default="8M")
    parser.add_argument("--loop-inputs", action="store_true")
    parser.add_argument("--disable-wipe", action="store_true")
    args = parser.parse_args(argv)
    return GraphOptions(
        inputs=tuple(args.inputs),
        output=args.output,
        output_format=args.output_format,
        remote_control_port=args.remote_control_port,
        codec=args.codec,
        bitrate=args.bitrate,
        loop_inputs=args.loop_inputs,
        enable_wipe=not args.disable_wipe,
    )


def main(argv: list[str] | None = None) -> None:
    options = parse_args(argv)
    application = build_application(options)
    application.start()
    print(
        f"Generic mixer started: {len(options.inputs)} inputs -> {options.output}; "
        f"control port {options.remote_control_port or 'disabled'}"
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        application.stop()


if __name__ == "__main__":
    main()
