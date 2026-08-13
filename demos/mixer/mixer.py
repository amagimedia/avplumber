"""N-input, video-only manual mixer demonstration.

The graph exposes fullscreen and paged 2/4/8/16-box scenes through the generic
AVPlumber mixer control protocol. It contains no audio or automatic selection
path; use ``tui.py`` to preview and take scenes manually.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
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


DEFAULT_FPS = 30
FPS_DEN = 1
HWACCEL = "@gpu"
MIXER_NAME = "mixer"
ROUTER_GROUP = "mixer_preheat_router"
GEOMETRY_GROUP = "mixer_preheated_geometry"
OUTPUT_GROUP = "output"
JANUS_DEFAULT_HOST = "127.0.0.1"
JANUS_DEFAULT_VIDEO_PORT = 5004
JANUS_DEFAULT_VIDEO_PT = 96
JANUS_DEFAULT_VIDEO_SSRC = 0x41565001
JANUS_DEFAULT_VIDEO_BITRATE_KBPS = 3_000
RTP_PACKET_SIZE = 1_200
PREHEAT_POLL_INTERVAL_SEC = 0.02


@dataclass(frozen=True)
class GraphOptions:
    inputs: tuple[str, ...]
    output: str | None = None
    output_format: str | None = None
    remote_control_port: int = 7777
    codec: str = "h264_nvenc"
    bitrate: str = "8M"
    fps: int = DEFAULT_FPS
    loop_inputs: bool = False
    janus_output: bool = False
    janus_host: str = JANUS_DEFAULT_HOST
    janus_video_port: int = JANUS_DEFAULT_VIDEO_PORT
    janus_video_pt: int = JANUS_DEFAULT_VIDEO_PT
    janus_video_ssrc: int = JANUS_DEFAULT_VIDEO_SSRC
    janus_video_bitrate_kbps: int = JANUS_DEFAULT_VIDEO_BITRATE_KBPS
    janus_rtcp_bind: str = "0.0.0.0"
    janus_rtcp_port: int = 0
    preheat_timeout_sec: float = 60.0

    def validate(self) -> None:
        if not self.inputs:
            raise ValueError("at least one input is required")
        if not self.output and not self.janus_output:
            raise ValueError("--output or --janus-output is required")
        if not self.codec.endswith("_nvenc"):
            raise ValueError("--codec must be an NVENC encoder for zero-copy output")
        if not 1 <= self.fps <= 240:
            raise ValueError("--fps must be between 1 and 240")
        if not 0 <= self.remote_control_port <= 65535:
            raise ValueError("remote_control_port must be between 0 and 65535")
        if not 0 <= self.janus_rtcp_port <= 65535:
            raise ValueError("janus_rtcp_port must be between 0 and 65535")
        if self.preheat_timeout_sec <= 0:
            raise ValueError("preheat_timeout_sec must be positive")
        if self.janus_output:
            if not self.janus_host:
                raise ValueError("janus_host is required for Janus output")
            if not 1 <= self.janus_video_port < 65535:
                raise ValueError("janus_video_port and its RTCP pair must be valid")
            if not 0 <= self.janus_video_pt <= 127:
                raise ValueError("janus_video_pt must be between 0 and 127")
            if not 0 <= self.janus_video_ssrc <= 0xFFFFFFFF:
                raise ValueError("janus_video_ssrc must be a 32-bit unsigned integer")
            if self.janus_video_bitrate_kbps <= 0:
                raise ValueError("janus_video_bitrate_kbps must be positive")


@dataclass
class MixerApplication:
    avp: object
    mixer: object
    input_groups: tuple[str, ...]
    normalized_input_edges: tuple[str, ...]
    preheated_output_edges: tuple[str, ...]
    preheat_timeout_sec: float
    rtcp_feedback_listener: object | None = None

    def _wait_for_edges(self, edges: tuple[str, ...], phase: str) -> None:
        deadline = time.monotonic() + self.preheat_timeout_sec
        while True:
            missing = [edge for edge in edges if self.avp.getEdge(edge).occupied == 0]
            if not missing:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"mixer preheat timed out during {phase}; "
                    f"{len(missing)}/{len(edges)} edges have no frame, first={missing[0]}"
                )
            time.sleep(PREHEAT_POLL_INTERVAL_SEC)

    def _wait_for_node(self, name: str) -> None:
        deadline = time.monotonic() + self.preheat_timeout_sec
        while not self.avp.node(name).isWorking:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"mixer preheat timed out starting node {name}")
            time.sleep(PREHEAT_POLL_INTERVAL_SEC)

    def start(self) -> None:
        for group in self.input_groups:
            self.avp.group(group).startNodes()
        self._wait_for_edges(self.normalized_input_edges, "input normalization")
        self.avp.group(ROUTER_GROUP).startNodes()
        self._wait_for_node("layout_preheat_router")
        self.avp.group(GEOMETRY_GROUP).startNodes()
        self._wait_for_edges(self.preheated_output_edges, "geometry warm-up")
        self.mixer.initialize_routes()
        self.mixer.start_groups()
        for node in (
            "mixer_comp_a",
            "mixer_comp_b",
            "mixer_otm_scene_a",
            "mixer_otm_scene_b",
            "mixer_out_sel_transition",
        ):
            self._wait_for_node(node)
        self.mixer.begin_transition_preheat()
        self._wait_for_edges(("mixer_trans_out",), "transition warm-up")
        self.mixer.finish_transition_preheat()
        self.avp.group(OUTPUT_GROUP).startNodes()
        if self.rtcp_feedback_listener is not None:
            self.rtcp_feedback_listener.start()
        self.avp.setReady()
        print(
            f"Generic mixer preheat complete: "
            f"{len(self.preheated_output_edges)} geometry paths and transition ready",
            flush=True,
        )

    def stop(self) -> None:
        if self.rtcp_feedback_listener is not None:
            self.rtcp_feedback_listener.stop()
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
        Bsf,
        DecVideo,
        Demux,
        EncVideo,
        FilterVideo,
        ForceFPS,
        ForceKeyFrame,
        InputRec,
        Mux,
        Output,
        PreheatVideoRouter,
        Realtime,
        Split,
    )
    from pyplumber.rtcp_feedback import RtcpFeedbackListener

    return SimpleNamespace(
        AVPlumber=AVPlumber,
        MixerGraphBuilder=MixerGraphBuilder,
        AssumeVideoFormat=AssumeVideoFormat,
        Bsf=Bsf,
        DecVideo=DecVideo,
        Demux=Demux,
        EncVideo=EncVideo,
        FilterVideo=FilterVideo,
        ForceFPS=ForceFPS,
        ForceKeyFrame=ForceKeyFrame,
        InputRec=InputRec,
        Mux=Mux,
        Output=Output,
        PreheatVideoRouter=PreheatVideoRouter,
        Realtime=Realtime,
        RtcpFeedbackListener=RtcpFeedbackListener,
        Split=Split,
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


def _build_input(
    avp, api, index: int, url: str, *, loop: bool, fps: int
) -> str:
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
        "fps": f"{fps}/{FPS_DEN}",
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
        "dst_frame_rate": f"{fps}/{FPS_DEN}",
        "hwaccel": HWACCEL,
        "auto_restart": "group",
        "group": group,
    }))
    return normalized_edge


def _route_edge(capacity: int, slot: int, mixer_slot: str) -> str:
    return f"route_{capacity}_{slot}_{mixer_slot.lower()}"


def _route_label(capacity: int, slot: int, mixer_slot: str) -> str:
    return f"capacity_{capacity}_slot_{slot}_{mixer_slot.lower()}"


def _build_preheated_layouts(
    avp, api, mixer, input_edges: list[str], *, fps: int
) -> tuple[str, ...]:
    outputs = []
    labels = []
    preheated_output_edges = []
    for capacity in PREHEAT_CAPACITIES:
        for slot in range(capacity):
            for mixer_slot in ("A", "B"):
                outputs.append(_route_edge(capacity, slot, mixer_slot))
                labels.append(_route_label(capacity, slot, mixer_slot))

    avp.addNode(api.PreheatVideoRouter({
        "name": "layout_preheat_router",
        "src": input_edges,
        "dst": outputs,
        "routes": [index % len(input_edges) for index in range(len(outputs))],
        "labels": labels,
        "width": CANONICAL_SOURCE_WIDTH,
        "height": CANONICAL_SOURCE_HEIGHT,
        "pixel_format": "cuda",
        "real_pixel_format": "nv12",
        "frame_rate": f"{fps}/{FPS_DEN}",
        "timebase": f"{FPS_DEN}/{fps}",
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
            preheated_output_edges.extend((
                f"{MIXER_NAME}_{layout_source_name(capacity, slot)}_scaled_a",
                f"{MIXER_NAME}_{layout_source_name(capacity, slot)}_scaled_b",
            ))
    return tuple(preheated_output_edges)


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


def _build_record_output(avp, api, options: GraphOptions, mixer_edge: str) -> None:
    if options.output is None:
        raise ValueError("record output needs an output URL or path")
    fps_edge = "program_fps"
    assumed_edge = "program_video"
    encoded_edge = "program_encoded"
    muxed_edge = "program_muxed"
    avp.addNode(api.ForceFPS({
        "name": "program_fps",
        "src": mixer_edge,
        "dst": fps_edge,
        "fps": f"{options.fps}/{FPS_DEN}",
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
            "g": options.fps * 2,
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


def _rtp_url(host: str, port: int, *, rtcp_port: int) -> str:
    return (
        f"rtp://{host}:{port}?pkt_size={RTP_PACKET_SIZE}"
        f"&rtcp_port={rtcp_port}"
    )


def _build_janus_output(avp, api, options: GraphOptions, mixer_edge: str) -> None:
    fps_edge = "janus_fps"
    keyframe_edge = "janus_keyframed"
    assumed_edge = "janus_video"
    encoded_edge = "janus_encoded"
    headers_edge = "janus_repeat_headers"
    muxed_edge = "janus_video_rtp_mux"
    bitrate = f"{options.janus_video_bitrate_kbps}k"

    avp.addNode(api.ForceFPS({
        "name": "janus_fps",
        "src": mixer_edge,
        "dst": fps_edge,
        "fps": f"{options.fps}/{FPS_DEN}",
        "group": OUTPUT_GROUP,
    }))
    avp.addNode(api.ForceKeyFrame({
        "name": "janus_force_keyframe",
        "src": fps_edge,
        "dst": keyframe_edge,
        "interval_sec": "1/1",
        "auto_restart": "panic",
        "group": OUTPUT_GROUP,
    }))
    avp.addNode(api.AssumeVideoFormat({
        "name": "janus_format",
        "src": keyframe_edge,
        "dst": assumed_edge,
        "width": CANVAS_WIDTH,
        "height": CANVAS_HEIGHT,
        "pixel_format": "cuda",
        "real_pixel_format": "nv12",
        "auto_restart": "panic",
        "group": OUTPUT_GROUP,
    }))
    avp.addNode(api.EncVideo({
        "name": "janus_encoder",
        "src": assumed_edge,
        "dst": encoded_edge,
        "codec": "h264_nvenc",
        "hwaccel": HWACCEL,
        "options": {
            "b": bitrate,
            "maxrate": bitrate,
            "bufsize": bitrate,
            "g": options.fps,
            "bf": 0,
            "preset": "p6",
            "profile": "baseline",
            "tune": "ull",
            "rc": "cbr",
            "rc-lookahead": 0,
            "zerolatency": 1,
            "delay": 0,
            "forced-idr": 1,
            "no-scenecut": 1,
            "strict_gop": 1,
            "aud": 1,
            "spatial-aq": 1,
            "temporal-aq": 0,
        },
        "auto_restart": "panic",
        "group": OUTPUT_GROUP,
    }))
    avp.addNode(api.Bsf({
        "name": "janus_repeat_headers",
        "src": encoded_edge,
        "dst": headers_edge,
        "bsf": "dump_extra=freq=keyframe",
        "auto_restart": "panic",
        "group": OUTPUT_GROUP,
    }))
    avp.addNode(api.Mux({
        "name": "janus_mux",
        "src": [headers_edge],
        "dst": muxed_edge,
        "ts_sort_wait": 0,
        "auto_restart": "on",
        "on_error": "panic",
        "group": OUTPUT_GROUP,
    }))
    avp.addNode(api.Output({
        "name": "janus_rtp_output",
        "src": muxed_edge,
        "url": _rtp_url(
            options.janus_host,
            options.janus_video_port,
            rtcp_port=options.janus_video_port + 1,
        ),
        "format": "rtp",
        "options": {
            "payload_type": options.janus_video_pt,
            "rtpflags": "skip_rtcp",
            "ssrc": options.janus_video_ssrc,
        },
        "auto_restart": "on",
        "on_error": "panic",
        "group": OUTPUT_GROUP,
    }))


def _build_outputs(avp, api, options: GraphOptions, mixer_edge: str):
    record_edge = mixer_edge
    janus_edge = mixer_edge
    if options.output and options.janus_output:
        record_edge = "program_video_record"
        janus_edge = "program_video_janus"
        avp.addNode(api.Split({
            "name": "split_program_video_output",
            "src": mixer_edge,
            "dst": [record_edge, janus_edge],
            "group": OUTPUT_GROUP,
            "on_error": "panic",
        }))

    if options.output:
        _build_record_output(avp, api, options, record_edge)
    if not options.janus_output:
        return None

    _build_janus_output(avp, api, options, janus_edge)

    def trigger_keyframe(_request: str) -> None:
        avp.executeCommandsFromString(
            "node.object.set janus_force_keyframe trigger true"
        )

    return api.RtcpFeedbackListener(
        bind_host=options.janus_rtcp_bind,
        bind_port=options.janus_rtcp_port,
        janus_host=options.janus_host,
        janus_rtcp_port=options.janus_video_port + 1,
        media_ssrc=options.janus_video_ssrc,
        on_keyframe_request=trigger_keyframe,
    )


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
        _build_input(
            avp,
            api,
            index,
            url,
            loop=options.loop_inputs,
            fps=options.fps,
        )
        for index, url in enumerate(options.inputs)
    ]
    mixer = api.MixerGraphBuilder(
        avp,
        name=MIXER_NAME,
        canvas=(CANVAS_WIDTH, CANVAS_HEIGHT),
        fps=(options.fps, FPS_DEN),
        hwaccel=HWACCEL,
        enable_wipe=False,
        defer_initial_routes=True,
    )
    preheated_output_edges = _build_preheated_layouts(
        avp, api, mixer, input_edges, fps=options.fps
    )
    _define_scenes(mixer, len(input_edges))
    mixer.set_initial_scene("fullscreen_0", slot="A")
    mixer_edge = mixer.build()
    rtcp_feedback_listener = _build_outputs(avp, api, options, mixer_edge)
    return MixerApplication(
        avp=avp,
        mixer=mixer,
        input_groups=tuple(_input_group(index) for index in range(len(input_edges))),
        normalized_input_edges=tuple(input_edges),
        preheated_output_edges=preheated_output_edges,
        preheat_timeout_sec=options.preheat_timeout_sec,
        rtcp_feedback_listener=rtcp_feedback_listener,
    )


def parse_args(argv: list[str] | None = None) -> GraphOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        required=True,
        metavar="PATH",
        help="Input media file or URL; repeat for each mixer input",
    )
    parser.add_argument("--output", help="Optional video-only output URL or path")
    parser.add_argument(
        "--output-format",
        help="Muxer format when it cannot be inferred from the output",
    )
    parser.add_argument("--remote-control-port", type=int, default=7777)
    parser.add_argument("--codec", default="h264_nvenc")
    parser.add_argument("--bitrate", default="8M")
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help=f"Mixer and output frame rate (default: {DEFAULT_FPS})",
    )
    parser.add_argument("--loop-inputs", action="store_true")
    parser.add_argument(
        "--janus-output",
        action="store_true",
        help="Publish the video-only program to Janus over RTP",
    )
    parser.add_argument("--janus-host", default=JANUS_DEFAULT_HOST)
    parser.add_argument(
        "--janus-video-port", type=int, default=JANUS_DEFAULT_VIDEO_PORT
    )
    parser.add_argument(
        "--janus-video-pt", type=int, default=JANUS_DEFAULT_VIDEO_PT
    )
    parser.add_argument(
        "--janus-video-ssrc",
        type=lambda value: int(value, 0),
        default=JANUS_DEFAULT_VIDEO_SSRC,
    )
    parser.add_argument(
        "--janus-video-bitrate-kbps",
        type=int,
        default=JANUS_DEFAULT_VIDEO_BITRATE_KBPS,
    )
    parser.add_argument("--janus-rtcp-bind", default="0.0.0.0")
    parser.add_argument("--janus-rtcp-port", type=int, default=0)
    parser.add_argument("--preheat-timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    return GraphOptions(
        inputs=tuple(args.inputs),
        output=args.output,
        output_format=args.output_format,
        remote_control_port=args.remote_control_port,
        codec=args.codec,
        bitrate=args.bitrate,
        fps=args.fps,
        loop_inputs=args.loop_inputs,
        janus_output=args.janus_output,
        janus_host=args.janus_host,
        janus_video_port=args.janus_video_port,
        janus_video_pt=args.janus_video_pt,
        janus_video_ssrc=args.janus_video_ssrc,
        janus_video_bitrate_kbps=args.janus_video_bitrate_kbps,
        janus_rtcp_bind=args.janus_rtcp_bind,
        janus_rtcp_port=args.janus_rtcp_port,
        preheat_timeout_sec=args.preheat_timeout,
    )


def main(argv: list[str] | None = None) -> None:
    options = parse_args(argv)
    application = build_application(options)
    application.start()
    targets = []
    if options.output:
        targets.append(options.output)
    if options.janus_output:
        targets.append(
            f"Janus RTP {options.janus_host}:{options.janus_video_port}"
        )
    print(
        f"Generic mixer started: {len(options.inputs)} inputs -> "
        f"{', '.join(targets)} at {options.fps} fps; control port "
        f"{options.remote_control_port or 'disabled'}"
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
