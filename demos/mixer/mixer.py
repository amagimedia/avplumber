"""N-input, video-only manual mixer demonstration.

The graph exposes fullscreen and paged 2/4/8/16-box scenes through the generic
AVPlumber mixer control protocol. It contains no audio or automatic selection
path; use ``tui.py`` to preview and take scenes manually.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace

from avpmixer import clipcache
from avpmixer import config as mixer_config
from avpmixer.dmabuf_inputs import (dmabuf_cuda_input_nodes, is_dmabuf_url, open_browser_windows,
                                    open_windows, refresh_windows, wait_for_sockets, window_id)
from avpmixer.inputs import build_input
from avpmixer.janus import JANUS_KEYFRAME_NODE, JanusVideoConfig, build_janus_output

try:
    from .layouts import (
        CANONICAL_SOURCE_HEIGHT,
        CANONICAL_SOURCE_WIDTH,
        CANVAS_HEIGHT,
        CANVAS_WIDTH,
        all_scenes,
    )
except ImportError:
    from layouts import (  # type: ignore[no-redef]
        CANONICAL_SOURCE_HEIGHT,
        CANONICAL_SOURCE_WIDTH,
        CANVAS_HEIGHT,
        CANVAS_WIDTH,
        all_scenes,
    )


DEFAULT_FPS = 30
FPS_DEN = 1
HWACCEL = "@gpu"
MIXER_NAME = "mixer"
ROUTER_GROUP = "mixer_preheat_router"
OUTPUT_GROUP = "output"
JANUS_DEFAULT_HOST = "127.0.0.1"
JANUS_DEFAULT_VIDEO_PORT = 5004
JANUS_DEFAULT_VIDEO_PT = 96
JANUS_DEFAULT_VIDEO_SSRC = 0x41565001
JANUS_DEFAULT_VIDEO_BITRATE_KBPS = 4_500
PREHEAT_POLL_INTERVAL_SEC = 0.02


@dataclass(frozen=True)
class GraphOptions:
    inputs: tuple[str, ...] = ()
    output: str | None = None
    output_format: str | None = None
    remote_control_port: int = 7777
    codec: str = "h264_nvenc"
    bitrate: str = "8M"
    fps: int = DEFAULT_FPS
    mixer_latency_ms: float | None = None
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
    wipe_file: str | None = None         # warm the media wipe chain up with this clip at start
    config: str | None = None            # JSON document (sources, wipes, scenes) instead of --input
    webui_url: str = ""                  # AVPlumber web UI to register the graph with
    cut_latency_encoder: str = ""        # opt-in cut-to-output observer on this encoder
    prewarm_cut_scenes: tuple[str, ...] = ()  # '*' selects all scene definitions
    # Wipe clips are held decoded in GPU memory by default: a take then costs no
    # file open, no decoder and no thread startup. 0 turns it off.
    wipe_cache_mb: float = 768.0
    # Browser pages from the DMA-BUF demo as sources: --input dmabuf://<window-id>
    dmabuf_socket_dir: str = "/tmp/dma-page"
    dmabuf_size: tuple[int, int] = (1280, 720)
    dmabuf_open: str | None = None       # page URL: open the named windows before building
    dmabuf_rest: str = "http://127.0.0.1:9009"

    @property
    def dmabuf_inputs(self) -> list[str]:
        return [window_id(url) for url in self.inputs if is_dmabuf_url(url)]

    def validate(self) -> None:
        if self.config and self.inputs:
            raise ValueError("--config replaces --input; pass one or the other")
        if not self.inputs and not self.config:
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
        if any(v <= 0 for v in self.dmabuf_size):
            raise ValueError("--dmabuf-size must be WxH with positive numbers")
        ids = self.dmabuf_inputs
        if len(ids) != len(set(ids)):
            raise ValueError("dmabuf window ids must be unique")
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
    input_edges: tuple[str, ...]
    routed_inputs: bool
    preheat_timeout_sec: float
    rtcp_feedback_listener: object | None = None
    wipe_file: str | None = None
    wipe_files: tuple[str, ...] = ()
    browser_windows: tuple[str, ...] = ()   # reloaded after the chains start: static pages paint only on load
    dmabuf_rest: str = ""
    wipe_cache_mb: float = 0.0              # hold decoded wipes in GPU memory
    cut_latency_encoder: str = ""
    prewarm_cut_scenes: tuple[str, ...] = ()

    def _preload_wipes(self) -> None:
        """Decode every wipe once into GPU memory (see avpmixer.clipcache).

        The loader group is started only here; a take starts the player group
        alone and replays what this left behind.
        """
        loader = clipcache.loader_group(MIXER_NAME)
        player = f"{MIXER_NAME}_wipe"
        cache_node = f"{MIXER_NAME}_wipe_cache"
        for clip in dict.fromkeys(c for c in (self.wipe_file, *self.wipe_files) if c):
            started = time.monotonic()
            # Both nodes need the clip before their group starts: the reader to
            # open the file, the cache to know which clip it is filling.
            value = json.dumps(clip)   # node.param.set parses the value as JSON
            self.avp.executeCommandsFromString(
                f"node.param.set {MIXER_NAME}_wipe_input url {value}\n"
                f"node.param.set {cache_node} url {value}")
            self.avp.group(player).startNodes()
            self.avp.group(loader).startNodes()
            deadline = started + self.preheat_timeout_sec
            held = None
            while time.monotonic() < deadline:
                try:
                    status = self.avp.node(cache_node).getObject("status")
                except Exception:
                    time.sleep(PREHEAT_POLL_INTERVAL_SEC)   # the group is still starting
                    continue
                held = next((c for c in status.get("clips", [])
                             if c["path"] == clip and c["complete"]), None)
                if held:
                    break
                time.sleep(PREHEAT_POLL_INTERVAL_SEC)
            self.avp.group(loader).stopNodes()
            self.avp.group(player).stopNodes()
            print(f"wipe cached: {clip} {held['frames'] if held else 0} frames, "
                  f"{(held['bytes'] if held else 0) / 1048576:.1f} MiB, "
                  f"{(time.monotonic() - started) * 1000:.0f} ms", flush=True)
            if not held:
                raise RuntimeError(f"wipe clip did not cache within the preheat timeout: {clip}")

    def _wait_for_edges(self, edges: tuple[str, ...], phase: str) -> None:
        deadline = time.monotonic() + self.preheat_timeout_sec
        while True:
            missing = [edge for edge in edges if self.avp.getEdge(edge).enqueued_total == 0]
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
        if self.browser_windows:
            refresh_windows(self.dmabuf_rest, list(self.browser_windows))
        self._wait_for_edges(self.input_edges, "input readiness")
        if self.routed_inputs:
            self.avp.group(ROUTER_GROUP).startNodes()
            self._wait_for_node("layout_preheat_router")
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
        try:
            self._wait_for_edges(("mixer_trans_out",), "transition warm-up")
        finally:
            self.mixer.finish_transition_preheat()
        self.mixer.start_output()
        self.avp.group(OUTPUT_GROUP).startNodes()
        self._wait_for_edges(("mixer_final_out",), "program output")
        if self.wipe_cache_mb:
            self._preload_wipes()
        else:
            for wipe_file in dict.fromkeys((self.wipe_file, *self.wipe_files)):
                if wipe_file:
                    self.mixer.warmup_wipe(wipe_file)
        if self.rtcp_feedback_listener is not None:
            self.rtcp_feedback_listener.start()
        if self.cut_latency_encoder:
            self._wait_for_node(self.cut_latency_encoder)
            self.avp.executeCommandsFromString("mixer.measurements " + json.dumps({
                "mixer": MIXER_NAME, "encoder": self.cut_latency_encoder,
            }))
        if self.prewarm_cut_scenes:
            scenes = self.mixer.scenes() if self.prewarm_cut_scenes == ("*",) else list(self.prewarm_cut_scenes)
            self.avp.executeCommandsFromString("mixer.prewarm " + json.dumps({"mixer": MIXER_NAME, "scenes": scenes}))
        self.avp.setReady()
        print(
            "Generic mixer preheat complete: compositors and transition ready",
            flush=True,
        )

    def stop(self) -> None:
        if self.rtcp_feedback_listener is not None:
            self.rtcp_feedback_listener.stop()
        self.avp.shutdown()


def load_avp_api():
    from pyplumber import AVPlumber
    from avpmixer import MixerGraphBuilder
    from pyplumber.node import (
        AssumeVideoFormat,
        Bsf,
        DecVideo,
        Demux,
        DrmPrimeToCuda,
        EncVideo,
        FilterVideo,
        ForceFPS,
        ForceKeyFrame,
        InputRec,
        IpcDmabufSource,
        Mux,
        OneToMany,
        Output,
        PreheatVideoRouter,
        Realtime,
        RepeatLastFrame,
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
        DrmPrimeToCuda=DrmPrimeToCuda,
        EncVideo=EncVideo,
        FilterVideo=FilterVideo,
        ForceFPS=ForceFPS,
        ForceKeyFrame=ForceKeyFrame,
        InputRec=InputRec,
        IpcDmabufSource=IpcDmabufSource,
        Mux=Mux,
        OneToMany=OneToMany,
        Output=Output,
        PreheatVideoRouter=PreheatVideoRouter,
        Realtime=Realtime,
        RepeatLastFrame=RepeatLastFrame,
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
    avp, api, index: int, url: str, *, loop: bool, fps: int, normalize: bool,
    options: "GraphOptions | None" = None,
) -> str:
    group = _input_group(index)
    if is_dmabuf_url(url):
        # A dma-browser window: DRM PRIME frames over its socket, already on the
        # shared monotonic clock; the same chain the DMA-BUF demo composes from.
        width, height = options.dmabuf_size
        nodes, fps_edge = dmabuf_cuda_input_nodes(
            api, prefix=f"input_{index}",
            socket=f"{options.dmabuf_socket_dir}/{window_id(url)}.sock",
            width=width, height=height, fps=fps, drm_hwaccel=None, cuda_hwaccel=HWACCEL,
            source_group=group, processing_group=group, hold=True)
        for node in nodes:
            avp.addNode(node)
    else:
        fps_edge = build_input(avp, api, str(index), url, group=group, fps=fps,
                               fps_den=FPS_DEN, hwaccel=HWACCEL, loop=loop)
    if not normalize:
        return fps_edge
    normalized_edge = f"input_{index}_normalized"
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


def _register_sources(avp, api, mixer, input_edges: list[str], *, fps: int) -> bool:
    # Compositor masks have 32 bits. Larger catalogues retain a small router
    # selecting the 16 visible positions, without per-layout filter branches.
    if len(input_edges) <= 32:
        for index, edge in enumerate(input_edges):
            mixer.add_source(f"source_{index}", pre_otm_edge=edge,
                             input_group=_input_group(index), default_graph="")
        return False
    labels = [f"slot_{i}_{slot}" for i in range(16) for slot in ("a", "b")]
    edges = [f"route_{label}" for label in labels]
    avp.addNode(api.PreheatVideoRouter({
        "name": "layout_preheat_router", "src": input_edges, "dst": edges,
        "routes": [-1] * len(edges), "labels": labels,
        "width": CANONICAL_SOURCE_WIDTH, "height": CANONICAL_SOURCE_HEIGHT,
        "pixel_format": "cuda", "real_pixel_format": "nv12",
        "frame_rate": f"{fps}/{FPS_DEN}", "timebase": f"{FPS_DEN}/{fps}",
        "timeline": mixer.timeline, "group": ROUTER_GROUP,
    }))
    for index in range(16):
        mixer.add_routed_source(
            f"source_{index}", pre_filter_edge_a=edges[2 * index],
            pre_filter_edge_b=edges[2 * index + 1], input_group=ROUTER_GROUP,
            route_router="layout_preheat_router", route_output_label_a=labels[2 * index],
            route_output_label_b=labels[2 * index + 1], default_graph="",
        )
    return True


def _define_scenes(mixer, input_count: int, routed: bool) -> None:
    for scene in all_scenes(input_count):
        sources, routes = {}, {}
        for placement in scene.placements:
            index = placement.slot_index if routed else placement.source_index
            source = f"source_{index}"
            sources[source] = {
                "dst_x": placement.x, "dst_y": placement.y,
                "dst_w": placement.width, "dst_h": placement.height, "fit": "contain",
            }
            if routed:
                routes[source] = placement.source_index
            else:
                # Preserve the same 16:9 source framing without materializing
                # an enlarged intermediate frame for every camera.
                sources[source]["source_canvas"] = {"w": CANONICAL_SOURCE_WIDTH, "h": CANONICAL_SOURCE_HEIGHT}
        mixer.add_scene(scene.name, sources, routes=routes)


def _build_record_output(avp, api, options: GraphOptions, mixer_edge: str, *,
                         width: int = CANVAS_WIDTH, height: int = CANVAS_HEIGHT) -> None:
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
        "width": width,
        "height": height,
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


def _build_outputs(avp, api, options: GraphOptions, mixer_edge: str, *,
                   width: int = CANVAS_WIDTH, height: int = CANVAS_HEIGHT):
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
        _build_record_output(avp, api, options, record_edge, width=width, height=height)
    if not options.janus_output:
        return None

    return build_janus_output(
        avp, api, janus_edge,
        JanusVideoConfig(
            host=options.janus_host, video_port=options.janus_video_port,
            payload_type=options.janus_video_pt, ssrc=options.janus_video_ssrc,
            bitrate_kbps=options.janus_video_bitrate_kbps,
            rtcp_bind=options.janus_rtcp_bind, rtcp_port=options.janus_rtcp_port,
        ),
        fps=options.fps, fps_den=FPS_DEN, width=width, height=height,
        hwaccel=HWACCEL, group=OUTPUT_GROUP,
    )


def build_application(options: GraphOptions, api=None) -> MixerApplication:
    options.validate()
    api = api or load_avp_api()
    if options.config:
        return _build_from_config(options, mixer_config.with_probed_sizes(mixer_config.load(options.config)), api)
    avp = api.AVPlumber()
    if options.remote_control_port:
        avp.enableControlServer(options.remote_control_port)
    avp.executeCommandsFromString(
        f'hwaccel.init {{ "name": "{HWACCEL}", "type": "cuda" }}'
    )
    dmabuf_ids = options.dmabuf_inputs
    if dmabuf_ids:
        if options.dmabuf_open:
            width, height = options.dmabuf_size
            open_browser_windows(options.dmabuf_rest, dmabuf_ids, options.dmabuf_open,
                                 width, height, options.fps)
        wait_for_sockets([f"{options.dmabuf_socket_dir}/{name}.sock" for name in dmabuf_ids],
                         options.preheat_timeout_sec)
    avp.edges.planCapacity("*", 4)

    input_edges = [
        _build_input(
            avp,
            api,
            index,
            url,
            loop=options.loop_inputs,
            fps=options.fps,
            normalize=len(options.inputs) > 32,
            options=options,
        )
        for index, url in enumerate(options.inputs)
    ]
    mixer = api.MixerGraphBuilder(
        avp,
        name=MIXER_NAME,
        canvas=(CANVAS_WIDTH, CANVAS_HEIGHT),
        fps=(options.fps, FPS_DEN),
        latency_ms=options.mixer_latency_ms,
        hwaccel=HWACCEL,
        enable_wipe=True,
        defer_initial_routes=True,
        defer_output=True,
        keyframe_node=JANUS_KEYFRAME_NODE if options.janus_output else None,
        cache_wipes_mb=options.wipe_cache_mb or None,
    )
    routed_inputs = _register_sources(
        avp, api, mixer, input_edges, fps=options.fps
    )
    _define_scenes(mixer, len(input_edges), routed_inputs)
    mixer.set_initial_scene("fullscreen_0", slot="A")
    mixer_edge = mixer.build()
    rtcp_feedback_listener = _build_outputs(avp, api, options, mixer_edge)
    return MixerApplication(
        avp=avp,
        mixer=mixer,
        input_groups=tuple(_input_group(index) for index in range(len(input_edges))),
        input_edges=tuple(input_edges),
        routed_inputs=routed_inputs,
        preheat_timeout_sec=options.preheat_timeout_sec,
        rtcp_feedback_listener=rtcp_feedback_listener,
        wipe_file=options.wipe_file,
        browser_windows=tuple(options.dmabuf_inputs), dmabuf_rest=options.dmabuf_rest,
        wipe_cache_mb=options.wipe_cache_mb,
        cut_latency_encoder=options.cut_latency_encoder,
        prewarm_cut_scenes=options.prewarm_cut_scenes,
    )


def _build_renditions(avp, api, options: GraphOptions, cfg, mixer_edge: str):
    """One encoder per rendition, all fed from the single composited program.

    The compositor renders once at the canvas rate; a rendition re-times and
    rescales that picture for its own target, so extra renditions cost an
    encode, not another composite.
    """
    edges = [mixer_edge]
    if len(cfg.renditions) > 1:
        edges = [f"program_rendition_{r.id}" for r in cfg.renditions]
        avp.addNode(api.Split({"name": "split_renditions", "src": mixer_edge, "dst": edges,
                               "group": OUTPUT_GROUP, "on_error": "panic"}))
    listener = None
    for rendition, edge in zip(cfg.renditions, edges):
        scaled = edge
        if (rendition.width, rendition.height) != (cfg.canvas_w, cfg.canvas_h):
            scaled = f"program_scaled_{rendition.id}"
            avp.addNode(api.FilterVideo({
                "name": f"scale_{rendition.id}", "src": edge, "dst": scaled,
                "graph": f"scale_cuda=w={rendition.width}:h={rendition.height}",
                "hwaccel": HWACCEL, "group": OUTPUT_GROUP,
            }))
        if rendition.target == "janus":
            listener = build_janus_output(
                avp, api, scaled,
                JanusVideoConfig(
                    host=options.janus_host,
                    video_port=rendition.port or options.janus_video_port,
                    payload_type=options.janus_video_pt, ssrc=options.janus_video_ssrc,
                    bitrate_kbps=rendition.bitrate_kbps,
                    rtcp_bind=options.janus_rtcp_bind, rtcp_port=options.janus_rtcp_port,
                ),
                fps=rendition.fps, fps_den=FPS_DEN, width=rendition.width, height=rendition.height,
                hwaccel=HWACCEL, group=OUTPUT_GROUP,
                profile=rendition.profile, preset=rendition.preset,
            )
        else:
            _build_record_output(avp, api, replace(options, output=rendition.target,
                                                   codec=rendition.codec, fps=rendition.fps),
                                 scaled, width=rendition.width, height=rendition.height)
    return listener


def _build_from_config(options: GraphOptions, cfg: "mixer_config.MixerConfig", api) -> MixerApplication:
    """Sources, wipes and scenes from a JSON document; one chain per source."""
    options = replace(options, fps=cfg.fps)   # the document owns the frame rate, outputs included
    avp = api.AVPlumber()
    if options.remote_control_port:
        avp.enableControlServer(options.remote_control_port)
    avp.executeCommandsFromString(f'hwaccel.init {{ "name": "{HWACCEL}", "type": "cuda" }}')
    browsers = [s for s in cfg.sources if s.kind == "browser"]
    if browsers:
        open_windows(options.dmabuf_rest, [{"id": s.id, "url": s.location, "width": s.width,
                                            "height": s.height, "fps": s.fps or cfg.fps} for s in browsers])
        wait_for_sockets([f"{options.dmabuf_socket_dir}/{s.id}.sock" for s in browsers],
                         options.preheat_timeout_sec)
    avp.edges.planCapacity("*", 4)

    mixer = api.MixerGraphBuilder(
        avp, name=MIXER_NAME, canvas=(cfg.canvas_w, cfg.canvas_h), fps=(cfg.fps, FPS_DEN),
        latency_ms=options.mixer_latency_ms, hwaccel=HWACCEL, enable_wipe=True,
        defer_initial_routes=True, defer_output=True,
        keyframe_node=JANUS_KEYFRAME_NODE if options.janus_output else None,
        cache_wipes_mb=options.wipe_cache_mb or None,
    )
    aliases = cfg.alias_counts
    input_edges: list[str] = []
    for index, source in enumerate(cfg.sources):
        group = _input_group(index)
        if source.kind == "browser":
            nodes, edge = dmabuf_cuda_input_nodes(
                api, prefix=f"input_{index}", socket=f"{options.dmabuf_socket_dir}/{source.id}.sock",
                width=source.width, height=source.height, fps=cfg.fps, drm_hwaccel=None,
                cuda_hwaccel=HWACCEL, source_group=group, processing_group=group, hold=True)
            for node in nodes:
                avp.addNode(node)
        else:
            edge = build_input(avp, api, str(index), source.location, group=group, fps=cfg.fps,
                               fps_den=FPS_DEN, hwaccel=HWACCEL, loop=source.loop)
        input_edges.append(edge)
        count = aliases[source.id]
        edges = [edge]
        if count > 1:
            # The same frames under several names: one fan-out, no second decoder.
            edges = [f"{edge}_alias{k}" for k in range(1, count + 1)]
            avp.addNode(api.OneToMany({
                "type": "one_to_many", "name": f"alias_{index}", "src": edge, "dst": edges,
                "outputs": (1 << count) - 1, "group": group,
            }))
        for k, alias_edge in enumerate(edges, start=1):
            mixer.add_source(mixer_config.alias_name(source.id, k), pre_otm_edge=alias_edge,
                             input_group=group, default_graph="")
    for scene in cfg.scenes:
        mixer.add_scene(scene.id, mixer_config.scene_layers(cfg, scene))
    mixer.set_initial_scene(cfg.initial_scene, slot="A")
    settings = json.dumps(cfg.settings(), separators=(",", ":")) + "\n"
    avp.registerControlCommand("mixer.settings", lambda _arg: settings, True)
    mixer_edge = mixer.build()
    if cfg.renditions:
        rtcp_feedback_listener = _build_renditions(avp, api, options, cfg, mixer_edge)
    else:
        rtcp_feedback_listener = _build_outputs(avp, api, options, mixer_edge,
                                                width=cfg.canvas_w, height=cfg.canvas_h)
    return MixerApplication(
        avp=avp, mixer=mixer,
        input_groups=tuple(_input_group(index) for index in range(len(cfg.sources))),
        input_edges=tuple(input_edges), routed_inputs=False,
        preheat_timeout_sec=options.preheat_timeout_sec,
        rtcp_feedback_listener=rtcp_feedback_listener,
        wipe_file=options.wipe_file, wipe_files=tuple(w.path for w in cfg.wipes),
        browser_windows=tuple(s.id for s in browsers), dmabuf_rest=options.dmabuf_rest,
        wipe_cache_mb=options.wipe_cache_mb,
        cut_latency_encoder=options.cut_latency_encoder,
        prewarm_cut_scenes=options.prewarm_cut_scenes,
    )


def parse_args(argv: list[str] | None = None) -> GraphOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        default=[],
        metavar="PATH",
        help="Input media file or URL; repeat for each mixer input",
    )
    parser.add_argument("--config", metavar="FILE",
                        help="JSON document with sources, wipes and scenes (replaces --input and the "
                             "built-in layouts; see doc/research/2026-09-08-mixer-config-schema.md)")
    parser.add_argument("--output", help="Optional video-only output URL or path")
    parser.add_argument("--dmabuf-socket-dir", default="/tmp/dma-page",
                        help="dma-browser socket directory for dmabuf://<window-id> inputs")
    parser.add_argument("--dmabuf-size", default="1280x720", metavar="WxH",
                        help="browser window size for dmabuf:// inputs")
    parser.add_argument("--dmabuf-open", metavar="URL",
                        help="open the dmabuf:// windows with this page through the dma-browser REST API")
    parser.add_argument("--dmabuf-rest", default="http://127.0.0.1:9009")
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
    parser.add_argument("--mixer-latency-ms", type=float, help="Native playout buffer (default: two output frames)")
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
    parser.add_argument("--wipe-file", help="Alpha wipe clip to warm the media-wipe chain up with at start "
                        "(the TUI still selects the clip for each wipe)")
    parser.add_argument("--wipe-cache-mb", type=float, default=768.0,
                        help="Hold decoded wipe clips in GPU memory, up to this many MiB "
                             "(0 decodes each wipe on every take)")
    parser.add_argument("--webui-url", default="",
                        help="Register the graph with an AVPlumber web UI, e.g. http://127.0.0.1:22222")
    parser.add_argument("--cut-latency-encoder", default="", metavar="NODE",
                        help="Measure CUT receipt to matching encoded frame at NODE (e.g. janus_encoder)")
    parser.add_argument("--prewarm-cut-scene", action="append", default=[], metavar="SCENE",
                        help="Keep source buffers warm for direct cuts (repeat; '*' selects all scenes)")
    args = parser.parse_args(argv)
    if not args.inputs and not args.config:
        parser.error("pass --input (repeatable) or --config FILE")
    return GraphOptions(
        inputs=tuple(args.inputs),
        output=args.output,
        output_format=args.output_format,
        remote_control_port=args.remote_control_port,
        codec=args.codec,
        bitrate=args.bitrate,
        fps=args.fps,
        mixer_latency_ms=args.mixer_latency_ms,
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
        wipe_file=args.wipe_file,
        dmabuf_socket_dir=args.dmabuf_socket_dir,
        dmabuf_size=parse_size(args.dmabuf_size),
        dmabuf_open=args.dmabuf_open,
        dmabuf_rest=args.dmabuf_rest,
        config=args.config,
        webui_url=args.webui_url,
        cut_latency_encoder=args.cut_latency_encoder,
        prewarm_cut_scenes=tuple(args.prewarm_cut_scene),
        wipe_cache_mb=args.wipe_cache_mb,
    )


def parse_size(text: str) -> tuple[int, int]:
    try:
        width, height = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise ValueError(f"expected WxH, got {text!r}") from None
    return width, height


def main(argv: list[str] | None = None) -> None:
    options = parse_args(argv)
    application = build_application(options)
    if options.webui_url:
        application.avp.registerWithWebUI(options.webui_url, "mixer", "")
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
            if options.webui_url:
                application.avp.heartbeat()
    except KeyboardInterrupt:
        pass
    finally:
        application.stop()


if __name__ == "__main__":
    main()
