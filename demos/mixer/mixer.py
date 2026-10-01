"""N-input, video-only manual mixer demonstration.

The graph exposes fullscreen and paged 2/4/8/16-box scenes through the generic
AVPlumber mixer control protocol. It contains no audio or automatic selection
path; use ``tui.py`` to preview and take scenes manually.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import logging
import sys
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace

from pyplumber.mixer.color import TEN_BIT_FORMATS, TRANSFER_TAGS, hdr_metadata, rendition_color
from pyplumber.mixer.backend import mixer_backend
from pyplumber.mixer import clipcache
from pyplumber.mixer import config as mixer_config
from pyplumber.mixer.config import FEEDS
from pyplumber.mixer.dsk import DownstreamKeyer, register_dsk_commands
from pyplumber.mixer.dmabuf_inputs import (dmabuf_cuda_input_nodes, is_dmabuf_url, open_browser_windows,
                                    open_windows, refresh_windows, wait_for_sockets, window_id)
from pyplumber.mixer.inputs import build_input, build_v210_input, build_raw420_input
from pyplumber.mixer.janus import (DEFAULT_KEYFRAME_MIN_INTERVAL_MS, JANUS_KEYFRAME_NODE,
                           JanusVideoConfig, RtcpFeedbackGroup, add_nodes, build_janus_output, dpb_options)

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


log = logging.getLogger("mixer")
DEFAULT_FPS = 30
FPS_DEN = 1
# Instance-owned: a global (@) device survives shutdown until CUDA static teardown.
HWACCEL = "mixer_gpu"
MIXER_NAME = "mixer"
ROUTER_GROUP = "mixer_preheat_router"
OUTPUT_GROUP = "output"
JANUS_DEFAULT_HOST = "127.0.0.1"
JANUS_DEFAULT_VIDEO_PORT = 5004
JANUS_DEFAULT_VIDEO_PT = 96
JANUS_DEFAULT_VIDEO_SSRC = 0x41565001
JANUS_DEFAULT_VIDEO_BITRATE_KBPS = 4_500
PREHEAT_POLL_INTERVAL_SEC = 0.02
# More --input sources than this are scaled to the canonical size and routed (see _register_sources).
DIRECT_INPUT_LIMIT = 32


@dataclass(frozen=True)
class GraphOptions:
    inputs: tuple[str, ...] = ()
    output: str | None = None
    output_format: str | None = None
    remote_control_port: int = 7777
    codec: str = "h264_nvenc"
    working_format: str = "nv12"   # compositor/transition sw_format; p210le keeps 10-bit 4:2:2
    bitrate: str = "8M"
    fps: int = DEFAULT_FPS
    mixer_latency_ms: float | None = None
    loop_inputs: bool = False
    input_color: str = ""          # declared contract for every --input (sdr/hlg/pq); "" = frame tags
    janus_output: bool = False
    janus_host: str = JANUS_DEFAULT_HOST
    janus_video_port: int = JANUS_DEFAULT_VIDEO_PORT
    janus_video_pt: int = JANUS_DEFAULT_VIDEO_PT
    janus_video_ssrc: int = JANUS_DEFAULT_VIDEO_SSRC
    janus_video_bitrate_kbps: int = JANUS_DEFAULT_VIDEO_BITRATE_KBPS
    keyframe_min_interval_ms: int = DEFAULT_KEYFRAME_MIN_INTERVAL_MS
    janus_rtcp_bind: str = "0.0.0.0"
    janus_rtcp_port: int = 0
    preheat_timeout_sec: float = 60.0
    wipe_file: str | None = None         # warm the media wipe chain up with this clip at start
    wipe_color: str = ""                 # explicit SDR override; empty preserves tags with an SDR fallback
    config: str | None = None            # JSON document (sources, wipes, scenes) instead of --input
    webui_url: str = ""                  # AVPlumber web UI to register the graph with
    cut_latency_encoder: str = ""        # opt-in cut-to-output observer on this encoder
    prewarm_cut_scenes: tuple[str, ...] = ()  # '*' selects all scene definitions
    wipe_cache_mb: float = 640.0        # GPU clip cache (the demo's two 2 s 540x960 wipes take ~0.5 GB); 0 decodes per take
    # Browser pages from the DMA-BUF demo as sources: --input dmabuf://<window-id>
    dmabuf_socket_dir: str = "/tmp/dma-page"
    dmabuf_size: tuple[int, int] = (1920, 1080)
    dmabuf_open: str | None = None       # page URL: open the named windows before building
    dmabuf_rest: str = "http://127.0.0.1:9009"
    browser_ring_size: int | None = None
    max_compositor_layers: int | None = None

    def __post_init__(self):
        if self.browser_ring_size is None:
            object.__setattr__(self, "browser_ring_size", mixer_config.default_browser_ring_size(self.fps))

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
        if self.working_format not in mixer_config.WORKING_FORMATS:
            raise ValueError(f"--working-format must be one of {mixer_config.WORKING_FORMATS}")
        if self.input_color and self.input_color not in TRANSFER_TAGS:
            raise ValueError("--input-color must be sdr, hlg or pq")
        if self.wipe_color not in ("", "sdr"):
            raise ValueError("--wipe-color supports sdr only; HDR alpha wipes are unsupported")
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
        if (type(self.keyframe_min_interval_ms) is not int
                or not 0 <= self.keyframe_min_interval_ms <= 2_147_483_647):
            raise ValueError("keyframe_min_interval_ms must be a non-negative integer")
        if any(v <= 0 for v in self.dmabuf_size):
            raise ValueError("--dmabuf-size must be WxH with positive numbers")
        if type(self.browser_ring_size) is not int or not 1 <= self.browser_ring_size <= 64:
            raise ValueError("--browser-ring-size must be an integer from 1 to 64")
        if self.max_compositor_layers is not None and (type(self.max_compositor_layers) is not int or not 1 <= self.max_compositor_layers <= 2_147_483_647):
            raise ValueError("--max-compositor-layers must be a positive 32-bit integer")
        ids = self.dmabuf_inputs
        if len(ids) != len(set(ids)):
            raise ValueError("dmabuf window ids must be unique")
        if ids and len(self.inputs) > DIRECT_INPUT_LIMIT:
            raise ValueError(f"dmabuf:// inputs need --config above {DIRECT_INPUT_LIMIT} --input sources: "
                             "scale_cuda cannot read zero-copy browser frames")
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
    aux_buses: tuple = ()
    wipe_cache_mb: float = 640.0            # hold decoded wipes in GPU memory
    cut_latency_encoder: str = ""
    prewarm_cut_scenes: tuple[str, ...] = ()

    def _preload_wipes(self) -> None:
        """Decode every wipe once into GPU memory (see pyplumber.mixer.clipcache).

        The loader group is started only here. The player group runs already
        (mixer.start_groups); a take arms it to replay what this left behind.
        """
        loader = clipcache.loader_group(MIXER_NAME)
        cache_node = f"{MIXER_NAME}_wipe_cache"
        self._wait_for_node(cache_node)   # its group starts asynchronously with the mixer's
        for clip in dict.fromkeys(c for c in (self.wipe_file, *self.wipe_files) if c):
            started = time.monotonic()
            # The reader needs the clip before its group starts, to open the file; the
            # running cache is told which clip the chain is about to deliver.
            value = json.dumps(clip)   # both commands parse the value as JSON
            self.avp.executeCommandsFromString(
                f"node.param.set {MIXER_NAME}_wipe_input url {value}\n"
                f"node.object.set {cache_node} load {value}")
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
        started = time.monotonic()
        for group in self.input_groups:
            self.avp.group(group).startNodes()
        if self.browser_windows:
            refresh_windows(self.dmabuf_rest, list(self.browser_windows))
        self._wait_for_edges(self.input_edges, "input readiness")
        log.info("Inputs ready: %d in %.1f s", len(self.input_edges), time.monotonic() - started)
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
        for bus in self.aux_buses:
            bus.start()
        log.info("Generic mixer preheat complete: compositors and transition ready in %.1f s",
                 time.monotonic() - started)

    def stop(self) -> None:
        started = time.monotonic()
        log.info("Stopping the graph")
        for bus in self.aux_buses:
            bus.stop()
        if self.rtcp_feedback_listener is not None:
            self.rtcp_feedback_listener.stop()
        # After a panic the graph is already shutting down under the manager lock and its groups
        # are stopping; group() would only wait for that. shutdown() below waits for it anyway.
        if self.avp.manager.shouldWork:
            # shutdown() stops one group after another, so a large show took minutes; stopNodes()
            # only signals the group's own thread. Asking every input and aux group first leaves
            # shutdown() joining groups that stop concurrently (its group order was never defined).
            for group in (*self.input_groups, *(bus.group for bus in self.aux_buses)):
                self.avp.group(group).stopNodes()
        self.avp.shutdown()
        log.info("Graph stopped in %.1f s", time.monotonic() - started)


def load_avp_api():
    from pyplumber import AVPlumber
    from pyplumber.mixer import MixerGraphBuilder
    from pyplumber.node import (
        AssumeVideoFormat,
        Bsf,
        CudaRectOverlay,
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
        RawToCuda,
        Realtime,
        RepeatLastFrame,
        SmoothTimestamps,
        Split,
        V210ToCuda,
    )
    from pyplumber.rtcp_feedback import RtcpFeedbackListener

    return SimpleNamespace(
        AVPlumber=AVPlumber,
        MixerGraphBuilder=MixerGraphBuilder,
        AssumeVideoFormat=AssumeVideoFormat,
        Bsf=Bsf,
        CudaRectOverlay=CudaRectOverlay,
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
        RawToCuda=RawToCuda,
        Realtime=Realtime,
        RepeatLastFrame=RepeatLastFrame,
        RtcpFeedbackListener=RtcpFeedbackListener,
        SmoothTimestamps=SmoothTimestamps,
        Split=Split,
        V210ToCuda=V210ToCuda,
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


# Event loops shared by the sources' pacing nodes (realtime, force_fps, smooth_timestamps).
# One loop's cost grows with the square of the sources it paces: each wake-up rescans
# every pending fd wait and every timer insert walks the sorted list. One loop at
# 100 sources used 43 % of a core, delaying realtime's timestamps and holding decoders;
# 4 loops cut that term 16x (about 33 sources each at 130) for three more mostly idle
# threads. Output, aux and wipe pacing keep the "default" loop to themselves.
PACING_LOOPS = 4


def _pacing_loop(index: int) -> str:
    return f"pacing_{index % PACING_LOOPS}"


def _init_avp(avp_options, api, on_error: ExitStack):
    """AVPlumber instance + control server + CUDA hwaccel — shared by both the
    --input and --config build paths. A failed or interrupted build shuts it down."""
    avp = api.AVPlumber()
    on_error.callback(avp.shutdown)
    if avp_options.remote_control_port:
        avp.enableControlServer(avp_options.remote_control_port)
    avp.executeCommandsFromString(f'hwaccel.init {{ "name": "{HWACCEL}", "type": "cuda" }}')
    # The queue rounds to 2**n - 1 slots: requesting 4 retains up to 7 frames.
    avp.edges.planCapacity("*", 3)
    return avp


def _make_builder(avp, api, options, *, canvas, fps, working_format, color="sdr", wipe_color=None):
    """The mixer builder, configured identically for both build paths (canvas,
    rate and working_format are the only per-path differences)."""
    return api.MixerGraphBuilder(
        avp, name=MIXER_NAME, canvas=canvas, fps=(fps, FPS_DEN),
        latency_ms=options.mixer_latency_ms, hwaccel=HWACCEL, enable_wipe=True,
        max_compositor_layers=options.max_compositor_layers or mixer_config.DEFAULT_MAX_COMPOSITOR_LAYERS,
        defer_initial_routes=True, defer_output=True,
        keyframe_node=JANUS_KEYFRAME_NODE if options.janus_output else None,
        cache_wipes_mb=options.wipe_cache_mb or None, working_format=working_format, color=color,
        wipe_color=options.wipe_color or wipe_color)


def _build_input(
    avp, api, index: int, url: str, *, loop: bool, fps: int, normalize: bool,
    options: "GraphOptions | None" = None,
) -> str:
    group = _input_group(index)
    if is_dmabuf_url(url):
        # A dma-browser window: DRM PRIME frames over its socket, already on the
        # shared monotonic clock; the same chain the DMA-BUF demo composes from.
        nodes, fps_edge = dmabuf_cuda_input_nodes(
            api, prefix=f"input_{index}",
            socket=f"{options.dmabuf_socket_dir}/{window_id(url)}.sock",
            fps=fps, drm_hwaccel=None, cuda_hwaccel=HWACCEL,
            source_group=group, processing_group=group, hold=True, browser_ring_size=options.browser_ring_size,
            event_loop=_pacing_loop(index))
        for node in nodes:
            avp.addNode(node)
    else:
        fps_edge = build_input(avp, api, str(index), url, group=group, fps=fps,
                               fps_den=FPS_DEN, hwaccel=HWACCEL, loop=loop, continuous_loop=True,
                               event_loop=_pacing_loop(index))
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
        "threads": mixer_backend().graph_threads,
        "hwaccel": HWACCEL,
        "auto_restart": "group",
        "group": group,
    }))
    return normalized_edge


def _register_sources(avp, api, mixer, input_edges: list[str], urls, *, fps: int, color: str = "") -> bool:
    # Keep the legacy --input routing threshold: larger catalogues use a small
    # router selecting the 16 visible positions, without per-layout branches.
    if len(input_edges) <= DIRECT_INPUT_LIMIT:
        for index, (edge, url) in enumerate(zip(input_edges, urls)):
            browser = is_dmabuf_url(url)   # packed RGB, always SDR; decoded files follow --input-color
            mixer.add_source(f"source_{index}", pre_otm_edge=edge, input_group=_input_group(index),
                             default_graph="", packed_rgb=browser, color="sdr" if browser else color or None)
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
            route_output_label_b=labels[2 * index + 1], default_graph="", color=color or None,
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


def _kbps(bitrate: str) -> int:
    """``--bitrate`` in FFmpeg notation (``8M``, ``6000k`` or bit/s) as kbit/s."""
    scale = {"k": 1, "K": 1, "M": 1000}.get(bitrate[-1])
    return int(float(bitrate[:-1]) * scale) if scale else int(bitrate) // 1000


def _flag_renditions(options: GraphOptions, width: int, height: int) -> tuple:
    """``--output`` / ``--janus-output`` as renditions: a ``program`` record file
    at the CLI codec and bitrate, and a ``janus`` stream whose codec follows depth."""
    record = mixer_config.Rendition("program", options.output or "", width, height, options.fps,
                                    _kbps(options.bitrate), options.codec, preset="p3")
    janus = mixer_config.Rendition("janus", "janus", width, height, options.fps, options.janus_video_bitrate_kbps)
    return tuple(r for r, wanted in ((record, options.output), (janus, options.janus_output)) if wanted)


def _hdr_metadata(r: "mixer_config.Rendition", target) -> dict | None:
    """HDR10 static metadata for a PQ output; nvenc emits the SEIs from it. HLG carries none."""
    return hdr_metadata(r.tonemap_peak * 100, max_cll=r.max_cll, max_fall=r.max_fall) if target.transfer == "pq" else None


def _build_record_output(avp, api, edge: str, r: "mixer_config.Rendition", *, codec: str,
                         enc_format: str, color: dict, output_format=None, hdr_metadata=None) -> None:
    """``force_fps -> assume_format -> nvenc -> mux -> output``, nodes named ``<id>_*``."""
    name = lambda suffix: f"{r.id}_{suffix}"  # noqa: E731
    bitrate = f"{r.bitrate_kbps}k"
    profile = r.profile or (("main10" if enc_format in TEN_BIT_FORMATS else "main") if "hevc" in codec else "high")
    add_nodes(avp, api, [
        ("ForceFPS", {"name": name("fps"), "src": edge, "dst": name("fps"), "fps": f"{r.fps}/{FPS_DEN}"}),
        ("AssumeVideoFormat", {"name": name("format"), "src": name("fps"), "dst": name("video"),
                               "width": r.width, "height": r.height, "pixel_format": "cuda",
                               "real_pixel_format": enc_format}),
        ("EncVideo", {"name": name("encoder"), "src": name("video"), "dst": name("encoded"),
                      "codec": codec, "hwaccel": HWACCEL,
                      **({"hdr_metadata": hdr_metadata} if hdr_metadata else {}),
                      "options": {"b": bitrate, "maxrate": bitrate, "bufsize": bitrate,
                                  "g": max(1, round(r.fps / FPS_DEN)) * 2, "bf": 0, "preset": r.preset,
                                  "tune": "ll", "profile": profile, **dpb_options(r.dpb_size), **color}}),
        ("Mux", {"name": name("mux"), "src": [name("encoded")], "dst": name("muxed"), "ts_sort_wait": 0}),
        ("Output", {"name": name("output"), "src": name("muxed"), "url": r.target,
                    "format": infer_output_format(r.target, output_format), "auto_restart": "panic"}),
    ], group=OUTPUT_GROUP)


def _rendition_target(r, working_format, color):
    """Encoder and color contract of a rendition, from the canvas when it declares neither."""
    codec = r.codec or ("hevc_nvenc" if working_format in TEN_BIT_FORMATS else "h264_nvenc")
    return codec, rendition_color(color, codec, r.color or None, r.tonemap)


def _build_renditions(avp, api, options: GraphOptions, renditions, feeds, *,
                      canvas, working_format: str, color="sdr", backend=None):
    """One encoder per rendition, all fed from the single composited program.

    The compositor renders once at the canvas rate; a rendition converts,
    re-times and rescales that picture for its own target, so extra renditions
    cost an encode, not another composite. *feeds* maps a rendition feed
    (dirty/clean) to its program edge; a plain edge serves every feed.
    """
    backend = mixer_backend(backend)
    if isinstance(feeds, str):
        feeds = dict.fromkeys(FEEDS, feeds)
    edges = {}
    for feed in FEEDS:
        group = [r for r in renditions if r.feed == feed]
        if len(group) == 1:
            edges[group[0].id] = feeds[feed]
        elif group:
            split = [f"program_rendition_{r.id}" for r in group]
            avp.addNode(api.Split({"name": "split_renditions" + ("" if feed == "dirty" else f"_{feed}"),
                                   "src": feeds[feed], "dst": split, "group": OUTPUT_GROUP, "on_error": "panic"}))
            edges.update((r.id, e) for r, e in zip(group, split))
    listeners = []
    for r in renditions:
        edge = edges[r.id]
        codec, target = _rendition_target(r, working_format, color)
        # 10-bit stays P010 for HEVC (Main10 carries depth and HDR); H.264 and 8-bit encode NV12.
        ten_bit = target.transfer != "sdr" or (working_format in TEN_BIT_FORMATS and "hevc" in codec)
        enc_format = "p010le" if ten_bit else "nv12"
        scale = backend.scale(width=r.width, height=r.height) + "," if (r.width, r.height) != canvas else ""
        scaled = f"program_scaled_{r.id}"
        avp.addNode(api.FilterVideo({
            "name": f"scale_{r.id}", "src": edge, "dst": scaled, "hwaccel": HWACCEL, "group": OUTPUT_GROUP,
            "threads": backend.graph_threads,
            "graph": scale + backend.conversion(target, enc_format, source=color, source_format=working_format,
                                              tonemap=r.tonemap or "clip", hdr_peak=r.tonemap_peak * 100,
                                              desat=r.tonemap_desat, param=r.tonemap_param),
        }))
        if r.target != "janus":
            _build_record_output(avp, api, scaled, r, codec=codec, enc_format=enc_format,
                                 color=target.tags, output_format=options.output_format,
                                 hdr_metadata=_hdr_metadata(r, target))
            continue
        listeners.append(build_janus_output(
            avp, api, scaled,
            JanusVideoConfig(
                host=options.janus_host, video_port=r.port or options.janus_video_port,
                payload_type=options.janus_video_pt, ssrc=options.janus_video_ssrc,
                bitrate_kbps=r.bitrate_kbps, keyframe_min_interval_ms=options.keyframe_min_interval_ms,
                rtcp_bind=options.janus_rtcp_bind, rtcp_port=options.janus_rtcp_port if not listeners else 0,
            ),
            fps=r.fps, fps_den=FPS_DEN, width=r.width, height=r.height, hwaccel=HWACCEL, group=OUTPUT_GROUP,
            codec=codec, profile=r.profile, preset=r.preset, enc_format=enc_format, color=target.tags,
            hdr_metadata=_hdr_metadata(r, target), prefix="janus" if not listeners else f"janus_{r.id}",
            dpb_size=r.dpb_size))
    return RtcpFeedbackGroup(listeners) if len(listeners) > 1 else next(iter(listeners), None)


def _application(avp, mixer, options: GraphOptions, input_edges, listener, **extra) -> MixerApplication:
    return MixerApplication(
        avp=avp, mixer=mixer, input_edges=tuple(input_edges),
        input_groups=tuple(_input_group(index) for index in range(len(input_edges))),
        rtcp_feedback_listener=listener, preheat_timeout_sec=options.preheat_timeout_sec,
        wipe_file=options.wipe_file, dmabuf_rest=options.dmabuf_rest, wipe_cache_mb=options.wipe_cache_mb,
        cut_latency_encoder=options.cut_latency_encoder, prewarm_cut_scenes=options.prewarm_cut_scenes, **extra)


def build_application(options: GraphOptions, api=None) -> MixerApplication:
    options.validate()
    api = api or load_avp_api()
    # Native threads left running by an exception or Ctrl-C mid-build hang interpreter exit.
    with ExitStack() as on_error:
        application = _build(options, api, on_error)
        on_error.pop_all()
    return application


def _build(options: GraphOptions, api, on_error: ExitStack) -> MixerApplication:
    if options.config:
        return _build_from_config(options, mixer_config.with_probed_sizes(mixer_config.load(
            options.config, max_compositor_layers=options.max_compositor_layers)), api, on_error)
    dmabuf_ids = options.dmabuf_inputs
    if dmabuf_ids:
        if options.dmabuf_open:
            width, height = options.dmabuf_size
            open_browser_windows(options.dmabuf_rest, dmabuf_ids, options.dmabuf_open,
                                 width, height, options.fps, options.browser_ring_size)
        wait_for_sockets([f"{options.dmabuf_socket_dir}/{name}.sock" for name in dmabuf_ids],
                         options.preheat_timeout_sec)

    avp = _init_avp(options, api, on_error)
    input_edges = [
        _build_input(avp, api, index, url, loop=options.loop_inputs, fps=options.fps,
                     normalize=len(options.inputs) > DIRECT_INPUT_LIMIT, options=options)
        for index, url in enumerate(options.inputs)
    ]
    canvas = (CANVAS_WIDTH, CANVAS_HEIGHT)
    mixer = _make_builder(avp, api, options, canvas=canvas, fps=options.fps, working_format=options.working_format)
    routed_inputs = _register_sources(avp, api, mixer, input_edges, options.inputs, fps=options.fps,
                                      color=options.input_color)
    _define_scenes(mixer, len(input_edges), routed_inputs)
    mixer.set_initial_scene("fullscreen_0", slot="A")
    listener = _build_renditions(avp, api, options, _flag_renditions(options, *canvas), mixer.build(),
                                 canvas=canvas, working_format=options.working_format, backend=mixer.backend)
    return _application(avp, mixer, options, input_edges, listener, routed_inputs=routed_inputs,
                        browser_windows=tuple(options.dmabuf_inputs))


def _build_from_config(options: GraphOptions, cfg: "mixer_config.MixerConfig", api,
                       on_error: ExitStack) -> MixerApplication:
    """Sources, wipes and scenes from a JSON document; one chain per source."""
    options = replace(options, fps=cfg.fps, max_compositor_layers=cfg.max_compositor_layers)
    if options.mixer_latency_ms is None and cfg.latency_ms is not None:
        options = replace(options, mixer_latency_ms=cfg.latency_ms)
    browsers = [s for s in cfg.sources if s.kind == "browser"]
    if browsers:
        open_windows(options.dmabuf_rest, [{"id": s.id, "url": s.location, "width": s.width,
                                            "height": s.height, "fps": s.fps or cfg.fps,
                                            "ringSize": cfg.browser_ring_size,
                                            "holdLastFrame": s.hold_last_frame} for s in browsers])
        wait_for_sockets([f"{options.dmabuf_socket_dir}/{s.id}.sock" for s in browsers],
                         options.preheat_timeout_sec)

    # Browser REST failures must occur before native control threads exist;
    # otherwise Python can print a traceback yet hang during interpreter exit.
    avp = _init_avp(options, api, on_error)
    canvas = (cfg.canvas_w, cfg.canvas_h)
    mixer = _make_builder(avp, api, options, canvas=canvas, fps=cfg.fps, working_format=cfg.working_format,
                          color=cfg.out_color, wipe_color=cfg.wipe_color or None)
    aliases = cfg.alias_counts
    blended_sources = {item.source for scene in cfg.scenes for item in scene.items if item.blend}
    blended_sources |= {key.source for key in cfg.dsk_keys}
    input_edges: list[str] = []
    for index, source in enumerate(cfg.sources):
        group = _input_group(index)
        if source.kind == "browser":
            nodes, edge = dmabuf_cuda_input_nodes(
                api, prefix=f"input_{index}", socket=f"{options.dmabuf_socket_dir}/{source.id}.sock",
                fps=cfg.fps, drm_hwaccel=None,
                cuda_hwaccel=HWACCEL, source_group=group, processing_group=group, hold=True,
                preserve_alpha=source.id in blended_sources, browser_ring_size=cfg.browser_ring_size,
                event_loop=_pacing_loop(index))
            for node in nodes:
                avp.addNode(node)
        elif source.kind == "v210":
            # True 10-bit 4:2:2 sources: packed v210 unpacked to P210 on the GPU
            # and stamped with their declared color contract. NVDEC only yields
            # 4:2:0, so this is the one path that keeps 4:2:2 through the canvas.
            edge = build_v210_input(
                avp, api, str(index), source.location, width=source.width, height=source.height,
                group=group, fps=cfg.fps, fps_den=FPS_DEN, hwaccel=HWACCEL, loop=source.loop,
                color=source.color.tags, event_loop=_pacing_loop(index))
        elif source.kind in ("nv12", "p010"):
            edge = build_raw420_input(
                avp, api, str(index), source.location, width=source.width, height=source.height,
                pixel_format="p010le" if source.kind == "p010" else "nv12",
                group=group, fps=cfg.fps, fps_den=FPS_DEN, hwaccel=HWACCEL, loop=source.loop,
                event_loop=_pacing_loop(index), pinned=cfg.raw_upload == "pinned")
        else:
            edge = build_input(avp, api, str(index), source.location, group=group, fps=cfg.fps,
                               fps_den=FPS_DEN, hwaccel=HWACCEL, loop=source.loop, continuous_loop=True,
                               event_loop=_pacing_loop(index))
        if source.filter_graph:
            filtered_edge = f"input_{index}_filtered"
            avp.addNode(api.FilterVideo({
                "name": f"source_filter_{index}", "src": edge, "dst": filtered_edge,
                "graph": (source.color.setparams + "," if source.color else "") + source.filter_graph, "hwaccel": HWACCEL,
                "group": group, "auto_restart": "group",
            }))
            edge = filtered_edge
        input_edges.append(edge)
        for k in range(1, aliases[source.id] + 1):
            # The reusable builder shares conversion and fan-out for identical edges.
            mixer.add_source(mixer_config.alias_name(source.id, k), pre_otm_edge=edge,
                             input_group=group, default_graph="",
                             color=None if source.filter_graph else source.color,
                             packed_rgb=source.kind == "browser",
                             premultiplied_alpha=source.kind == "browser" and source.id in blended_sources,
                             pixel_format=source.filter_output_format or
                             {"v210": "p210le", "nv12": "nv12", "p010": "p010le"}.get(source.kind))
    for scene in cfg.scenes:
        mixer.add_scene(scene.id, mixer_config.scene_layers(cfg, scene))
    mixer.set_initial_scene(cfg.initial_scene, slot="A")
    from pyplumber.mixer.aux import make_aux, register_aux_commands
    aux = tuple(make_aux(avp, api, mixer, cfg, bus) for bus in cfg.aux_buses)
    keyer = DownstreamKeyer(avp, api, mixer, cfg, group=OUTPUT_GROUP) if cfg.dsk_keys else None
    renditions = cfg.renditions or _flag_renditions(options, *canvas)
    program = mixer.build()
    feeds = dict.fromkeys(FEEDS, program)
    if keyer:
        feeds = {**feeds, **keyer.build(program, clean=any(r.feed == "clean" for r in renditions))}
        register_dsk_commands(avp, keyer)
    pgm_taps = [b.pgm_edge for b in aux if b.pgm_edge]
    if pgm_taps:
        # The multiview PGM tile shows what goes to air: the keyed program.
        tapped = "program_after_aux_tap"
        avp.addNode(api.OneToMany({
            "name": "program_aux_tap", "src": feeds["dirty"], "dst": [tapped, *pgm_taps],
            "outputs": 1, "subscribed_outputs": {edge: edge for edge in pgm_taps}, "group": OUTPUT_GROUP,
        }))
        feeds["dirty"] = tapped
    if aux:
        for bus in aux:
            bus.build(options)
        register_aux_commands(avp, aux)
    settings_data = cfg.settings()
    preview_outputs = []
    if keyer:
        for r in renditions:
            if r.feed != "clean" or r.target != "janus":
                continue
            codec, target = _rendition_target(r, cfg.working_format, cfg.out_color)
            preview_outputs.append({"bus": f"clean_{r.id}", "label": f"Program clean · {'SDR' if target.transfer == 'sdr' else 'HDR'}",
                                    "rendition": r.id, "codec": "h265" if "hevc" in codec else "h264",
                                    "color": "sdr" if target.transfer == "sdr" else "hdr",
                                    "port": r.port, "mountpoint": r.port, "fps": r.fps})
    if aux:
        settings_data["aux_buses"] = [b.bus.id for b in aux]
        labels = {"pgm_pvw_grid": "Program preview", "source_pages": "Multiviewer"}
        preview_outputs += [
            {"bus": b.bus.id, "label": labels[b.bus.layout], "layout": b.bus.layout, "rendition": r.id,
             "codec": "h264", "color": "sdr", "port": r.port, "mountpoint": r.port, "fps": r.fps}
            for b in aux for r in b.bus.renditions]
    if preview_outputs:
        settings_data["preview_outputs"] = preview_outputs
    settings = json.dumps(settings_data, separators=(",", ":")) + "\n"
    avp.registerControlCommand("mixer.settings", lambda _arg: settings, True)
    listener = _build_renditions(avp, api, options, renditions, feeds, canvas=canvas,
                                 working_format=cfg.working_format, color=cfg.out_color, backend=mixer.backend)
    return _application(avp, mixer, options, input_edges, listener, routed_inputs=False,
                        wipe_files=tuple(w.path for w in cfg.wipes), browser_windows=tuple(s.id for s in browsers), aux_buses=aux)


def parse_args(argv: list[str] | None = None) -> GraphOptions:
    """Every option's ``dest`` is a GraphOptions field, so the parsed namespace
    maps onto the dataclass directly; an unmapped option fails loudly instead
    of being silently dropped."""
    p = argparse.ArgumentParser(description=__doc__)
    add = p.add_argument
    add("--input", dest="inputs", action="append", default=[], metavar="PATH",
        help="Input media file or URL; repeat for each mixer input")
    add("--config", metavar="FILE",
        help="JSON document with sources, wipes and scenes (replaces --input and the built-in "
             "layouts; see doc/research/2026-09-08-mixer-config-schema.md)")
    add("--output", help="Optional video-only output URL or path")
    add("--output-format", help="Muxer format when it cannot be inferred from the output")
    add("--codec", default="h264_nvenc")
    add("--bitrate", default="8M")
    add("--fps", type=int, default=DEFAULT_FPS, help=f"Mixer and output frame rate (default: {DEFAULT_FPS})")
    add("--wipe-color", default="", choices=("sdr",),
        help="Override wipe color tags as SDR (default: preserve tags, assume SDR for missing tags)")
    add("--input-color", default="", choices=("", "sdr", "hlg", "pq"),
        help="color contract declared for every --input file (default: trust the decoded frame tags)")
    add("--working-format", default="nv12",
        help="compositor/transition sw_format; p210le keeps 10-bit 4:2:2 on the canvas "
             "(renditions subsample to P010/NV12 for NVENC automatically)")
    add("--mixer-latency-ms", type=float, help="Native playout buffer (default: two output frames)")
    add("--max-compositor-layers", type=int, help="Layer budget per compositor; overrides JSON max_compositor_layers (default: 256)")
    add("--loop-inputs", action="store_true")
    add("--remote-control-port", type=int, default=7777)
    add("--dmabuf-socket-dir", default="/tmp/dma-page",
        help="dma-browser socket directory for dmabuf://<window-id> inputs")
    add("--dmabuf-size", default="1920x1080", metavar="WxH",
        help="browser window size for dmabuf:// inputs")
    add("--dmabuf-open", metavar="URL",
        help="open the dmabuf:// windows with this page through the dma-browser REST API")
    add("--dmabuf-rest", default="http://127.0.0.1:9009")
    add("--browser-ring-size", type=int,
        help="maximum outstanding DMA-BUF frames per browser (default 6 at 25/30 fps, 9 otherwise; JSON config owns this in --config mode)")
    add("--janus-output", action="store_true", help="Publish the video-only program to Janus over RTP")
    add("--janus-host", default=JANUS_DEFAULT_HOST)
    add("--janus-video-port", type=int, default=JANUS_DEFAULT_VIDEO_PORT)
    add("--janus-video-pt", type=int, default=JANUS_DEFAULT_VIDEO_PT)
    add("--janus-video-ssrc", type=lambda v: int(v, 0), default=JANUS_DEFAULT_VIDEO_SSRC)
    add("--janus-video-bitrate-kbps", type=int, default=JANUS_DEFAULT_VIDEO_BITRATE_KBPS)
    add("--janus-rtcp-bind", default="0.0.0.0")
    add("--janus-rtcp-port", type=int, default=0)
    add("--keyframe-min-interval-ms", type=int, default=DEFAULT_KEYFRAME_MIN_INTERVAL_MS,
        help="Minimum forced-keyframe spacing for Janus output in media time "
             "(default: 150 ms; 0 disables rate limiting)")
    add("--preheat-timeout", dest="preheat_timeout_sec", type=float, default=60.0)
    add("--wipe-file", help="Alpha wipe clip to warm the media-wipe chain up with at start "
                            "(the TUI still selects the clip for each wipe)")
    add("--wipe-cache-mb", type=float, default=GraphOptions.wipe_cache_mb,
        help="GPU wipe cache budget in MiB (default: 640; 0 decodes on each take)")
    add("--webui-url", default="", help="Register the graph with an AVPlumber web UI, e.g. http://127.0.0.1:22222")
    add("--cut-latency-encoder", default="", metavar="NODE",
        help="Measure CUT receipt to matching encoded frame at NODE (e.g. janus_encoder)")
    add("--prewarm-cut-scene", dest="prewarm_cut_scenes", action="append", default=[], metavar="SCENE",
        help="Keep source buffers warm for direct cuts (repeat; '*' selects all scenes)")
    args = vars(p.parse_args(argv))
    if not args["inputs"] and not args["config"]:
        p.error("pass --input (repeatable) or --config FILE")
    args["inputs"] = tuple(args["inputs"])
    args["prewarm_cut_scenes"] = tuple(args["prewarm_cut_scenes"])
    args["dmabuf_size"] = parse_size(args["dmabuf_size"])   # ValueError on bad WxH, not a usage exit
    return GraphOptions(**args)

def parse_size(text: str) -> tuple[int, int]:
    try:
        width, height = (int(v) for v in text.lower().split("x"))
    except ValueError:
        raise ValueError(f"expected WxH, got {text!r}") from None
    return width, height


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    options = parse_args(argv)
    started = time.monotonic()
    try:
        application = build_application(options)
    except KeyboardInterrupt:
        return   # the build shut down what it had started
    log.info("Graph built in %.1f s", time.monotonic() - started)
    try:
        _run_application(application, options)
    except KeyboardInterrupt:
        return
    finally:
        application.stop()
    # Nothing is on air any more; a supervisor (setup_runtime) restarts the mixer on this exit.
    sys.exit("Mixer graph shut down after a node failure (auto_restart panic); exiting")


def _run_application(application: MixerApplication, options: GraphOptions) -> None:
    """Run until interrupted; return when a node panic has shut the graph down."""
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
    while application.avp.manager.shouldWork:
        time.sleep(1)
        if options.webui_url:
            application.avp.heartbeat()


if __name__ == "__main__":
    main()
