"""Config-driven mixer application, output assembly and startup/shutdown.

Importing this module does not load the native engine. Applications supply show
configuration and optionally extend each physical input before the mixer fans it
out to scenes and auxiliary buses.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field, replace
import json
import logging
import time
from types import SimpleNamespace
from typing import Callable

from . import clipcache
from . import config as mixer_config
from .backend import mixer_backend
from .color import TEN_BIT_FORMATS, default_codec, hdr_metadata, rendition_color, rendition_format
from .config import FEEDS
from .dsk import DownstreamKeyer, register_dsk_commands
from .dmabuf_inputs import dmabuf_cuda_input_nodes, open_windows, refresh_windows, wait_for_sockets
from .inputs import build_input, build_v210_input, build_raw420_input
from .janus import (DEFAULT_KEYFRAME_MIN_INTERVAL_MS, JANUS_KEYFRAME_NODE, JanusVideoConfig,
                    RtcpFeedbackGroup, add_nodes, build_janus_output, dpb_options)
from ..transform import transform_output, transform_params


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

# Event loops shared by the sources' pacing nodes (realtime, force_fps, smooth_timestamps).
# One loop's cost grows with the square of the sources it paces: each wake-up rescans
# every pending fd wait and every timer insert walks the sorted list. One loop at
# 100 sources used 43 % of a core, delaying realtime's timestamps and holding decoders;
# 4 loops cut that term 16x (about 33 sources each at 130) for three more mostly idle
# threads. Output, aux and wipe pacing keep the "default" loop to themselves.
PACING_LOOPS = 4


@dataclass(frozen=True)
class MixerOptions:
    """Runtime and fallback output settings; the show config owns sources and scenes."""
    output: str | None = None
    output_format: str | None = None
    remote_control_port: int = 7777
    codec: str = "h264_nvenc"
    working_format: str = "nv12"   # compositor/transition sw_format; p210le keeps 10-bit 4:2:2
    bitrate: str = "8M"
    fps: int = DEFAULT_FPS
    mixer_latency_ms: float | None = None
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
    cut_latency_encoder: str = ""        # opt-in cut-to-output observer on this encoder
    prewarm_cut_scenes: tuple[str, ...] = ()  # '*' selects all scene definitions
    wipe_cache_mb: float = 640.0        # GPU clip cache (two 2 s 540x960 wipes take ~0.5 GB); 0 decodes per take
    dmabuf_socket_dir: str = "/tmp/dma-page"
    dmabuf_rest: str = "http://127.0.0.1:9009"
    browser_ring_size: int | None = None
    max_compositor_layers: int | None = None

    def __post_init__(self):
        if self.browser_ring_size is None:
            object.__setattr__(self, "browser_ring_size", mixer_config.default_browser_ring_size(self.fps))

    def validate(self) -> None:
        if self.working_format not in mixer_config.WORKING_FORMATS:
            raise ValueError(f"--working-format must be one of {mixer_config.WORKING_FORMATS}")
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
        if type(self.browser_ring_size) is not int or not 1 <= self.browser_ring_size <= 64:
            raise ValueError("--browser-ring-size must be an integer from 1 to 64")
        if self.max_compositor_layers is not None and (type(self.max_compositor_layers) is not int or not 1 <= self.max_compositor_layers <= 2_147_483_647):
            raise ValueError("--max-compositor-layers must be a positive 32-bit integer")
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


@dataclass(frozen=True)
class SourceContext:
    """A physical input after its configured transform/filter, before fan-out.

    Add processing nodes to ``group`` and return their video edge (or ``edge``
    for an unchanged input). Preserve geometry, storage/color contracts and
    timestamps. Branches that consume video must explicitly split the edge;
    queues do not broadcast to multiple readers.
    """
    avp: object
    api: object
    source: mixer_config.Source
    index: int
    edge: str
    group: str
    hwaccel: str
    fps: int


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
    aux_buses: object = None                # pyplumber.mixer.aux.AuxBuses, with a config that has buses
    wipe_cache_mb: float = 640.0            # hold decoded wipes in GPU memory
    cut_latency_encoder: str = ""
    prewarm_cut_scenes: tuple[str, ...] = ()
    _startup_error: str | None = field(default=None, init=False, repr=False)

    def _check_startup(self) -> None:
        if self._startup_error:
            raise RuntimeError(self._startup_error)

    def _preload_wipes(self) -> None:
        """Decode every wipe once into GPU memory (see pyplumber.mixer.clipcache).

        The loader group is started only here. The player group runs already
        (mixer.start_groups); a take arms it to replay what this left behind.
        A clip that is not cached whole, or that a later clip evicted, fails the start.
        """
        self._wait_for_node(f"{MIXER_NAME}_{clipcache.CACHE_NODE}")   # its group starts asynchronously with the mixer's
        clips = tuple(dict.fromkeys(c for c in (self.wipe_file, *self.wipe_files) if c))
        for clip in clips:
            started = time.monotonic()
            held = clipcache.preload(self.avp, MIXER_NAME, clip, timeout_sec=self.preheat_timeout_sec,
                                     poll_sec=PREHEAT_POLL_INTERVAL_SEC, check=self._check_startup)
            print(f"wipe cached: {clip} {held['frames']} frames, {held['bytes'] / 1048576:.1f} MiB, "
                  f"{(time.monotonic() - started) * 1000:.0f} ms", flush=True)
        clipcache.require_cached(self.avp, MIXER_NAME, clips)

    def _wait_for_edges(self, edges: tuple[str, ...], phase: str, data_type: str | None = None) -> None:
        deadline = time.monotonic() + self.preheat_timeout_sec
        while True:
            self._check_startup()
            missing = [edge for edge in edges if self.avp.getEdge(edge, data_type).enqueued_total == 0]
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
            self._check_startup()
            if time.monotonic() >= deadline:
                raise RuntimeError(f"mixer preheat timed out starting node {name}")
            time.sleep(PREHEAT_POLL_INTERVAL_SEC)

    def start(self) -> None:
        # Control requests wait for setReady(), so Setup cannot inspect failed nodes
        # while preheating. Abort here on the native error instead of waiting for frames.
        original = self.avp.on_exception
        self._startup_error = None
        def failed(name, kind, message):
            if self._startup_error is None:
                self._startup_error = f"Mixer startup failed at {name} ({kind}): {message}"
            original(name, kind, message)
        self.avp.on_exception = failed
        try:
            self._start()
        finally:
            self.avp.on_exception = original

    def _start(self) -> None:
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
        if self.aux_buses:
            self.aux_buses.start()
            self._wait_for_edges(tuple(f"{bus.prefix}_encoded" for bus in self.aux_buses), "auxiliary output", "packet")
        self._check_startup()
        self.avp.setReady()
        log.info("Generic mixer preheat complete: compositors and transition ready in %.1f s",
                 time.monotonic() - started)

    def stop(self) -> None:
        started = time.monotonic()
        log.info("Stopping the graph")
        if self.aux_buses:
            self.aux_buses.stop()
        if self.rtcp_feedback_listener is not None:
            self.rtcp_feedback_listener.stop()
        # After a panic the graph is already shutting down under the manager lock and its groups
        # are stopping; group() would only wait for that. shutdown() below waits for it anyway.
        if self.avp.manager.shouldWork:
            # shutdown() stops one group after another, so a large show took minutes; stopNodes()
            # only signals the group's own thread. Asking every input and aux group first leaves
            # shutdown() joining groups that stop concurrently (its group order was never defined).
            for group in (*self.input_groups, *(bus.group for bus in self.aux_buses or ())):
                self.avp.group(group).stopNodes()
        self.avp.shutdown()
        log.info("Graph stopped in %.1f s", time.monotonic() - started)


def load_avp_api():
    from pyplumber import AVPlumber
    from pyplumber.mixer import MixerGraphBuilder
    from pyplumber.node import (
        AssumeVideoFormat,
        Bsf,
        CudaTransform,
        DecVideo,
        Demux,
        DrmPrimeToCuda,
        EncVideo,
        FilterVideo,
        ForceFPS,
        ForceKeyFrame,
        InputRec,
        IpcDmabufSource,
        MixerCompositor,
        MixerKeyer,
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
        CudaTransform=CudaTransform,
        DecVideo=DecVideo,
        Demux=Demux,
        DrmPrimeToCuda=DrmPrimeToCuda,
        EncVideo=EncVideo,
        FilterVideo=FilterVideo,
        ForceFPS=ForceFPS,
        ForceKeyFrame=ForceKeyFrame,
        InputRec=InputRec,
        IpcDmabufSource=IpcDmabufSource,
        MixerCompositor=MixerCompositor,
        MixerKeyer=MixerKeyer,
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


def _kbps(bitrate: str) -> int:
    """``--bitrate`` in FFmpeg notation (``8M``, ``6000k`` or bit/s) as kbit/s."""
    scale = {"k": 1, "K": 1, "M": 1000}.get(bitrate[-1])
    return int(float(bitrate[:-1]) * scale) if scale else int(bitrate) // 1000


def _flag_renditions(options: MixerOptions, width: int, height: int) -> tuple:
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
    codec = r.codec or default_codec(working_format)
    return codec, rendition_color(color, codec, r.color or None, r.tonemap)


def _build_renditions(avp, api, options: MixerOptions, renditions, feeds, *,
                      canvas, working_format: str, color="sdr", backend=None):
    """One encoder per rendition, all fed from the single composited program.

    The compositor renders once at the canvas rate, so an extra rendition costs an
    encode, not another composite. Per feed, the renditions whose size differs from
    the canvas are scaled by one cuda_transform, one draw per distinct size. Each
    rendition then converts color and storage in its own filter, after the scaling,
    so a smaller output converts fewer pixels; force_fps in its encoder chain makes
    the constant rate. *feeds* maps a rendition feed (dirty/clean) to its program
    edge; a plain edge serves every feed.
    """
    backend = mixer_backend(backend)
    if isinstance(feeds, str):
        feeds = dict.fromkeys(FEEDS, feeds)
    resized = [r for r in renditions if (r.width, r.height) != canvas]
    edges = {r.id: f"program_sized_{r.id}" for r in resized}
    for feed in FEEDS:
        suffix = "" if feed == "dirty" else f"_{feed}"
        group = [r for r in renditions if r.feed == feed]
        direct = [r for r in group if r not in resized]
        sized = {}   # (width, height) -> edges of the renditions of that size
        for r in group:
            if r in resized:
                sized.setdefault((r.width, r.height), []).append(edges[r.id])
        # The feed's readers: each canvas-size rendition, and one transform for all the others.
        readers = [f"program_rendition_{r.id}" for r in direct] + (["program_transform" + suffix] if sized else [])
        if len(readers) == 1:
            readers = [feeds[feed]]
        elif readers:
            avp.addNode(api.Split({"name": "split_renditions" + suffix, "src": feeds[feed], "dst": readers,
                                   "group": OUTPUT_GROUP, "on_error": "panic"}))
        edges.update((r.id, edge) for r, edge in zip(direct, readers))
        if sized:
            # No per-output fps: one draw then serves renditions of any rate, and each
            # rendition's force_fps picks and repeats frames as its encoder needs.
            avp.addNode(api.CudaTransform(transform_params(
                readers[-1], [transform_output(dst, width, height, sw_format=working_format)
                              for (width, height), dst in sized.items()],
                hwaccel=HWACCEL, name="transform_renditions" + suffix, group=OUTPUT_GROUP, on_error="panic")))
    listeners = []
    for r in renditions:
        codec, target = _rendition_target(r, working_format, color)
        enc_format = rendition_format(working_format, codec, target, r.profile)
        scaled = f"program_scaled_{r.id}"
        # Named scale_<id> for the size it used to change; it converts color and storage only.
        avp.addNode(api.FilterVideo({
            "name": f"scale_{r.id}", "src": edges[r.id], "dst": scaled, "hwaccel": HWACCEL, "group": OUTPUT_GROUP,
            "threads": backend.graph_threads,
            "graph": backend.conversion(target, enc_format, source=color, source_format=working_format,
                                        tonemap=r.tonemap or "clip", hdr_peak=r.tonemap_peak * 100,
                                        desat=r.tonemap_desat, param=r.tonemap_param),
            # Below a transform the graph is built from the first frame. A transform with
            # several sizes declares no format to build it from earlier, and a CUDA graph built
            # early is built again when the first frame brings its frame pool.
            **({"defer_preliminary_init": True} if r in resized else {}),
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


def _application(avp, mixer, options: MixerOptions, input_edges, listener, **extra) -> MixerApplication:
    return MixerApplication(
        avp=avp, mixer=mixer, input_edges=tuple(input_edges),
        input_groups=tuple(_input_group(index) for index in range(len(input_edges))),
        rtcp_feedback_listener=listener, preheat_timeout_sec=options.preheat_timeout_sec,
        wipe_file=options.wipe_file, dmabuf_rest=options.dmabuf_rest, wipe_cache_mb=options.wipe_cache_mb,
        cut_latency_encoder=options.cut_latency_encoder, prewarm_cut_scenes=options.prewarm_cut_scenes, **extra)


def build_application(cfg: mixer_config.MixerConfig, options: MixerOptions | None = None, *,
                      process_source: Callable[[SourceContext], str] | None = None,
                      api=None) -> MixerApplication:
    """Build a show without starting it; the caller owns ``start()`` / ``stop()``.

    ``process_source`` runs once per physical source, including sources not in
    the initial scene. Its returned edge is shared by every scene alias and AUX
    view. Build failures, including callback exceptions, shut down the engine.
    """
    options = options or MixerOptions()
    # Config-owned renditions need no redundant CLI-style output flag. Also
    # enable keyframe control and validate Janus settings for those renditions.
    options = replace(options, fps=cfg.fps, working_format=cfg.working_format,
                      janus_output=any(r.target == "janus" for r in cfg.renditions)
                      if cfg.renditions else options.janus_output)
    MixerOptions.validate(options)
    if not cfg.renditions and not options.output and not options.janus_output:
        raise ValueError("at least one rendition or output is required")
    cfg = mixer_config.with_probed_sizes(cfg)
    api = api or load_avp_api()
    with ExitStack() as on_error:
        application = _build_from_config(options, cfg, api, on_error, process_source=process_source)
        on_error.pop_all()
    return application


def _build_from_config(options: MixerOptions, cfg: "mixer_config.MixerConfig", api,
                       on_error: ExitStack, *, process_source=None) -> MixerApplication:
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
                               event_loop=_pacing_loop(index), decoder_params=source.decoder_params)
        if source.transform:
            transformed_edge = f"input_{index}_transformed"
            # pass_arrays: the color stage and the compositors read arrays, so a frame that already
            # has the size goes on undrawn.
            avp.addNode(api.CudaTransform(transform_params(
                edge, [transform_output(transformed_edge, source.transform.width, source.transform.height,
                                        sw_format=source.transform.sw_format, fit=source.transform.fit,
                                        crop=source.transform.crop, pass_arrays=True)],
                hwaccel=HWACCEL, name=f"transform_{index}", group=group, auto_restart="group")))
            edge = transformed_edge
        if source.filter_graph:
            filtered_edge = f"input_{index}_filtered"
            avp.addNode(api.FilterVideo({
                "name": f"source_filter_{index}", "src": edge, "dst": filtered_edge,
                "graph": (source.color.setparams + "," if source.color else "") + source.filter_graph, "hwaccel": HWACCEL,
                "group": group, "auto_restart": "group",
            }))
            edge = filtered_edge
        if process_source is not None:
            edge = process_source(SourceContext(avp, api, source, index, edge, group, HWACCEL, cfg.fps))
            if not isinstance(edge, str) or not edge.strip():
                raise ValueError(f"process_source for {source.id!r} must return a non-empty video edge name")
        input_edges.append(edge)
        for k in range(1, aliases[source.id] + 1):
            # The reusable builder shares conversion and fan-out for identical edges.
            mixer.add_source(mixer_config.alias_name(source.id, k), pre_otm_edge=edge,
                             input_group=group, default_graph="",
                             color=None if source.filter_graph else source.color,
                             packed_rgb=source.kind == "browser",
                             premultiplied_alpha=source.kind == "browser" and source.id in blended_sources,
                             pixel_format=source.filter_output_format or
                             (source.transform.sw_format if source.transform
                              else mixer_config.RAW_STORAGE.get(source.kind)))
    for scene in cfg.scenes:
        mixer.add_scene(scene.id, mixer_config.scene_layers(cfg, scene))
    mixer.set_initial_scene(cfg.initial_scene, slot="A")
    from pyplumber.mixer.aux import AuxBus, AuxBuses
    aux = AuxBuses(avp, [AuxBus(avp, api, mixer, cfg, bus) for bus in cfg.aux_buses]) if cfg.aux_buses else None
    keyer = DownstreamKeyer(avp, api, mixer, cfg, group=OUTPUT_GROUP) if cfg.dsk_keys else None
    renditions = cfg.renditions or _flag_renditions(options, *canvas)
    program = mixer.build()
    feeds = dict.fromkeys(FEEDS, program)
    if keyer:
        feeds = {**feeds, **keyer.build(program, clean=any(r.feed == "clean" for r in renditions))}
        register_dsk_commands(avp, keyer)
    pgm_taps = [b.pgm_edge for b in aux or () if b.pgm_edge]
    if pgm_taps:
        # An aux pgm cell shows what goes to air: the keyed program.
        tapped = "program_after_aux_tap"
        avp.addNode(api.OneToMany({
            "name": "program_aux_tap", "src": feeds["dirty"], "dst": [tapped, *pgm_taps],
            "outputs": 1, "subscribed_outputs": {edge: edge for edge in pgm_taps}, "group": OUTPUT_GROUP,
        }))
        feeds["dirty"] = tapped
    if aux:
        for bus in aux:
            bus.build(options)
        aux.register_commands()
    settings = json.dumps(cfg.settings(), separators=(",", ":")) + "\n"
    avp.registerControlCommand("mixer.settings", lambda _arg: settings, True)
    listener = _build_renditions(avp, api, options, renditions, feeds, canvas=canvas,
                                 working_format=cfg.working_format, color=cfg.out_color, backend=mixer.backend)
    return _application(avp, mixer, options, input_edges, listener, routed_inputs=False,
                        wipe_files=tuple(w.path for w in cfg.wipes), browser_windows=tuple(s.id for s in browsers), aux_buses=aux)
