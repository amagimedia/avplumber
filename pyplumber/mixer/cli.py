"""Config-driven mixer launcher with an optional --input shortcut.

The graph exposes fullscreen and paged 2/4/8/16-box scenes through the generic
AVPlumber mixer control protocol. It contains no audio or automatic selection
path; use ``tui.py`` to preview and take scenes manually.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import logging
import time
from dataclasses import dataclass

from pyplumber.mixer.application import (
    DEFAULT_FPS, FPS_DEN, HWACCEL, ROUTER_GROUP, JANUS_DEFAULT_HOST, JANUS_DEFAULT_VIDEO_PORT, JANUS_DEFAULT_VIDEO_PT,
    JANUS_DEFAULT_VIDEO_SSRC, JANUS_DEFAULT_VIDEO_BITRATE_KBPS,
    MixerOptions, MixerApplication, load_avp_api, run_application, _input_group, _pacing_loop, _init_avp, _make_builder, _flag_renditions,
    _build_renditions, _application, build_application as build_config_application,
)
from pyplumber.mixer.color import TEN_BIT_FORMATS, TRANSFER_TAGS
from pyplumber.mixer import config as mixer_config
from pyplumber.mixer.dmabuf_inputs import (
    dmabuf_cuda_input_nodes, is_dmabuf_url, open_browser_windows, wait_for_sockets, window_id,
)
from pyplumber.mixer.inputs import build_input
from pyplumber.mixer.janus import DEFAULT_KEYFRAME_MIN_INTERVAL_MS
from pyplumber.transform import transform_output, transform_params

from .tools.layouts import (
    CANONICAL_SOURCE_HEIGHT, CANONICAL_SOURCE_WIDTH, CANVAS_HEIGHT, CANVAS_WIDTH, all_scenes,
)


log = logging.getLogger("mixer")
# More --input sources than this are scaled to the canonical size and routed (see _register_sources).
DIRECT_INPUT_LIMIT = 32


@dataclass(frozen=True)
class GraphOptions(MixerOptions):
    inputs: tuple[str, ...] = ()
    loop_inputs: bool = False
    input_color: str = ""          # declared contract for every --input (sdr/hlg/pq); "" = frame tags
    config: str | None = None            # JSON document (sources, wipes, scenes) instead of --input
    webui_url: str = ""                  # AVPlumber web UI to register the graph with
    dmabuf_size: tuple[int, int] = (1920, 1080)
    dmabuf_open: str | None = None       # page URL: open the named windows before building

    @property
    def dmabuf_inputs(self) -> list[str]:
        return [window_id(url) for url in self.inputs if is_dmabuf_url(url)]

    def validate(self) -> None:
        super().validate()
        if self.config and self.inputs:
            raise ValueError("--config replaces --input; pass one or the other")
        if not self.inputs and not self.config:
            raise ValueError("at least one input is required")
        if not self.config and not self.output and not self.janus_output:
            raise ValueError("--output or --janus-output is required")
        if self.input_color and self.input_color not in TRANSFER_TAGS:
            raise ValueError("--input-color must be sdr, hlg or pq")
        if any(v <= 0 for v in self.dmabuf_size):
            raise ValueError("--dmabuf-size must be WxH with positive numbers")
        ids = self.dmabuf_inputs
        if len(ids) != len(set(ids)):
            raise ValueError("dmabuf window ids must be unique")
        if ids and len(self.inputs) > DIRECT_INPUT_LIMIT and self.input_color not in ("", "sdr"):
            raise ValueError(f"dmabuf:// inputs cannot join more than {DIRECT_INPUT_LIMIT} --input sources under "
                             f"--input-color {self.input_color}; use --config. A routed slot applies that one "
                             "color setting to every source it shows, and a browser page is SDR")


def _build_input(
    avp, api, index: int, url: str, *, loop: bool, fps: int, normalize: bool,
    options: GraphOptions,
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
    # Fit the source into the canonical size, black bars around it. The node declares that size
    # to the router; rate and time base are still read from the pacing node above it.
    # Storage is the decoders' 4:2:0 at the working depth: a source already at the canonical
    # size is passed on undrawn, an 8-bit source is promoted on a 10-bit show and a browser
    # frame is converted from RGB. cuda_transform does not reduce depth: a 10-bit source on an
    # nv12 show fails at its first frame.
    storage = "p010le" if options.working_format in TEN_BIT_FORMATS else "nv12"
    avp.addNode(api.CudaTransform(transform_params(
        fps_edge, [transform_output(normalized_edge, CANONICAL_SOURCE_WIDTH, CANONICAL_SOURCE_HEIGHT,
                                    sw_format=storage, fit="contain")],
        hwaccel=HWACCEL, name=f"normalize_{index}", group=group, auto_restart="group")))
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
        return build_config_application(mixer_config.load(
            options.config, max_compositor_layers=options.max_compositor_layers), options, api=api)
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
    add("--max-compositor-layers", type=int, help="Layer budget per compositor; overrides JSON max_compositor_layers "
        f"(default: {mixer_config.DEFAULT_MAX_COMPOSITOR_LAYERS})")
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
    run_application(application, webui_url=options.webui_url)


if __name__ == "__main__":
    main()
