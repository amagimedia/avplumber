#!/usr/bin/env python3
"""N-window DMA-BUF import, CUDA grid mix, and Janus scaling test."""

import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.environ.get("PYPLUMBER_PATH", "/opt/avplumber"))

import pyplumber
from pyplumber.node import (
    AssumeVideoFormat,
    CudaRectOverlay,
    EglImageCudaOverlay,
    FilterVideo,
    OneToMany,
    SmoothTimestamps,
)

from dmabuf_browser_common import (
    CountInputFrames,
    CountOutputFrames,
    FrameCounts,
    duplicate_stats,
    make_dmabuf_cuda_input_nodes,
    make_dmabuf_egl_input_nodes,
    make_janus_h264_output_nodes,
)
from dmabuf_output_config import resolve_output_config
from dmabuf_scale_layout import MAX_COMPOSITOR_INPUTS, fit_filter_graph, fit_rect, grid_cells


DEFAULT_SOURCE_URL = (
    "https://app.singular.live/output/6W76ei5ZNekKkYhe8nw5o8/Output?aspect=16:9"
)


def env_int(name, default, minimum=1):
    value = int(os.environ.get(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def env_float(name, default, minimum=0.0):
    value = float(os.environ.get(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def rest_request(base_url, method, path, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        method=method,
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    return json.loads(payload) if payload else None


def open_browser_windows(
    base_url, source_url, source_count, width, height, fps, prefix, reopen_delay_sec
):
    rest_request(base_url, "GET", "/window/close/all")
    if reopen_delay_sec:
        time.sleep(reopen_delay_sec)
    for index in range(source_count):
        rest_request(
            base_url,
            "POST",
            "/window/open",
            {
                "id": f"{prefix}_{index:02d}",
                "url": source_url,
                "width": width,
                "height": height,
                "fps": fps,
                "audio": False,
            },
        )


def wait_for_sockets(paths, timeout_sec):
    deadline = time.monotonic() + timeout_sec
    missing = set(paths)
    while missing and time.monotonic() < deadline:
        missing = {path for path in missing if not os.path.exists(path)}
        if missing:
            time.sleep(0.25)
    if missing:
        raise RuntimeError(f"DMA-BUF sockets did not appear: {sorted(missing)}")


def wait_for_edge(avp, edge, timeout_sec, data_type=None):
    deadline = time.monotonic() + timeout_sec
    while avp.getEdge(edge, data_type).occupied == 0:
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Timed out waiting for the first frame on {edge}")
        time.sleep(0.05)


def wait_with_heartbeats(avp, duration_sec):
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        time.sleep(min(1, max(0, deadline - time.monotonic())))
        avp.heartbeat()


def make_mpdecimate_probe(
    prefix,
    src,
    source_width,
    source_height,
    probe_width,
    probe_height,
    fps,
    group,
    cuda_hwaccel,
):
    assumed_edge = f"{prefix}_diag_assumed"
    downloaded_edge = f"{prefix}_diag_downloaded"
    counted_edge = f"{prefix}_diag_counted"
    unique_edge = f"{prefix}_diag_unique"
    counts = FrameCounts()
    nodes = [
        AssumeVideoFormat(
            {
                "width": source_width,
                "height": source_height,
                "pixel_format": "cuda",
                "real_pixel_format": "rgb0",
                "src": src,
                "dst": assumed_edge,
                "group": group,
                "auto_restart": "panic",
            }
        ),
        FilterVideo(
            {
                "graph": (
                    f"scale_cuda=w={probe_width}:h={probe_height},"
                    "hwdownload,format=rgb0,format=yuv444p"
                ),
                "hwaccel": cuda_hwaccel,
                "src": assumed_edge,
                "dst": downloaded_edge,
                "dst_width": probe_width,
                "dst_height": probe_height,
                "dst_pixel_format": "yuv444p",
                "dst_frame_rate": f"{fps}/1",
                "group": group,
                "name": f"{prefix}_diag_download",
                "auto_restart": "panic",
            }
        ),
        CountInputFrames(
            {
                "src": downloaded_edge,
                "dst": counted_edge,
                "group": group,
                "name": f"{prefix}_diag_input_count",
            },
            counts,
        ),
        FilterVideo(
            {
                "graph": os.environ.get("MPDECIMATE_FILTER", "mpdecimate"),
                "src": counted_edge,
                "dst": unique_edge,
                "dst_width": probe_width,
                "dst_height": probe_height,
                "dst_pixel_format": "yuv444p",
                "dst_frame_rate": f"{fps}/1",
                "group": group,
                "name": f"{prefix}_diag_mpdecimate",
                "auto_restart": "panic",
            }
        ),
        CountOutputFrames(
            {
                "src": unique_edge,
                "group": group,
                "name": f"{prefix}_diag_output_count",
            },
            counts,
        ),
    ]
    return nodes, counts


def print_duplicate_report(counts_by_source, final=False):
    label = "final" if final else "progress"
    totals = [0, 0, 0]
    for index, counts in enumerate(counts_by_source):
        input_frames, unique_frames, duplicates, duplicate_pct = duplicate_stats(counts)
        totals[0] += input_frames
        totals[1] += unique_frames
        totals[2] += duplicates
        print(
            f"[dmabuf_scale_mpdecimate] {label} source={index} "
            f"input={input_frames} unique={unique_frames} duplicates={duplicates} "
            f"duplicate_pct={duplicate_pct:.3f}",
            flush=True,
        )
    total_pct = 100.0 * totals[2] / totals[0] if totals[0] else 0.0
    print(
        f"[dmabuf_scale_mpdecimate] {label} source=all input={totals[0]} "
        f"unique={totals[1]} duplicates={totals[2]} duplicate_pct={total_pct:.3f}",
        flush=True,
    )


mode = os.environ.get("TEST_MODE", "grid").strip().lower()
if mode not in {"single", "grid"}:
    sys.exit("TEST_MODE must be 'single' or 'grid'")

requested_count = env_int("SOURCE_COUNT", 8)
source_count = 1 if mode == "single" else requested_count
compositor_backend = os.environ.get("COMPOSITOR_BACKEND", "egl_cuda").strip().lower()
if compositor_backend not in {"egl_cuda", "cuda"}:
    sys.exit("COMPOSITOR_BACKEND must be 'egl_cuda' or 'cuda'")
if compositor_backend == "cuda" and source_count > MAX_COMPOSITOR_INPUTS:
    sys.exit(
        f"SOURCE_COUNT exceeds cuda_rect_overlay's {MAX_COMPOSITOR_INPUTS}-input mask"
    )

source_url = os.environ.get("SOURCE_URL", DEFAULT_SOURCE_URL)
source_width = env_int("SOURCE_WIDTH", 1920, 16)
source_height = env_int("SOURCE_HEIGHT", 1080, 16)
fps = env_int("FPS", 60)
canvas_width = env_int("CANVAS_WIDTH", 1920, 16)
canvas_height = env_int("CANVAS_HEIGHT", 1080, 16)
socket_dir = os.environ.get("SOCKET_DIR", "/tmp/dma-page")
window_prefix = os.environ.get("WINDOW_PREFIX", "scale")
rest_url = os.environ.get("DMA_BROWSER_REST_URL", "http://127.0.0.1:9009")
render_node = os.environ.get("RENDER_NODE", "/dev/dri/renderD128")
duration_sec = env_float("TEST_DURATION_SEC", 0)
report_interval_sec = env_float("REPORT_INTERVAL_SEC", 10, 0.1)
mpdecimate_warmup_sec = env_float("MPDECIMATE_WARMUP_SEC", 0)
pre_graph_warmup_sec = env_float("PRE_GRAPH_WARMUP_SEC", 0)
browser_reopen_delay_sec = env_float("BROWSER_REOPEN_DELAY_SEC", 1)
reopen_browser_windows = env_flag("REOPEN_BROWSER_WINDOWS", True)
dmabuf_pool_size = env_int("DMA_BROWSER_DMABUF_POOL_SIZE", 11)
diagnose_duplicates = env_flag("MPDECIMATE_INPUTS", False)
if diagnose_duplicates and compositor_backend != "cuda":
    sys.exit("MPDECIMATE_INPUTS requires COMPOSITOR_BACKEND=cuda; diagnostics run separately")
if mpdecimate_warmup_sec and not diagnose_duplicates:
    sys.exit("MPDECIMATE_WARMUP_SEC requires MPDECIMATE_INPUTS=1")
use_drm_hwaccel = env_flag("USE_DRM_HWACCEL", True)
mpdecimate_width = env_int("MPDECIMATE_WIDTH", 640, 16)
mpdecimate_height = env_int("MPDECIMATE_HEIGHT", 360, 16)

window_ids = [f"{window_prefix}_{index:02d}" for index in range(source_count)]
sockets = [os.path.join(socket_dir, f"{window_id}.sock") for window_id in window_ids]
if reopen_browser_windows:
    open_browser_windows(
        rest_url,
        source_url,
        source_count,
        source_width,
        source_height,
        fps,
        window_prefix,
        browser_reopen_delay_sec,
    )
wait_for_sockets(sockets, env_float("SOCKET_TIMEOUT_SEC", 120, 1))
if pre_graph_warmup_sec:
    print(
        f"[dmabuf_scale] waiting {pre_graph_warmup_sec:g}s before graph startup",
        flush=True,
    )
    time.sleep(pre_graph_warmup_sec)

avp = pyplumber.AVPlumber()
if use_drm_hwaccel:
    avp.executeCommandsFromString(
        f'hwaccel.init {{ "name": "@drm", "type": "drm", "device": "{render_node}" }}'
    )
avp.executeCommandsFromString('hwaccel.init { "name": "@gpu", "type": "cuda" }')
avp.edges.planCapacity("*", 4)

cells = grid_cells(source_count, canvas_width, canvas_height)
mix_edges = []
layers = []
input_groups = []
counts_by_source = []

for index, (socket, cell) in enumerate(zip(sockets, cells)):
    prefix = f"source_{index:02d}"
    group = f"input_{index:02d}"
    input_groups.append(group)
    if compositor_backend == "egl_cuda":
        nodes, input_edge = make_dmabuf_egl_input_nodes(
            prefix=prefix,
            socket=socket,
            width=source_width,
            height=source_height,
            fps=fps,
            drm_hwaccel="@drm" if use_drm_hwaccel else None,
            max_cache_entries=dmabuf_pool_size,
            source_group=group,
            processing_group=group,
        )
    else:
        nodes, input_edge = make_dmabuf_cuda_input_nodes(
            prefix=prefix,
            socket=socket,
            width=source_width,
            height=source_height,
            fps=fps,
            drm_hwaccel="@drm" if use_drm_hwaccel else None,
            cuda_hwaccel="@gpu",
            source_group=group,
            processing_group=group,
        )
    for node in nodes:
        avp.addNode(node)

    mix_input_edge = input_edge
    if diagnose_duplicates:
        mix_input_edge = f"{prefix}_mix_input"
        diagnostic_edge = f"{prefix}_diagnostic_input"
        avp.addNode(
            OneToMany(
                {
                    "name": f"{prefix}_diagnostic_split",
                    "src": input_edge,
                    "dst": [mix_input_edge, diagnostic_edge],
                    "outputs": 3,
                    "group": group,
                }
            )
        )
        diagnostic_nodes, counts = make_mpdecimate_probe(
            prefix,
            diagnostic_edge,
            source_width,
            source_height,
            mpdecimate_width,
            mpdecimate_height,
            fps,
            group,
            "@gpu",
        )
        counts_by_source.append(counts)
        for node in diagnostic_nodes:
            avp.addNode(node)

    fitted = fit_rect(cell, source_width, source_height)
    if compositor_backend == "egl_cuda":
        mix_edges.append(mix_input_edge)
        layers.append(
            {
                "dst_x": fitted.x,
                "dst_y": fitted.y,
                "dst_w": fitted.width,
                "dst_h": fitted.height,
            }
        )
    else:
        cell_edge = f"{prefix}_cell"
        avp.addNode(
            FilterVideo(
                {
                    "name": f"{prefix}_fit_cell",
                    "src": mix_input_edge,
                    "dst": cell_edge,
                    "graph": fit_filter_graph(fitted),
                    "hwaccel": "@gpu",
                    "dst_width": fitted.width,
                    "dst_height": fitted.height,
                    "dst_pixel_format": "cuda",
                    "dst_frame_rate": f"{fps}/1",
                    "group": group,
                    "auto_restart": "panic",
                }
            )
        )
        mix_edges.append(cell_edge)
        layers.append({"dst_x": fitted.x, "dst_y": fitted.y})

if compositor_backend == "egl_cuda":
    mixed_program_edge = "scale_grid_program"
    avp.addNode(
        EglImageCudaOverlay(
            {
                "name": "scale_grid_mixer",
                "src": mix_edges,
                "dst": mixed_program_edge,
                "hwaccel": "@gpu",
                "width": canvas_width,
                "height": canvas_height,
                "fps": f"{fps}/1",
                "layers": layers,
                "max_cache_entries": source_count * dmabuf_pool_size,
                "debug_log_every_n": env_int("MIXER_DEBUG_EVERY_N", 0, 0),
                "group": "mixer",
            }
        )
    )
elif source_count == 1:
    mixed_program_edge = mix_edges[0]
else:
    mixed_program_edge = "scale_grid_program"
    avp.addNode(
        CudaRectOverlay(
            {
                "name": "scale_grid_mixer",
                "src": mix_edges,
                "dst": mixed_program_edge,
                "hwaccel": "@gpu",
                "width": canvas_width,
                "height": canvas_height,
                "sw_format": "rgb0",
                "layers": layers,
                "active_inputs": (1 << source_count) - 1,
                "warmup_timeout_ms": 5000,
                "debug_log_every_n": env_int("MIXER_DEBUG_EVERY_N", 0, 0),
                "group": "mixer",
            }
        )
    )

if compositor_backend == "egl_cuda":
    # The compositor owns the program clock and publishes CUDA/RGB0 format
    # metadata. A downstream fps/timestamp chain would only add buffering.
    program_edge = mixed_program_edge
    ready_program_edge = mixed_program_edge
else:
    fps_filtered_edge = "scale_program_fps_filtered"
    fps_limited_edge = "scale_program_fps_limited"
    avp.addNode(
        FilterVideo(
            {
                "name": "scale_program_fps",
                "graph": f"fps=fps={fps}",
                "hwaccel": "@gpu",
                "src": mixed_program_edge,
                "dst": fps_filtered_edge,
                "dst_width": canvas_width,
                "dst_height": canvas_height,
                "dst_pixel_format": "cuda",
                "dst_frame_rate": f"{fps}/1",
                "group": "rate",
                "auto_restart": "panic",
            }
        )
    )
    avp.addNode(
        AssumeVideoFormat(
            {
                "name": "scale_program_format",
                "width": canvas_width,
                "height": canvas_height,
                "pixel_format": "cuda",
                "real_pixel_format": "rgb0",
                "src": fps_filtered_edge,
                "dst": fps_limited_edge,
                "group": "rate",
                "auto_restart": "panic",
            }
        )
    )

    program_edge = "scale_program_timestamped"
    avp.addNode(
        SmoothTimestamps(
            {
                "name": "scale_program_timestamps",
                "fps": f"{fps}/1",
                "drift_window": 0,
                "discontinuity_threshold": 3600,
                "src": fps_limited_edge,
                "dst": program_edge,
                "group": "output",
                "auto_restart": "panic",
            }
        )
    )
    ready_program_edge = fps_limited_edge

janus_host = os.environ.get("JANUS_HOST", "127.0.0.1")
janus_port = env_int("JANUS_VIDEO_PORT", 5004)
janus_rtcp_port = env_int("JANUS_VIDEO_RTCP_PORT", janus_port + 1)
janus_pt = env_int("JANUS_VIDEO_PT", 96)
janus_ssrc = env_int("JANUS_VIDEO_SSRC", 0x41565001)
output_url = f"rtp://{janus_host}:{janus_port}?pkt_size=1200&rtcp_port={janus_rtcp_port}"
output_format, output_url, output_options = resolve_output_config(
    os.environ,
    output_url,
    janus_pt,
    janus_ssrc,
)
for node in make_janus_h264_output_nodes(
    prefix="scale_janus",
    src=program_edge,
    group="output",
    cuda_hwaccel="@gpu",
    fps=fps,
    bitrate=os.environ.get("VIDEO_BITRATE", "8000k"),
    output_format=output_format,
    output_url=output_url,
    output_options=output_options,
    encoder_preset=os.environ.get("VIDEO_ENCODER_PRESET", "p4"),
    encoder_tune=os.environ.get("VIDEO_ENCODER_TUNE", "ll"),
    encoder_profile=os.environ.get("VIDEO_ENCODER_PROFILE", "baseline"),
):
    avp.addNode(node)

graph_start_timeout_sec = env_float("GRAPH_START_TIMEOUT_SEC", 60, 1)
if compositor_backend != "egl_cuda":
    avp.group("rate").startNodes()
for group in input_groups:
    avp.group(group).startNodes()
if compositor_backend != "egl_cuda":
    for ready_edge in mix_edges:
        wait_for_edge(avp, ready_edge, graph_start_timeout_sec)
if compositor_backend == "egl_cuda" or source_count > 1:
    avp.group("mixer").startNodes()
wait_for_edge(avp, ready_program_edge, graph_start_timeout_sec)
if mpdecimate_warmup_sec:
    print(
        f"[dmabuf_scale] warming mpdecimate for {mpdecimate_warmup_sec:g}s",
        flush=True,
    )
    wait_with_heartbeats(avp, mpdecimate_warmup_sec)
    for counts in counts_by_source:
        counts.reset()
avp.group("output").startNodes()

print(
    f"[dmabuf_scale] mode={mode} sources={source_count} "
    f"backend={compositor_backend} "
    f"input={source_width}x{source_height}@{fps} "
    f"canvas={canvas_width}x{canvas_height} mpdecimate={diagnose_duplicates} "
    f"dmabuf_pool_size={dmabuf_pool_size} "
    f"mpdecimate_size={mpdecimate_width}x{mpdecimate_height} "
    f"output_format={output_format} output={output_url}",
    flush=True,
)

started = time.monotonic()
next_report = started + report_interval_sec
completed_duration = False
try:
    while True:
        time.sleep(1)
        avp.heartbeat()
        now = time.monotonic()
        if diagnose_duplicates and now >= next_report:
            print_duplicate_report(counts_by_source)
            next_report = now + report_interval_sec
        if duration_sec and now - started >= duration_sec:
            completed_duration = True
            break
except KeyboardInterrupt:
    pass
finally:
    if diagnose_duplicates:
        print_duplicate_report(counts_by_source, final=True)

# Python diagnostic counter nodes can still be waiting for the GIL while the
# native graph destructor joins them. A finite measurement has already emitted
# all of its results, so let process teardown release the CUDA/EGL resources
# instead of entering that destructor path.
if completed_duration:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
