"""Shared node builders for the DMA-BUF browser demo graphs."""

import threading

from pyplumber.node import (
    AssumeVideoFormat,
    Bsf,
    DrmPrimeToEglImage,
    DrmPrimeToCuda,
    EncVideo,
    FilterVideo,
    ForceKeyFrame,
    IpcDmabufSource,
    Mux,
    Output,
    PythonNode,
)


class FrameCounts:
    def __init__(self):
        self._lock = threading.Lock()
        self._input = 0
        self._output = 0

    def increment_input(self):
        with self._lock:
            self._input += 1

    def increment_output(self):
        with self._lock:
            self._output += 1

    def snapshot(self):
        with self._lock:
            return self._input, self._output

    def reset(self):
        with self._lock:
            self._input = 0
            self._output = 0


class CountInputFrames(PythonNode):
    def __init__(self, args, counts):
        self._counts = counts
        super().__init__(args)

    def process(self):
        frame = self._src.get()
        if frame:
            self._counts.increment_input()
            self._dst.enqueue(frame)


class CountOutputFrames(PythonNode):
    def __init__(self, args, counts):
        self._counts = counts
        super().__init__(args)

    def process(self):
        frame = self._src.get()
        if frame:
            self._counts.increment_output()


def duplicate_stats(counts):
    input_frames, unique_frames = counts.snapshot()
    duplicates = max(0, input_frames - unique_frames)
    duplicate_pct = 100.0 * duplicates / input_frames if input_frames else 0.0
    return input_frames, unique_frames, duplicates, duplicate_pct


def make_dmabuf_cuda_input_nodes(
    *,
    prefix,
    socket,
    width,
    height,
    fps,
    drm_hwaccel,
    cuda_hwaccel,
    source_group,
    processing_group,
):
    """Return the nodes and final CUDA RGB0 edge for one browser socket."""
    drm_edge = f"{prefix}_drm"
    assumed_edge = f"{prefix}_assumed"
    raw_cuda_edge = f"{prefix}_cuda_raw"
    cuda_edge = f"{prefix}_cuda"
    source_params = {
        "socket": socket,
        "dst": drm_edge,
        "group": source_group,
        "name": f"{prefix}_receive",
        "auto_restart": "group",
        "fps": f"{fps}/1",
    }
    if drm_hwaccel:
        source_params["hwaccel"] = drm_hwaccel
    nodes = [
        IpcDmabufSource(source_params),
        AssumeVideoFormat(
            {
                "width": width,
                "height": height,
                "pixel_format": "drm_prime",
                "real_pixel_format": "rgb0",
                "src": drm_edge,
                "dst": assumed_edge,
                "group": processing_group,
                "auto_restart": "panic",
            }
        ),
        DrmPrimeToCuda(
            {
                "hwaccel": cuda_hwaccel,
                "drop_alpha": True,
                "src": assumed_edge,
                "dst": raw_cuda_edge,
                "group": processing_group,
                "name": f"{prefix}_to_cuda",
                "auto_restart": "group",
            }
        ),
        FilterVideo(
            {
                # All browser windows use the same host monotonic clock. Snap
                # that shared clock to absolute 1/FPS boundaries before
                # changing its time base, so independently phased paint events
                # for one tick coalesce into one composite. Synthesizing one
                # timestamp per received frame would let the windows drift when
                # one misses a paint.
                "graph": (
                    f"setpts=round(PTS*TB*{fps})/(TB*{fps}),"
                    f"settb=expr=1/{fps}"
                ),
                "hwaccel": cuda_hwaccel,
                "src": raw_cuda_edge,
                "dst": cuda_edge,
                "dst_width": width,
                "dst_height": height,
                "dst_pixel_format": "cuda",
                "dst_frame_rate": f"{fps}/1",
                "group": processing_group,
                "name": f"{prefix}_timestamp",
                "auto_restart": "panic",
            }
        ),
    ]
    return nodes, cuda_edge


def make_dmabuf_egl_input_nodes(
    *,
    prefix,
    socket,
    width,
    height,
    fps,
    drm_hwaccel,
    max_cache_entries,
    source_group,
    processing_group,
):
    """Return the nodes and cached EGLImage edge for one browser socket."""
    drm_edge = f"{prefix}_drm"
    assumed_edge = f"{prefix}_assumed"
    egl_edge = f"{prefix}_egl"
    source_params = {
        "socket": socket,
        "dst": drm_edge,
        "group": source_group,
        "name": f"{prefix}_receive",
        "auto_restart": "group",
        "fps": f"{fps}/1",
    }
    if drm_hwaccel:
        source_params["hwaccel"] = drm_hwaccel
    return [
        IpcDmabufSource(source_params),
        AssumeVideoFormat(
            {
                "width": width,
                "height": height,
                "pixel_format": "drm_prime",
                "real_pixel_format": "rgb0",
                "src": drm_edge,
                "dst": assumed_edge,
                "group": processing_group,
                "auto_restart": "panic",
            }
        ),
        DrmPrimeToEglImage(
            {
                "src": assumed_edge,
                "dst": egl_edge,
                # Match the proven OBS importer: reuse the immutable EGLImage for
                # each physical allocation. The IPC release ACK keeps Chromium
                # from recycling that allocation until CUDA has finished reading.
                "cache_mode": "reuse",
                "ttl": 3,
                "max_cache_entries": max_cache_entries,
                "debug_log_every_n": 60,
                "group": processing_group,
                "name": f"{prefix}_to_egl",
                "auto_restart": "group",
            }
        ),
    ], egl_edge


def make_janus_h264_output_nodes(
    *,
    prefix,
    src,
    group,
    cuda_hwaccel,
    fps,
    bitrate,
    output_format,
    output_url,
    output_options,
    encoder_preset="p4",
    encoder_tune="ll",
    encoder_profile="baseline",
):
    """Return the common zero-copy CUDA-to-NVENC output chain."""
    keyframe_edge = f"{prefix}_keyframed"
    encoded_edge = f"{prefix}_encoded"
    headers_edge = f"{prefix}_headers"
    muxed_edge = f"{prefix}_muxed"
    return [
        ForceKeyFrame(
            {
                "interval_sec": "1/1",
                "src": src,
                "dst": keyframe_edge,
                "group": group,
                "auto_restart": "panic",
            }
        ),
        EncVideo(
            {
                "codec": "h264_nvenc",
                "hwaccel": cuda_hwaccel,
                "src": keyframe_edge,
                "dst": encoded_edge,
                "group": group,
                "name": f"{prefix}_encode",
                "auto_restart": "panic",
                "options": {
                    "b": bitrate,
                    "maxrate": bitrate,
                    "bufsize": bitrate,
                    "g": fps,
                    "bf": 0,
                    "rc": "cbr",
                    "preset": encoder_preset,
                    "tune": encoder_tune,
                    "profile": encoder_profile,
                    "forced-idr": 1,
                },
            }
        ),
        Bsf(
            {
                "bsf": "dump_extra=freq=keyframe",
                "src": encoded_edge,
                "dst": headers_edge,
                "group": group,
                "auto_restart": "panic",
            }
        ),
        Mux(
            {
                "src": [headers_edge],
                "dst": muxed_edge,
                "ts_sort_wait": 0,
                "group": group,
                "auto_restart": "panic",
            }
        ),
        Output(
            {
                "format": output_format,
                "url": output_url,
                "options": output_options,
                "src": muxed_edge,
                "group": group,
                "name": f"{prefix}_output",
                "auto_restart": "panic",
            }
        ),
    ]
