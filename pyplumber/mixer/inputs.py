"""Decode chain shared by the mixer and playlist demos.

``input_rec -> demux -> dec_video -> [speed_video] -> realtime(set_pts) -> [force_fps]``

``realtime(set_pts=True)`` rebases every source onto the host monotonic clock,
which is the time base the native mixer schedules cuts in. The mixer keeps
native input cadence; other callers retain rate normalization by default.
"""

from __future__ import annotations

from typing import Optional


# Input watchdog in seconds: effectively never, so a looped or paused file input is not torn down.
INPUT_TIMEOUT_S = 3_942_000_000


def build_input(avp, api, tag: str, url: str, *, group: str, fps: int, fps_den: int = 1,
                hwaccel: Optional[str] = "@gpu", loop: bool = False, continuous_loop: bool = False,
                input_params: Optional[dict] = None,
                speed_team: Optional[str] = None, speed: float = 1.0,
                pause_team: Optional[str] = None, sync_team: Optional[str] = None,
                realtime_params: Optional[dict] = None, decoder_params: Optional[dict] = None,
                decoded_filter: str = "", decoded_filter_threads: Optional[int] = None,
                pause_params: Optional[dict] = None,
                auto_restart: Optional[str] = "group", event_loop: Optional[str] = None,
                native_rate: bool = False) -> str:
    """Add the chain for one source and return its paced output edge.

    ``native_rate`` preserves source cadence for a clocked compositor; otherwise
    the shared playlist chain still normalizes to ``fps/fps_den``.

    ``auto_restart="group"`` restarts the chain when a live source drops; pass
    ``None`` for file playback that is looped or seeked instead.  ``continuous_loop``
    keeps timestamps rising across loop passes, so the decoder and ``realtime`` see
    no jump back at each wrap; only for free-running loops, since seeks then land on
    the shifted timeline.  ``pause_team``
    adds a ``pause`` node before ``realtime`` and ``sync_team`` names the
    realtime team, so ``pause/resume <pause_team>`` holds the picture and
    ``seek <sync_team> now <ts>`` re-cues the input (the replay demo's wiring).
    ``hwaccel=None`` keeps decoded frames on the CPU; ``decoded_filter`` runs
    before speed, pause and realtime pacing, with ``decoded_filter_threads``
    slice threads when given (FFmpeg's default otherwise).  ``event_loop`` names the event loop that runs the
    pacing nodes (the instance's ``"default"`` loop when omitted).
    """
    edge = lambda suffix: f"input_{tag}_{suffix}"  # noqa: E731
    restart = {} if auto_restart is None else {"auto_restart": auto_restart}
    sync = {} if sync_team is None else {"sync_team": sync_team}
    avp.addNode(api.InputRec({
        "name": f"input_{tag}", "url": url, "dst": edge("packets"), "loop": loop,
        "loop_continuous_ts": loop and continuous_loop, "initial_timeout": 20, "timeout": INPUT_TIMEOUT_S, "group": group,
        **(input_params or {}),
    }))
    avp.addNode(api.Demux({
        "name": f"demux_{tag}", "src": edge("packets"),
        "routing": {"?v:0": edge("video_packets")}, "wait_for_keyframe": False,
        "group": group, **restart,
    }))
    avp.addNode(api.DecVideo({
        "name": f"decode_{tag}", "src": edge("video_packets"), "dst": edge("decoded"),
        # The GPU decodes: a libavcodec frame thread only reserves one more decode surface, 16 of them
        # (about 55 MB of VRAM per 1080p decoder) on a 16-vCPU host.
        **({"pixel_format": "?cuda", "hwaccel": hwaccel, "options": {"threads": 1}} if hwaccel else {}),
        "group": group, **restart,
        **(decoder_params or {}),
    }))
    realtime_src = edge("decoded")
    if decoded_filter:
        avp.addNode(api.FilterVideo({
            "name": f"filter_{tag}", "src": realtime_src, "dst": edge("filtered"),
            "graph": decoded_filter, "group": group,
            **({"hwaccel": hwaccel} if hwaccel else {}),
            **({} if decoded_filter_threads is None else {"threads": decoded_filter_threads}),
        }))
        realtime_src = edge("filtered")
    if speed_team is not None:
        avp.addNode(api.SpeedVideo({
            "name": f"speed_{tag}", "src": realtime_src, "dst": edge("speeded"),
            "team": speed_team, "speed": speed, "sync_node": f"realtime_{tag}", "group": group, **sync,
        }))
        realtime_src = edge("speeded")
    if pause_team is not None:
        avp.addNode(api.Pause({
            "name": f"pause_{tag}", "src": realtime_src, "dst": edge("paused"),
            "team": pause_team, "group": group, **sync, **(pause_params or {}),
        }))
        realtime_src = edge("paused")
    return _pace(avp, api, tag, realtime_src, fps=fps, fps_den=fps_den, group=group,
                 sync_team=sync_team, realtime_params=realtime_params, event_loop=event_loop, native_rate=native_rate)


def _pace(avp, api, tag: str, src: str, *, fps: int, fps_den: int, group: str,
          sync_team: Optional[str] = None, realtime_params: Optional[dict] = None,
          event_loop: Optional[str] = None, native_rate: bool = False) -> str:
    """Rebase onto the host clock; optionally normalize cadence for non-clocked consumers."""
    loop = {} if event_loop is None else {"event_loop": event_loop}
    timing = {}
    if native_rate:
        # Without tick_source, realtime still follows wallclock. tick_period selects
        # its internal time base (one quarter of this period): 1/120000 represents
        # 1001/24000, 1001/30000 and 1001/60000 exactly. Millisecond PTS can cross
        # the compositor's nearest-tick boundary and cause repeat/drop pairs.
        # Its built-in thresholds are in milliseconds, so preserve their durations
        # explicitly when choosing the finer time base.
        timing = {"tick_period": "1/30000", "negative_time_tolerance": .25,
                  "discontinuity_threshold": 1.0}
    avp.addNode(api.Realtime({
        "name": f"realtime_{tag}", "src": src, "dst": f"input_{tag}_realtime",
        "set_pts": True, "group": group, **loop,
        **({} if sync_team is None else {"team": sync_team}), **timing, **(realtime_params or {}),
    }))
    if native_rate:
        return f"input_{tag}_realtime"
    # set_pts stamps the release schedule in whole milliseconds: center_phase keeps that rounding from
    # flipping frames between output ticks (duplicate + drop pairs) on sources half a frame off the grid.
    avp.addNode(api.ForceFPS({
        "name": f"fps_{tag}", "src": f"input_{tag}_realtime", "dst": f"input_{tag}_fps",
        "fps": f"{fps}/{fps_den}", "center_phase": True, "group": group, **loop,
    }))
    return f"input_{tag}_fps"


def v210_row_stride(width: int) -> int:
    """Standard v210 row stride: ceil(width/48) * 128 bytes."""
    return ((width + 47) // 48) * 128


def _raw_file_restart(loop: bool) -> dict:
    """A looped raw file never ends, so only a one-shot file restarts its group."""
    return {} if loop else {"auto_restart": "group"}


def _raw_file_packets(avp, api, tag: str, path: str, *, pixel_format: str, video_size: str,
                      group: str, fps: int, fps_den: int, loop: bool) -> str:
    """``input_rec(rawvideo) -> demux`` head of a headerless raw file source: one
    picture per packet, timestamps rising across loop passes. Returns the packet edge."""
    restart = _raw_file_restart(loop)
    avp.addNode(api.InputRec({
        "name": f"input_{tag}", "url": path, "dst": f"input_{tag}_packets", "loop": loop,
        "loop_continuous_ts": loop, "format": "rawvideo", "initial_timeout": 20, "timeout": INPUT_TIMEOUT_S, "group": group,
        "options": {"pixel_format": pixel_format, "video_size": video_size, "framerate": f"{fps}/{fps_den}"},
    }))
    avp.addNode(api.Demux({
        "name": f"demux_{tag}", "src": f"input_{tag}_packets", "routing": {"v:0": f"input_{tag}_packed"},
        "wait_for_keyframe": False, "group": group, **restart,
    }))
    return f"input_{tag}_packed"


# Capacity of a raw source's CUDA edge (input_<tag>_cuda): realtime keeps the head
# frame there until it is due, so the upload runs one frame ahead of pacing and
# each source holds one uploaded frame in VRAM before realtime (as NVDEC does).
RAW_UPLOADED_CAPACITY = 1


def build_raw420_input(avp, api, tag: str, path: str, *, width: int, height: int, group: str,
                       pixel_format: str, fps: int, fps_den: int = 1,
                       hwaccel: str = "@gpu", loop: bool = False, event_loop: Optional[str] = None,
                       pinned: bool = False, native_rate: bool = False) -> str:
    """CPU/GPU interop source: raw NV12/P010 file -> CUDA frames -> paced output edge.

    Default: ``input_rec -> demux -> dec_video(rawvideo) -> filter(setpts) ->
    realtime(set_pts) -> force_fps -> filter(hwupload)``. Pacing before upload bounds
    transfer work to the requested frame rate; rawvideo only wraps the existing bytes.

    ``pinned``: ``input_rec -> demux -> dec_video(rawvideo) ->
    filter(hwupload_cuda=pinned=1) -> realtime(set_pts) -> force_fps``.
    The patched filter copies decoded planes through pinned staging on its own
    stream, using the shared CUDA device. It uploads one frame ahead of pacing
    (RAW_UPLOADED_CAPACITY). Requires the pinned-upload FFmpeg patch.
    """
    if pixel_format not in ("nv12", "p010le"):
        raise ValueError("raw 4:2:0 upload requires nv12 or p010le")
    if not pinned:
        from .backends.cuda import CudaMixerBackend   # backends load only when used
        # Neither setpts nor hwupload does CPU slice work: one filter thread each.
        threads = CudaMixerBackend.graph_threads
        edge = build_input(avp, api, tag, path, group=group, fps=fps, fps_den=fps_den,
                           hwaccel=None, loop=loop, auto_restart=None if loop else "group",
                           input_params={"format": "rawvideo", "options": {
                               "pixel_format": pixel_format, "video_size": f"{width}x{height}",
                               "framerate": f"{fps}/{fps_den}"}},
                           decoder_params={"codec": "rawvideo", "pixel_format": pixel_format},
                           # InputRec seeks back to PTS zero at each loop. Count frames
                           # before pacing; setpts changes metadata only, not pixels.
                           decoded_filter=f"setpts=N*{fps_den}/({fps}*TB)", decoded_filter_threads=threads,
                           event_loop=event_loop, native_rate=native_rate)
        output = f"input_{tag}_uploaded"
        avp.addNode(api.FilterVideo({"name": f"upload_{tag}", "src": edge, "dst": output,
                                    "graph": "hwupload", "hwaccel": hwaccel, "threads": threads, "group": group}))
        return output
    packets = _raw_file_packets(avp, api, tag, path, pixel_format=pixel_format, video_size=f"{width}x{height}",
                                group=group, fps=fps, fps_den=fps_den, loop=loop)
    uploaded = f"input_{tag}_cuda"
    # addNode creates the edge, so the plan must come first.
    avp.edges.planCapacity(uploaded, RAW_UPLOADED_CAPACITY)
    decoded = f"input_{tag}_decoded"
    avp.addNode(api.DecVideo({
        "name": f"decode_{tag}", "src": packets, "dst": decoded,
        "codec": "rawvideo", "pixel_format": pixel_format,
        "group": group, **_raw_file_restart(loop),
    }))
    avp.addNode(api.FilterVideo({
        "name": f"upload_{tag}", "src": decoded, "dst": uploaded,
        "graph": "hwupload_cuda=pinned=1", "hwaccel": hwaccel, "threads": 1,
        "group": group, **_raw_file_restart(loop),
    }))
    return _pace(avp, api, tag, uploaded, fps=fps, fps_den=fps_den, group=group, event_loop=event_loop, native_rate=native_rate)


def build_v210_input(avp, api, tag: str, path: str, *, width: int, height: int, group: str,
                     fps: int, fps_den: int = 1, hwaccel: str = "@gpu", loop: bool = False,
                     color: Optional[dict] = None, event_loop: Optional[str] = None,
                     native_rate: bool = False) -> str:
    """Headerless packed v210 file -> GPU unpack (P210, 10-bit 4:2:2) -> paced output edge.

    ``input_rec -> demux -> dec_video(rawvideo/gray) ->
    filter(hwupload_cuda=v210_width=...) -> realtime(set_pts) -> force_fps``.
    The packed bytes carry no metadata, so the color contract (HLG/BT.2020 for
    the HDR sources) is supplied here and stamped on the CUDA frames.
    """
    stride = v210_row_stride(width)
    packets = _raw_file_packets(avp, api, tag, path, pixel_format="gray", video_size=f"{stride}x{height}",
                                group=group, fps=fps, fps_den=fps_den, loop=loop)
    decoded = f"input_{tag}_decoded"
    avp.addNode(api.DecVideo({
        "name": f"decode_{tag}", "src": packets, "dst": decoded,
        "codec": "rawvideo", "pixel_format": "gray",
        "group": group, **_raw_file_restart(loop),
    }))
    # Gray wraps the packed bytes; the patched uploader performs v210 unpacking
    # on the GPU and publishes P210 frames at the actual picture width.
    graph = f"hwupload_cuda=v210_width={width}"
    if color:
        names = {"color_range": "range", "colorspace": "colorspace",
                 "color_primaries": "color_primaries", "color_trc": "color_trc",
                 "chroma_location": "chroma_location"}
        tags = ":".join(f"{target}={color[source]}" for source, target in names.items() if source in color)
        if tags:
            graph += ",setparams=" + tags
    avp.addNode(api.FilterVideo({
        "name": f"unpack_{tag}", "src": decoded, "dst": f"input_{tag}_cuda",
        "hwaccel": hwaccel, "graph": graph, "threads": 1,
        "group": group, **_raw_file_restart(loop),
    }))
    return _pace(avp, api, tag, f"input_{tag}_cuda", fps=fps, fps_den=fps_den, group=group, event_loop=event_loop, native_rate=native_rate)
