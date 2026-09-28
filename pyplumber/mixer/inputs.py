"""Decode chain shared by the mixer and playlist demos.

``input_rec -> demux -> dec_video -> [speed_video] -> realtime(set_pts) -> force_fps``

``realtime(set_pts=True)`` rebases every source onto the host monotonic clock,
which is the time base the native mixer schedules cuts in.
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
                auto_restart: Optional[str] = "group", event_loop: Optional[str] = None) -> str:
    """Add the chain for one source and return its output edge (``input_<tag>_fps``).

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
        **({"pixel_format": "?cuda", "hwaccel": hwaccel} if hwaccel else {}),
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
                 sync_team=sync_team, realtime_params=realtime_params, event_loop=event_loop)


def _pace(avp, api, tag: str, src: str, *, fps: int, fps_den: int, group: str,
          sync_team: Optional[str] = None, realtime_params: Optional[dict] = None,
          event_loop: Optional[str] = None) -> str:
    """``realtime(set_pts) -> force_fps`` tail shared by every source chain:
    rebase onto the host clock, then fix the rate. Returns ``input_<tag>_fps``."""
    loop = {} if event_loop is None else {"event_loop": event_loop}
    avp.addNode(api.Realtime({
        "name": f"realtime_{tag}", "src": src, "dst": f"input_{tag}_realtime",
        "set_pts": True, "group": group, **loop,
        **({} if sync_team is None else {"team": sync_team}), **(realtime_params or {}),
    }))
    # set_pts stamps the wall clock at release: center_phase keeps its jitter from flipping frames
    # between output ticks (duplicate + drop pairs) on sources whose phase sits half a frame off.
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
                       pinned: bool = False) -> str:
    """CPU/GPU interop source: raw NV12/P010 file -> CUDA frames -> paced output edge.

    Default: ``input_rec -> demux -> dec_video(rawvideo) -> filter(setpts) ->
    realtime(set_pts) -> force_fps -> filter(hwupload)``. Pacing before upload bounds
    transfer work to the requested frame rate; rawvideo only wraps the existing bytes.

    ``pinned``: ``input_rec -> demux -> raw_to_cuda -> realtime(set_pts) -> force_fps``.
    raw_to_cuda uploads each packet through pinned staging on its own stream, so
    no decoder, setpts or FFmpeg hwupload is involved; it uploads one frame ahead
    of pacing (RAW_UPLOADED_CAPACITY). Opt-in until measured against the default.
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
                           event_loop=event_loop)
        output = f"input_{tag}_uploaded"
        avp.addNode(api.FilterVideo({"name": f"upload_{tag}", "src": edge, "dst": output,
                                    "graph": "hwupload", "hwaccel": hwaccel, "threads": threads, "group": group}))
        return output
    packets = _raw_file_packets(avp, api, tag, path, pixel_format=pixel_format, video_size=f"{width}x{height}",
                                group=group, fps=fps, fps_den=fps_den, loop=loop)
    uploaded = f"input_{tag}_cuda"
    # addNode creates the edge, so the plan must come first.
    avp.edges.planCapacity(uploaded, RAW_UPLOADED_CAPACITY)
    avp.addNode(api.RawToCuda({
        "name": f"upload_{tag}", "src": packets, "dst": uploaded,
        "hwaccel": hwaccel, "width": width, "height": height, "pixel_format": pixel_format,
        "fps": f"{fps}/{fps_den}", "timebase": "1/90000",
        "group": group, **_raw_file_restart(loop),
    }))
    return _pace(avp, api, tag, uploaded, fps=fps, fps_den=fps_den, group=group, event_loop=event_loop)


def build_v210_input(avp, api, tag: str, path: str, *, width: int, height: int, group: str,
                     fps: int, fps_den: int = 1, hwaccel: str = "@gpu", loop: bool = False,
                     color: Optional[dict] = None, event_loop: Optional[str] = None) -> str:
    """Headerless packed v210 file -> GPU unpack (P210, 10-bit 4:2:2) -> paced output edge.

    ``input_rec -> demux -> v210_to_cuda -> realtime(set_pts) -> force_fps``.
    The packed bytes carry no metadata, so the color contract (HLG/BT.2020 for
    the HDR sources) is supplied here and stamped on the CUDA frames.
    """
    stride = v210_row_stride(width)
    packets = _raw_file_packets(avp, api, tag, path, pixel_format="gray", video_size=f"{stride}x{height}",
                                group=group, fps=fps, fps_den=fps_den, loop=loop)
    avp.addNode(api.V210ToCuda({
        "name": f"unpack_{tag}", "src": packets, "dst": f"input_{tag}_cuda",
        "hwaccel": hwaccel, "width": width, "height": height, "stride": stride,
        "fps": f"{fps}/{fps_den}", "timebase": "1/90000", "sw_format": "p210le",
        "group": group, **_raw_file_restart(loop), **(color or {}),
    }))
    return _pace(avp, api, tag, f"input_{tag}_cuda", fps=fps, fps_den=fps_den, group=group, event_loop=event_loop)
