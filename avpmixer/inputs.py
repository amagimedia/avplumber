"""Decode chain shared by the mixer and playlist demos.

``input_rec -> demux -> dec_video -> [speed_video] -> realtime(set_pts) -> force_fps``

``realtime(set_pts=True)`` rebases every source onto the host monotonic clock,
which is the time base the native mixer schedules cuts in.
"""

from __future__ import annotations

from typing import Optional


def build_input(avp, api, tag: str, url: str, *, group: str, fps: int, fps_den: int = 1,
                hwaccel: str = "@gpu", loop: bool = False,
                input_params: Optional[dict] = None,
                speed_team: Optional[str] = None, speed: float = 1.0,
                pause_team: Optional[str] = None, sync_team: Optional[str] = None,
                realtime_params: Optional[dict] = None, decoder_params: Optional[dict] = None,
                pause_params: Optional[dict] = None,
                auto_restart: Optional[str] = "group") -> str:
    """Add the chain for one source and return its output edge (``input_<tag>_fps``).

    ``auto_restart="group"`` restarts the chain when a live source drops; pass
    ``None`` for file playback that is looped or seeked instead.  ``pause_team``
    adds a ``pause`` node before ``realtime`` and ``sync_team`` names the
    realtime team, so ``pause/resume <pause_team>`` holds the picture and
    ``seek <sync_team> now <ts>`` re-cues the input (the replay demo's wiring).
    """
    edge = lambda suffix: f"input_{tag}_{suffix}"  # noqa: E731
    restart = {} if auto_restart is None else {"auto_restart": auto_restart}
    sync = {} if sync_team is None else {"sync_team": sync_team}
    avp.addNode(api.InputRec({
        "name": f"input_{tag}", "url": url, "dst": edge("packets"), "loop": loop,
        "initial_timeout": 20, "timeout": 3_942_000_000, "group": group,
        **(input_params or {}),
    }))
    avp.addNode(api.Demux({
        "name": f"demux_{tag}", "src": edge("packets"),
        "routing": {"?v:0": edge("video_packets")}, "wait_for_keyframe": False,
        "group": group, **restart,
    }))
    avp.addNode(api.DecVideo({
        "name": f"decode_{tag}", "src": edge("video_packets"), "dst": edge("decoded"),
        "pixel_format": "?cuda", "hwaccel": hwaccel, "group": group, **restart,
        **(decoder_params or {}),
    }))
    realtime_src = edge("decoded")
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
                 sync_team=sync_team, realtime_params=realtime_params)


def _pace(avp, api, tag: str, src: str, *, fps: int, fps_den: int, group: str,
          sync_team: Optional[str] = None, realtime_params: Optional[dict] = None) -> str:
    """``realtime(set_pts) -> force_fps`` tail shared by every source chain:
    rebase onto the host clock, then fix the rate. Returns ``input_<tag>_fps``."""
    avp.addNode(api.Realtime({
        "name": f"realtime_{tag}", "src": src, "dst": f"input_{tag}_realtime",
        "set_pts": True, "group": group,
        **({} if sync_team is None else {"team": sync_team}), **(realtime_params or {}),
    }))
    avp.addNode(api.ForceFPS({
        "name": f"fps_{tag}", "src": f"input_{tag}_realtime", "dst": f"input_{tag}_fps",
        "fps": f"{fps}/{fps_den}", "group": group,
    }))
    return f"input_{tag}_fps"


def v210_row_stride(width: int) -> int:
    """Standard v210 row stride: ceil(width/48) * 128 bytes."""
    return ((width + 47) // 48) * 128


def build_v210_input(avp, api, tag: str, path: str, *, width: int, height: int, group: str,
                     fps: int, fps_den: int = 1, hwaccel: str = "@gpu", loop: bool = False,
                     working_format: str = "p210le", color: Optional[dict] = None) -> str:
    """Headerless packed v210 file -> GPU unpack -> paced output edge.

    ``input_rec -> demux -> v210_to_cuda -> realtime(set_pts) -> force_fps``.
    The packed bytes carry no metadata, so the color contract (HLG/BT.2020 for
    the HDR sources) is supplied here and stamped on the CUDA frames.
    """
    edge = lambda suffix: f"input_{tag}_{suffix}"  # noqa: E731
    restart = {} if loop else {"auto_restart": "group"}
    stride = v210_row_stride(width)
    avp.addNode(api.InputRec({
        "name": f"input_{tag}", "url": path, "dst": edge("packets"), "loop": loop,
        "format": "rawvideo", "initial_timeout": 20, "timeout": 3_942_000_000, "group": group,
        "options": {"pixel_format": "gray", "video_size": f"{stride}x{height}",
                    "framerate": f"{fps}/{fps_den}"},
    }))
    avp.addNode(api.Demux({
        "name": f"demux_{tag}", "src": edge("packets"), "routing": {"v:0": edge("packed")},
        "wait_for_keyframe": False, "group": group, **restart,
    }))
    avp.addNode(api.V210ToCuda({
        "name": f"unpack_{tag}", "src": edge("packed"), "dst": edge("cuda"),
        "hwaccel": hwaccel, "width": width, "height": height, "stride": stride,
        "fps": f"{fps}/{fps_den}", "timebase": "1/90000", "format": working_format,
        "group": group, **restart, **(color or {}),
    }))
    return _pace(avp, api, tag, edge("cuda"), fps=fps, fps_den=fps_den, group=group)
