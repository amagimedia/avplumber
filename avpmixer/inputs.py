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
                realtime_params: Optional[dict] = None,
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
            "team": pause_team, "group": group, **sync,
        }))
        realtime_src = edge("paused")
    avp.addNode(api.Realtime({
        "name": f"realtime_{tag}", "src": realtime_src, "dst": edge("realtime"),
        "set_pts": True, "group": group,
        **({} if sync_team is None else {"team": sync_team}), **(realtime_params or {}),
    }))
    avp.addNode(api.ForceFPS({
        "name": f"fps_{tag}", "src": edge("realtime"), "dst": edge("fps"),
        "fps": f"{fps}/{fps_den}", "group": group,
    }))
    return edge("fps")
