"""Media clips replayed from GPU memory.

An optional module beside the mixer graph: it inserts a ``clip_cache`` node at
the end of a normal decode chain and splits that chain into two groups. The
loader group is started once at startup to fill the cache; the player group is
what a live transition starts, so a take costs no file open, no decoder and no
thread startup. Nothing here reaches into the compositor or the orchestrator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def loader_group(mixer_name: str) -> str:
    """Group holding the decode chain; started only to fill the cache."""
    return f"{mixer_name}_wipe_load"


def cache_node(*, name: str, src: str, dst: str, group: str, fps: str,
               store: str = "clips", budget_mb: float | None = None,
               url: str = "") -> Dict[str, Any]:
    """Parameters for the clip_cache node.

    ``url`` names the clip. The mixer sets it when it arms a media wipe, exactly
    as it did on the file reader this node replaces, so the transition code needs
    no knowledge of the cache.
    """
    node: Dict[str, Any] = {"name": name, "src": src, "dst": dst, "group": group,
                            "fps": fps, "cache": store, "url": url}
    if budget_mb is not None:
        node["budget_mb"] = budget_mb
    return node
