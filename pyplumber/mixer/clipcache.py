"""Media clips replayed from GPU memory.

An optional module beside the mixer graph: it inserts a ``clip_cache`` node at
the end of a normal decode chain and splits that chain into two groups. The
loader group is started once at startup to fill the cache; the player group
runs from startup on, idle, and a live transition only arms it, so a take costs
no file open, no decoder and no node or thread startup. Nothing here reaches
into the compositor or the orchestrator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

# The shared store the cache node fills. The mixer reads it by this name for
# mixer.status, so the status report and the node see the same clips.
STORE = "clips"


def loader_group(mixer_name: str) -> str:
    """Group holding the decode chain; started only to fill the cache."""
    return f"{mixer_name}_wipe_load"


def cache_node(*, name: str, src: str, dst: str, group: str, fps: str,
               store: str = STORE, budget_mb: float | None = None,
               url: str = "") -> Dict[str, Any]:
    """Parameters for the clip_cache node.

    ``url`` names the clip the decode chain delivers when both groups start
    together (a preload without a running player). The running node is told
    what to load with its ``load`` object and what to replay with ``play``,
    which is how the mixer arms a media wipe.
    """
    node: Dict[str, Any] = {"name": name, "src": src, "dst": dst, "group": group,
                            "fps": fps, "cache": store, "url": url}
    if budget_mb is not None:
        node["budget_mb"] = budget_mb
    return node
