"""Browser pages from the DMA-BUF demo as mixer sources.

One ``dma-browser`` window is one Unix socket delivering DRM PRIME frames.
``dmabuf_cuda_input_nodes`` turns it into a CUDA edge on the shared monotonic
clock, snapped to the 1/fps grid, exactly as
``demos/dmabuf-browser/graph/dmabuf_browser_common.py`` does for that demo
(kept there unchanged because the demo's runtime image has no ``avpmixer``).
The REST helpers open the windows and wait for their sockets.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import List, Tuple

SCHEME = "dmabuf://"


def is_dmabuf_url(url: str) -> bool:
    return url.startswith(SCHEME)


def window_id(url: str) -> str:
    name = url[len(SCHEME):]
    if not name or "/" in name:
        raise ValueError(f"dmabuf input needs a window id: {url!r}")
    return name


def dmabuf_cuda_input_nodes(api, *, prefix: str, socket: str, width: int, height: int, fps: int,
                            drm_hwaccel: str | None, cuda_hwaccel: str, source_group: str,
                            processing_group: str, hold: bool = False) -> Tuple[list, str]:
    """Return the node list and the final CUDA edge for one browser socket.

    With *hold*, a ``repeat_last_frame`` node re-emits the last frame at *fps*
    while the page is not painting, so static pages keep feeding the mixer."""
    drm_edge, assumed_edge, raw_edge, cuda_edge = (f"{prefix}_{s}" for s in ("drm", "assumed", "cuda_raw", "cuda"))
    source = {"socket": socket, "dst": drm_edge, "group": source_group, "name": f"{prefix}_receive",
              "auto_restart": "group", "fps": f"{fps}/1"}
    if drm_hwaccel:
        source["hwaccel"] = drm_hwaccel
    nodes = [
        api.IpcDmabufSource(source),
        api.AssumeVideoFormat({"width": width, "height": height, "pixel_format": "drm_prime",
                               "real_pixel_format": "rgb0", "src": drm_edge, "dst": assumed_edge,
                               "group": processing_group, "auto_restart": "panic"}),
        api.DrmPrimeToCuda({"hwaccel": cuda_hwaccel, "drop_alpha": True, "src": assumed_edge,
                            "dst": raw_edge, "group": processing_group, "name": f"{prefix}_to_cuda",
                            "auto_restart": "group"}),
        api.FilterVideo({
            # Snap the shared host clock to absolute 1/fps boundaries before changing
            # its time base, so independently phased paints coalesce into one tick.
            "graph": f"setpts=round(PTS*TB*{fps})/(TB*{fps}),settb=expr=1/{fps}",
            "hwaccel": cuda_hwaccel, "src": raw_edge, "dst": cuda_edge, "dst_width": width,
            "dst_height": height, "dst_pixel_format": "cuda", "dst_frame_rate": f"{fps}/1",
            "group": processing_group, "name": f"{prefix}_timestamp", "auto_restart": "panic"}),
    ]
    if hold:
        held_edge = f"{prefix}_held"
        nodes.append(api.RepeatLastFrame({"src": cuda_edge, "dst": held_edge, "fps": f"{fps}/1",
                                          "group": processing_group, "name": f"{prefix}_hold",
                                          "auto_restart": "group"}))
        return nodes, held_edge
    return nodes, cuda_edge


def rest_request(base_url: str, method: str, path: str, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(f"{base_url}{path}", data=data, method=method,
                                     headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    return json.loads(payload) if payload else None


def open_browser_windows(base_url: str, ids: List[str], page_url: str, width: int, height: int,
                         fps: int) -> None:
    """Open (or reopen) the named windows on one page; other windows are left alone."""
    open_windows(base_url, [{"id": name, "url": page_url, "width": width, "height": height, "fps": fps}
                            for name in ids])


def open_windows(base_url: str, windows: List[dict]) -> None:
    """Open (or reopen) windows given as {id, url, width, height, fps} dicts."""
    status = rest_request(base_url, "GET", "/status") or {}
    existing = {w.get("id") for w in status.get("windows", [])}
    for spec in windows:
        if spec["id"] in existing:
            rest_request(base_url, "POST", "/window/close", {"id": spec["id"]})
        rest_request(base_url, "POST", "/window/open", {**spec, "audio": False})


def refresh_windows(base_url: str, ids: List[str]) -> None:
    """Reload pages so ones that only paint on load paint again into a connected chain."""
    for name in ids:
        rest_request(base_url, "POST", "/window/refresh", {"id": name})


def wait_for_sockets(paths: List[str], timeout_sec: float) -> None:
    deadline = time.monotonic() + timeout_sec
    missing = set(paths)
    while missing and time.monotonic() < deadline:
        missing = {path for path in missing if not os.path.exists(path)}
        if missing:
            time.sleep(0.25)
    if missing:
        raise RuntimeError(f"DMA-BUF sockets did not appear: {sorted(missing)}")
