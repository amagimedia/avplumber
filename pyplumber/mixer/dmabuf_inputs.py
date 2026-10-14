"""Shared browser capture for the mixer application API.

One ``dma-browser`` window supplies DRM PRIME frames over a Unix socket.
The helpers manage windows and import their frames into CUDA on the mixer clock.
Applications declare browser sources through ``pyplumber.mixer.build_application``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import List, Tuple

from .config import browser_frame_queue_for, default_browser_frame_queue

SCHEME = "dmabuf://"
# How long an imported browser allocation stays registered without being used.
IMPORT_IDLE_MS = 10_000
# The return queue: frames the mixer has finished with wait here and go back to the browser one
# per new frame (ipc_dmabuf_source `return_queue`), so the browser's pool gets one buffer back for
# each it hands out and stops destroying and allocating buffers. It is part of the window's frame
# queue, which the browser service counts as frames not yet returned (its `ringSize`), and which
# must be at most DMA_BROWSER_EXPORT_BUFFERS - 2 (config.browser_frame_queue_for): Chromium lends
# that many buffers per window and the browser needs two of them for the frame being delivered
# and the one in its paint handler.
RETURN_QUEUE = 3
# Frames of the queue left to the mixer's own pipeline before any go to the return queue.
MIN_FRAMES_IN_FLIGHT = 3


def return_queue(frame_queue: int) -> int:
    """How much of a window's *frame_queue* is the return queue: three of the default nine, less
    for a shorter queue, none for one of three frames or fewer. A zero-copy browser needs none
    (BrowserCaptureMode)."""
    return max(0, min(RETURN_QUEUE, frame_queue - MIN_FRAMES_IN_FLIGHT))


@dataclass(frozen=True)
class BrowserCaptureMode:
    """How the browser service hands frames out, as it announces in ``/status`` (``captureMode``,
    ``exportBuffers``, on the service and on every window). The stock copy path keeps a retained
    pool per window and the mixer paces returns through a return queue. The zero-copy export is one
    capacity-bounded queue in Chromium: *export_buffers* buffers per window, a frame lent until the
    consumer acknowledges it, no paint while all are out. The mixer then acknowledges each frame as
    soon as its last user lets go (no return queue) and holds at most capacity less one."""
    zero_copy: bool = False
    export_buffers: int | None = None

    # Where a zero-copy frame is held besides a compositor's waiting queue, each at most one
    # distinct frame at a time: the receiver (its fence wait), the chain's edges and nodes up to
    # the fan-out (the edges run at capacity 1), the compositors' shown frame, and the release on
    # its way back. The browser keeps one buffer free to paint into and one may be in delivery.
    HELD_OUTSIDE_QUEUE = 4
    BROWSER_RESERVE = 2
    # The frames a compositor needs waiting at 30 fps with the default two-tick latency: a paint is
    # stamped for the next tick and drawn two ticks after it, so three wait while the fourth arrives.
    QUEUE_FOR_DEFAULT_LATENCY = 3

    def frame_queue(self, configured: int) -> int:
        """The frames this mixer may hold of a window: *configured*, lowered to the zero-copy
        browser's capacity less one."""
        if self.zero_copy and self.export_buffers:
            return min(configured, browser_frame_queue_for(self.export_buffers))
        return configured

    def playout_queue_limit(self) -> int | None:
        """How many frames of a zero-copy window a compositor may hold waiting for their tick so
        that every holder together stays within the browser's capacity less its reserve: the
        capacity less two, less the four held elsewhere; at least one. None for the stock path.
        Below QUEUE_FOR_DEFAULT_LATENCY (capacity under nine) the compositor repeats frames."""
        if not (self.zero_copy and self.export_buffers):
            return None
        return max(1, min(8, self.export_buffers - self.BROWSER_RESERVE - self.HELD_OUTSIDE_QUEUE))

    @classmethod
    def minimum_capacity(cls) -> int:
        """The smallest browser capacity that lets a compositor keep its default latency without
        repeating frames: reserve + held elsewhere + the queue the latency needs = 9."""
        return cls.BROWSER_RESERVE + cls.HELD_OUTSIDE_QUEUE + cls.QUEUE_FOR_DEFAULT_LATENCY


def browser_capture_mode(base_url: str) -> BrowserCaptureMode:
    """Ask the browser service how it captures. An unreachable or older service (no
    ``captureMode``) is the stock copy path."""
    try:
        status = rest_request(base_url, "GET", "/status") or {}
    except (RuntimeError, TimeoutError, OSError) as exc:
        logging.getLogger(__name__).warning("Browser %s: cannot read /status (%s); assuming the stock capture",
                                            base_url, exc)
        return BrowserCaptureMode()
    sources = [status, *status.get("windows", [])]
    mode = next((s.get("captureMode") for s in sources if s.get("captureMode")), None)
    buffers = next((s.get("exportBuffers") for s in sources if s.get("exportBuffers")), None)
    result = BrowserCaptureMode(zero_copy=mode == "zero-copy",
                                export_buffers=int(buffers) if isinstance(buffers, int) and buffers > 0 else None)
    logging.getLogger(__name__).info("Browser %s captures %s%s", base_url, mode or "copy (stock)",
                                     f" with {result.export_buffers} buffers per window" if result.export_buffers else "")
    if result.zero_copy and result.export_buffers and result.export_buffers < BrowserCaptureMode.minimum_capacity():
        logging.getLogger(__name__).warning(
            "Browser %s: %d buffers per window is below the %d a compositor needs at its default latency; "
            "browser frames will repeat (set DMA_BROWSER_EXPORT_BUFFERS=%d or more)",
            base_url, result.export_buffers, BrowserCaptureMode.minimum_capacity(), BrowserCaptureMode.minimum_capacity())
    return result


def zero_copy_chain_edges(prefix: str, hold: bool = True) -> List[str]:
    """The edges of one browser chain (dmabuf_cuda_input_nodes) to run at capacity one when the
    browser is zero-copy: a frame waiting in an edge is a buffer the browser cannot paint into."""
    return [f"{prefix}_{s}" for s in ("drm", "cuda_raw", "cuda", *(("held",) if hold else ()))]


def is_dmabuf_url(url: str) -> bool:
    return url.startswith(SCHEME)


def window_id(url: str) -> str:
    name = url[len(SCHEME):]
    if not name or "/" in name:
        raise ValueError(f"dmabuf input needs a window id: {url!r}")
    return name


def dmabuf_cuda_input_nodes(api, *, prefix: str, socket: str,
                            # Unused (geometry comes from each frame); kept for downstream callers.
                            width: int | None = None, height: int | None = None, fps: int,
                            drm_hwaccel: str | None, cuda_hwaccel: str, source_group: str,
                            processing_group: str, hold: bool = False,
                            preserve_alpha: bool = False, browser_frame_queue: int | None = None,
                            event_loop: str | None = None, zero_copy: bool = False) -> Tuple[list, str]:
    """Return the node list and the final CUDA edge for one browser socket.

    With *hold*, a ``repeat_last_frame`` node re-emits the last frame at *fps*
    while the page is not painting, so static pages keep feeding the mixer.
    *preserve_alpha* retains alpha-bearing DRM formats for blended scene items. Chromium's
    alpha is premultiplied; the consumer tags it (the mixer does in its colour node).
    Geometry and format come from each frame, and ``smooth_timestamps`` already stamps
    in 1/*fps*, so the chain has no format or time-base nodes of its own.
    *event_loop* names the loop that runs ``smooth_timestamps`` (``"default"`` when omitted).
    With *zero_copy* (BrowserCaptureMode) there is no return queue: each frame goes back as soon as
    its last user lets go, since Chromium's own queue has no buffer to spare; and the importer keeps
    every import for good (no idle expiry, no purge), since the browser's pool never changes while
    the window lives. The caller plans zero_copy_chain_edges() at capacity one."""
    if browser_frame_queue is None:
        browser_frame_queue = default_browser_frame_queue(fps)
    drm_edge, raw_edge, cuda_edge = (f"{prefix}_{s}" for s in ("drm", "cuda_raw", "cuda"))
    source = {"socket": socket, "dst": drm_edge, "group": source_group, "name": f"{prefix}_receive",
              "auto_restart": "group", "fps": f"{fps}/1"}
    if drm_hwaccel:
        source["hwaccel"] = drm_hwaccel
    if return_queue(browser_frame_queue) and not zero_copy:
        source["return_queue"] = return_queue(browser_frame_queue)
    nodes = [
        api.IpcDmabufSource(source),
        # zero_copy: the compositor samples the mapped DMA-BUF directly; no 8 MB copy per frame.
        # Keep the recycling allocation pool registered, including live frames.
        # The ring bounds outstanding frames, not retired Chromium allocations:
        # expire idle imports instead of pinning them indefinitely. Lookups refresh
        # returning allocations before expiry. A page that paints now and then returns
        # to an allocation seconds later, not ring cycles later: an import dropped in
        # between is registered again on its next use, and each release takes the CUDA
        # library's context lock exclusively, stalling every other thread in the context.
        api.DrmPrimeToCuda({"hwaccel": cuda_hwaccel, "drop_alpha": not preserve_alpha, "src": drm_edge,
                            "dst": raw_edge, "group": processing_group, "name": f"{prefix}_to_cuda",
                            "auto_restart": "group", "zero_copy": True,
                            "max_imports": max(32, 2 * browser_frame_queue + 2),
                            # A zero-copy browser's pool is fixed: an import is kept until the window
                            # closes or its geometry changes (0 retains; a release takes the CUDA
                            # context lock and stalls every draw).
                            "import_ttl_ms": 0 if zero_copy else
                            max(IMPORT_IDLE_MS, (4000 * browser_frame_queue + fps - 1) // fps)}),
        # Number paints on the canvas grid instead of rounding each arrival time to it.
        # A browser paints on its own clock, arriving up to ~7 ms early or late; rounding
        # an arrival near the middle of a slot flipped between two slots, putting two
        # paints in one tick and none in the next (one discard plus one repeat, up to
        # 4 per second on an unlucky browser). smooth_timestamps gives each paint the
        # previous slot plus one and resyncs only when the average drift passes 20 ms.
        # A paint gap longer than 100 ms (a page that stopped painting) resyncs at once,
        # so the next paint is not stamped late and discarded as stale.
        # round_up: number the first paint (and any resync) to the next canvas tick,
        # not the nearest, so a paint always precedes the tick it is shown at.
        api.SmoothTimestamps({
            "fps": f"{fps}/1", "discontinuity_threshold": 0.1, "round_up": True,
            "src": raw_edge, "dst": cuda_edge, "group": processing_group,
            "name": f"{prefix}_smooth", "auto_restart": "panic",
            **({} if event_loop is None else {"event_loop": event_loop})}),
    ]
    if hold:
        held_edge = f"{prefix}_held"
        nodes.append(api.RepeatLastFrame({"src": cuda_edge, "dst": held_edge, "fps": f"{fps}/1",
                                          "group": processing_group, "name": f"{prefix}_hold",
                                          "auto_restart": "group"}))
        return nodes, held_edge
    return nodes, cuda_edge


def rest_request(base_url: str, method: str, path: str, body=None, timeout: float = 60):
    """A TimeoutError means the service may still be carrying the request out."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(f"{base_url}{path}", data=data, method=method,
                                     headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Browser {method} {path}: HTTP {exc.code}: {detail or exc.reason}") from exc
    except TimeoutError as exc:
        raise TimeoutError(f"Browser {method} {path}: no answer within {timeout:g} s") from exc
    return json.loads(payload) if payload else None


def open_browser_windows(base_url: str, ids: List[str], page_url: str, width: int, height: int,
                         fps: int, browser_frame_queue: int | None = None) -> None:
    """Ensure the named windows match the requested page and capture settings."""
    if browser_frame_queue is None:
        browser_frame_queue = default_browser_frame_queue(fps)
    open_windows(base_url, [{"id": name, "url": page_url, "width": width, "height": height, "fps": fps,
                            "ringSize": browser_frame_queue, "holdLastFrame": True}
                            for name in ids])


def open_windows(base_url: str, windows: List[dict]) -> None:
    """Reuse matching windows; recreate only pages whose capture settings changed.

    Each open waits for its page load and a browser worker process opens one page at a time,
    so the changed pages open concurrently with one request in flight per worker."""
    started = time.monotonic()
    status = rest_request(base_url, "GET", "/status") or {}
    existing = {w["id"]: w for w in status.get("windows", [])}
    changed = []
    for spec in windows:
        wanted = {**spec, "audio": False}
        current = existing.get(spec["id"])
        if current and current.get("stats", {}).get("quarantinedFrameCount", 0):
            raise RuntimeError(f"Browser {spec['id']} has quarantined DMA-BUF frames; restart its browser worker")
        if not current or any(current.get(key) != value for key, value in wanted.items()):
            changed.append((wanted, current is not None))

    if changed:
        # The show's page count lets the service spread pages evenly over the fewest workers.
        try:
            rest_request(base_url, "POST", "/workers/plan", {"windows": len(windows)})
        except RuntimeError:   # an older service without the endpoint fills workers in order
            pass

    def reopen(wanted, exists):
        if exists:
            rest_request(base_url, "POST", "/window/close", {"id": wanted["id"]})
        opened = rest_request(base_url, "POST", "/window/open", wanted)
        ignored = [key for key in ("ringSize", "holdLastFrame") if key in wanted
                   and (opened or {}).get(key) != wanted[key]]
        if ignored:
            raise RuntimeError(f"Browser service did not apply {', '.join(ignored)}; "
                               "update dma-browser before starting the mixer")

    # A single-process browser service reports no workers: one request at a time.
    with ThreadPoolExecutor(max_workers=len(status.get("workers") or [None])) as pool:
        for opening in [pool.submit(reopen, *change) for change in changed]:
            opening.result()
    logging.getLogger(__name__).info("Browser windows ready: %d of %d opened in %.1f s",
                                     len(changed), len(windows), time.monotonic() - started)


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
