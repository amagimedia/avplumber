"""Navigate and refresh browser source windows through the DMA-BUF browser service.

``mixer.source.set`` stops an FFmpeg/NDI input group and sets its ``url``; a browser pad is an
``ipc_dmabuf_source`` chain whose page lives in ``dma-browser``. These commands call that
service's REST API and leave the mixer graph alone. Window id is the show source id (as
``open_windows`` names it). Registered only when the show has browser sources.
"""
from __future__ import annotations

import json
import threading
from typing import Callable, Iterable, Optional

from .config import ConfigError
from .dmabuf_inputs import refresh_windows, update_window


class BrowserController:
    def __init__(self, *, dmabuf_rest: str, browser_ids: Iterable[str], mixer: str = "mixer",
                 navigate: Optional[Callable[[str, str, str], None]] = None,
                 refresh: Optional[Callable[[str, list[str]], None]] = None,
                 urls: Optional[dict[str, str]] = None):
        """*navigate(base, id, url)* and *refresh(base, ids)* default to the dma-browser helpers.
        *urls* is the shared id→page map ``mixer.sources`` reads; updated on each navigate."""
        self.dmabuf_rest, self.mixer = dmabuf_rest, mixer
        self.browser_ids = frozenset(browser_ids)
        self.urls = urls if urls is not None else {}
        self._navigate = navigate or update_window
        self._refresh = refresh or refresh_windows
        self._locks = {source_id: threading.Lock() for source_id in self.browser_ids}

    def navigate(self, request: dict) -> dict:
        source_id, url = self._id_and_url(request, "browser.navigate")
        with self._locks[source_id]:
            try:
                self._navigate(self.dmabuf_rest, source_id, url)
            except Exception as exc:
                raise RuntimeError(f"browser.navigate: {exc}") from None
            self.urls[source_id] = url
        return {"id": source_id, "url": url}

    def refresh(self, request: dict) -> dict:
        source_id = self._id(request, "browser.refresh")
        with self._locks[source_id]:
            try:
                self._refresh(self.dmabuf_rest, [source_id])
            except Exception as exc:
                raise RuntimeError(f"browser.refresh: {exc}") from None
        return {"id": source_id}

    def _id(self, request: dict, who: str) -> str:
        mixer = request.get("mixer", self.mixer)
        if mixer != self.mixer:
            raise ConfigError(f"unknown mixer: {mixer}")
        source_id = request.get("id")
        if not isinstance(source_id, str) or not source_id:
            raise ConfigError(f"{who} needs a non-empty id")
        unknown = set(request) - {"mixer", "id", "url"}
        if who == "browser.refresh":
            unknown = set(request) - {"mixer", "id"}
        if unknown:
            raise ConfigError(f"{who}: unsupported fields {sorted(unknown)}")
        if source_id not in self.browser_ids:
            raise ConfigError(f"{who}: source {source_id!r} is not a browser source")
        return source_id

    def _id_and_url(self, request: dict, who: str) -> tuple[str, str]:
        source_id = self._id(request, who)
        url = request.get("url")
        if not isinstance(url, str) or not url:
            raise ConfigError(f"{who} needs a non-empty url")
        return source_id, url


def register_browser_commands(avp, controller: BrowserController) -> None:
    def handle(method):
        return lambda arg: json.dumps(method(json.loads(arg))) + "\n"

    avp.registerControlCommand("mixer.browser.navigate", handle(controller.navigate), True)
    avp.registerControlCommand("mixer.browser.refresh", handle(controller.refresh), True)
