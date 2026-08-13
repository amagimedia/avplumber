"""Deterministic fakes shared by playlist policy and UI tests."""

from __future__ import annotations

from typing import List

from playlist import (BackendEvent, Clip, ElementMode, InMemoryBackend,
                      PlaylistController, PlaylistMode)


def clips(*specs) -> List[Clip]:
    """Build named clips with stable IDs; specs may include mode and kwargs."""
    result = []
    for spec in specs:
        if isinstance(spec, str):
            name, mode, kwargs = spec, ElementMode.PLAY_TO_END, {}
        elif len(spec) == 2:
            name, mode, kwargs = spec[0], spec[1], {}
        else:
            name, mode, kwargs = spec[0], spec[1], spec[2]
        result.append(Clip(
            url=f"/media/{name}", name=name, item_id=f"item-{name}",
            element_mode=mode, **kwargs))
    return result


class FakePlaybackBackend(InMemoryBackend):
    def __init__(self):
        super().__init__(auto_ready=False)

    def ready(self, clip: Clip, request_id: int) -> None:
        self.events.append(BackendEvent("ready", clip.item_id, request_id))

    def failed(self, clip: Clip, request_id: int, message: str) -> None:
        self.events.append(BackendEvent(
            "failed", clip.item_id, request_id, message=message))

    def eof(self, clip: Clip) -> None:
        self.events.append(BackendEvent("eof", clip.item_id))


def controller(clip_list=None, mode=PlaylistMode.LOOP_ALL):
    backend = FakePlaybackBackend()
    playlist = clip_list if clip_list is not None else clips("a", "b", "c")
    return PlaylistController(backend, playlist, mode=mode), backend


def finish_pending(ctl: PlaylistController, backend: FakePlaybackBackend) -> int:
    status = ctl.status()
    assert status.pending_index is not None
    request_id = backend.calls[-1][1]
    backend.ready(ctl.clips[status.pending_index], request_id)
    ctl.poll(0)
    return request_id
