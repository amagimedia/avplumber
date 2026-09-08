"""Deterministic fakes shared by playlist policy and UI tests."""

from __future__ import annotations

from typing import List

from playlist import Clip, ElementMode, InMemoryBackend, PlaylistController, PlaylistMode


def clips(*specs) -> List[Clip]:
    """Build named clips with stable IDs; specs may include mode and kwargs."""
    result = []
    for spec in specs:
        if isinstance(spec, str):
            name, mode, kwargs = spec, ElementMode.PLAY_TO_END, {}
        elif len(spec) == 2:
            name, mode, kwargs = spec[0], spec[1], {}
        else:
            name, mode, kwargs = spec
        result.append(Clip(url=f"/media/{name}", name=name, item_id=f"item-{name}",
                           element_mode=mode, **kwargs))
    return result


class Clock:
    def __init__(self, now=0):
        self.now = now

    def __call__(self):
        return self.now


def controller(clip_list=None, mode=PlaylistMode.LOOP_ALL, auto_on_air=True, **kwargs):
    clock = Clock()
    backend = InMemoryBackend(auto_on_air=auto_on_air, clock=clock)
    ctl = PlaylistController(backend, clip_list if clip_list is not None else clips("a", "b", "c"),
                             mode=mode, **kwargs)
    ctl.poll(0)
    return ctl, backend, clock


def advance(ctl, clock, ms):
    clock.now += ms
    ctl.poll(clock.now)


def cues(backend):
    return [c for c in backend.calls if c[0] == "cue"]
