"""Shared test helpers: a fake command sink that records the exact AVP command
stream, plus small clip builders.  This is the backbone of the regression
harness -- every playlist feature is asserted against the recorded commands."""
from __future__ import annotations

import json
from typing import List

from playlist import Clip, ElementMode, PlaylistController, PlaylistMode


class Recorder:
    """Capture every command the controller emits."""
    def __init__(self):
        self.cmds: List[str] = []

    def __call__(self, cmd: str) -> None:
        self.cmds.append(cmd)

    def clear(self) -> None:
        self.cmds.clear()

    def matching(self, prefix: str) -> List[str]:
        return [c for c in self.cmds if c.startswith(prefix)]

    def added_nodes(self):
        out = []
        for c in self.cmds:
            if c.startswith("node.add "):
                out.append(json.loads(c[len("node.add "):]))
        return out

    def deleted_nodes(self):
        return [c[len("node.delete "):] for c in self.cmds
                if c.startswith("node.delete ")]

    def objects_set(self):
        """Return list of (node, key, value_str) for node.object.set commands."""
        out = []
        for c in self.cmds:
            if c.startswith("node.object.set "):
                _, _, rest = c.partition("node.object.set ")
                node, key, val = rest.split(" ", 2)
                out.append((node, key, val))
        return out


class FakeBackend(Recorder):
    """Records commands AND enforces the backend's node-namespace rule:
    node.add rejects a name that already exists (mirrors createNode's
    'Name busy'); node.delete frees it.  Lets tests catch rebuild bugs that a
    plain recorder would miss."""
    def __init__(self):
        super().__init__()
        self.live: set = set()

    def __call__(self, cmd: str) -> None:
        super().__call__(cmd)
        if cmd.startswith("node.add "):
            node = json.loads(cmd[len("node.add "):])
            name = node["name"]
            if name in self.live:
                raise AssertionError(f"Name busy: {name}")
            self.live.add(name)
        elif cmd.startswith("node.delete "):
            name = cmd[len("node.delete "):]
            self.live.discard(name)


def clips(*specs) -> List[Clip]:
    """specs: name or (name, element_mode) or (name, element_mode, kwargs)."""
    out = []
    for s in specs:
        if isinstance(s, str):
            out.append(Clip(url=f"/media/{s}", name=s))
        elif len(s) == 2:
            out.append(Clip(url=f"/media/{s[0]}", name=s[0], element_mode=s[1]))
        else:
            out.append(Clip(url=f"/media/{s[0]}", name=s[0], element_mode=s[1], **s[2]))
    return out


def controller(clip_list=None, mode=PlaylistMode.LOOP_ALL, **kw):
    rec = Recorder()
    cl = clip_list if clip_list is not None else clips("a", "b", "c")
    ctl = PlaylistController(rec, cl, mode=mode, **kw)
    rec.clear()  # drop construction noise (there is none, but be explicit)
    return ctl, rec
