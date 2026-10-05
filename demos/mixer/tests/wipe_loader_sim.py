"""The native side of a wipe preload, as pyplumber.mixer.clipcache drives it.

Stands in for the loader chain (its group, nodes and edges) and for the clip
cache's ``load``/``forget``/``status``. Each load plays the next entry of
``script``; without one the clip arrives whole.
"""
import json
from types import SimpleNamespace

from pyplumber.mixer import clipcache

WHOLE = {"decoded": 120, "cached": 120}


class WipeLoaderSim:
    def __init__(self, mixer_name="mixer"):
        prefix = mixer_name + "_"
        self.loader = clipcache.loader_group(mixer_name)
        self.cache = prefix + clipcache.CACHE_NODE
        self.loader_nodes = {prefix + name for name in clipcache.LOADER_NODES}
        self.held = {prefix + name: 0 for name in clipcache.LOADER_EDGES}   # frames left on each edge
        self.decoded_edge = prefix + clipcache.DECODED_EDGE
        self.cache_input = prefix + clipcache.CACHE_INPUT_EDGE
        # Per load: decoded and cached frame counts; ends=False never delivers the
        # end marker; stop_polls keeps the chain alive that many looks after a stop
        # (a stalled upload); leftover is what it leaves on its edges and sends late.
        self.script = []
        self.outcome = WHOLE
        self.decoded_total = 0
        self.clips = {}
        self.loading = ""
        self.alive = 0        # looks until the stopped chain's nodes are gone
        self.backlog = 0      # late frames the cache still has to discard
        self.loads = []
        self.events = []
        self.dirty_loads = []   # accepted while an earlier load's chain or frames were still there

    def executeCommandsFromString(self, commands):
        for line in commands.splitlines():
            parts = line.split(" ", 3)
            if parts[:2] != ["node.object.set", self.cache]:
                continue
            clip = json.loads(parts[3])
            self.events.append(f"{parts[2]} {clip}")
            if parts[2] == "forget":
                self.clips.pop(clip, None)
            elif parts[2] == "load":
                if self.alive or self.backlog or any(self.held.values()):
                    self.dirty_loads.append(clip)
                self.loads.append(clip)
                self.loading = clip
                self.outcome = self.script.pop(0) if self.script else WHOLE

    def startNodes(self):
        self.events.append("start")
        self.alive = -1   # running
        self.decoded_total += self.outcome["decoded"] + 1   # the decoder's end marker is an item too
        if self.outcome.get("ends", True):
            if self.outcome["cached"]:
                self.clips[self.loading] = self.outcome["cached"]
            self.loading = ""

    def stopNodes(self):
        self.events.append("stop")
        self.alive = self.outcome.get("stop_polls", 0)
        leftover = self.outcome.get("leftover", 0)
        self.backlog = leftover
        self.held = dict.fromkeys(self.held, leftover)

    def group(self, name):
        return self if name == self.loader else None

    def status(self, _key="status"):
        return {"loading": self.loading,
                "clips": [{"path": path, "frames": frames, "bytes": frames << 20, "complete": True}
                          for path, frames in self.clips.items()]}

    def node(self, name):
        if name == self.cache:
            return SimpleNamespace(isWorking=True, getObject=self.status)
        if name not in self.loader_nodes:
            return None
        working = self.alive != 0
        if self.alive > 0:
            self.alive -= 1
        return SimpleNamespace(isWorking=working)

    def _clear(self, name):
        assert not self.alive, "an edge was cleared under a running node"
        self.events.append("clear " + name)
        self.held[name] = 0

    def getEdge(self, name, data_type=None):
        if name in self.held:
            return SimpleNamespace(occupied=self.held[name], clear=lambda: self._clear(name),
                                   enqueued_total=self.decoded_total if name == self.decoded_edge else 0)
        if name == self.cache_input:
            occupied, self.backlog = self.backlog, max(self.backlog - 1, 0)   # the cache discards as it runs
            return SimpleNamespace(occupied=occupied, enqueued_total=0)       # its consumer runs: no clear()
        return None
