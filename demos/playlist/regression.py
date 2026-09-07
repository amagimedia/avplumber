#!/usr/bin/env python3
"""Live regression against a running ``server.py``, over its control port.

    python3 regression.py --port 7778 [--fast]

Every check drives the same JSON protocol the TUI uses and asserts on the
returned status.  Timing checks wait for scheduled transitions, so a full run
takes about a minute; ``--fast`` skips them.  Exit status 0 on all PASS.
Use ``--local`` to run the same checks against the in-memory backend.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Callable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from control import LocalClient, RemoteClient, Snapshot  # noqa: E402


class Regression:
    def __init__(self, client, fast: bool, media_url: str):
        self.client, self.fast, self.media_url = client, fast, media_url
        self.results: List[Tuple[str, bool, str]] = []

    async def status(self) -> Snapshot:
        return await self.client.status()

    async def send(self, verb: str, **arg) -> Snapshot:
        await self.client.send(verb, **arg)
        await asyncio.sleep(0.05)
        return await self.status()

    async def wait(self, predicate: Callable[[Snapshot], bool], timeout: float, what: str) -> Snapshot:
        deadline = time.monotonic() + timeout
        while True:
            s = await self.status()
            if predicate(s):
                return s
            if time.monotonic() >= deadline:
                raise AssertionError(f"timeout waiting for {what}; last status: transport={s.transport} "
                                     f"active={s.active} error={s.error!r}")
            await asyncio.sleep(0.05)

    async def check(self, name: str, coro) -> None:
        try:
            await coro
            self.results.append((name, True, ""))
            print(f"PASS  {name}", flush=True)
        except Exception as exc:  # noqa: BLE001
            self.results.append((name, False, str(exc)))
            print(f"FAIL  {name}: {exc}", flush=True)

    # ---- checks ------------------------------------------------------------
    async def startup(self):
        s = await self.wait(lambda s: s.transport == "Playing" and s.output_alive, 15, "startup")
        assert s.active == 0 and s.next_at_ms is not None and not s.error, s.data

    async def transport(self):
        s = await self.send("pause")
        assert s.transport == "Paused" and s.next_at_ms is None
        pos = s.position_ms
        await asyncio.sleep(0.4)
        s = await self.status()
        assert s.position_ms == pos, "position moved while paused"
        s = await self.send("play")
        assert s.transport == "Playing" and s.next_at_ms is not None
        s = await self.send("stop")
        assert s.transport == "Stopped" and s.position_ms is None
        s = await self.send("play")
        s = await self.wait(lambda s: s.transport == "Playing" and s.active == 0, 10, "restart")
        assert s.position_ms < 1500, "restart did not begin at cue-in"

    async def manual_take(self):
        s = await self.send("take", index=2)
        s = await self.wait(lambda s: s.active == 2 and s.transport == "Playing", 10, "take 3")
        assert s.clips[2]["cue_in"] == 2000 and 1800 <= s.position_ms < 3500, s.position_ms
        s = await self.send("next")
        s = await self.wait(lambda s: s.active == 3, 10, "next")
        s = await self.send("prev")
        s = await self.wait(lambda s: s.active == 2, 10, "prev")
        assert not s.error, s.error

    async def element_actions(self):
        s = await self.send("hold", index=4)          # inactive element: parks quietly
        s = await self.send("park", index=4)
        assert s.transport == "Playing" and s.selected == 4
        s = await self.send("select", index=0)
        assert s.selected == 0 and s.active != 0

    async def modes(self):
        for mode, expect_next in (("PlayCurrent", None), ("LoopCurrent", "self"), ("PlayAll", "other"),
                                  ("LoopAll", "other")):
            s = await self.send("mode", mode=mode)
            assert s.mode == mode
            if expect_next is None:
                assert s.next is None and s.next_at_ms is None
            elif expect_next == "self":
                assert s.next == s.active
            else:
                assert s.next is not None and s.next != s.active and s.next_at_ms is not None

    async def transitions(self):
        s = await self.send("transition", transition="Fade", duration_ms=800)
        assert s.transition == "Fade" and s.transition_ms == 800
        s = await self.send("take", index=1)
        s = await self.wait(lambda s: s.active == 1, 10, "fade take")
        s = await self.send("transition", transition="Cut")
        assert s.transition == "Cut"

    async def editing(self):
        n = len((await self.status()).clips)
        s = await self.send("add", clip={"url": self.media_url, "name": "added"})
        assert len(s.clips) == n + 1 and s.clips[-1]["name"] == "added"
        s = await self.send("edit", index=n, clip={"url": self.media_url, "name": "added", "cue_in": 1000,
                                                    "cue_out": 3000, "speed": 1.0})
        assert s.clips[n]["cue_in"] == 1000 and s.clips[n]["cue_out"] == 3000
        s = await self.send("end_mode", index=n, end="Timed", duration=1500)
        assert s.clips[n]["end"] == "Timed" and s.clips[n]["duration"] == 1500
        s = await self.send("enable", index=n, enabled=False)
        assert s.clips[n]["disabled"]
        s = await self.send("move", index=n, to=0)
        assert s.clips[0]["name"] == "added"
        s = await self.send("remove", index=0)
        assert len(s.clips) == n and not s.error, s.error

    async def scheduled_advance(self):
        await self.send("mode", mode="LoopAll")
        s = await self.send("take", index=2)                       # 6 s element (2 s .. 8 s)
        s = await self.wait(lambda s: s.active == 2, 10, "take 3")
        armed_at = s.next_at_ms
        assert armed_at is not None and s.next == 3
        s = await self.wait(lambda s: s.active == 3, 12, "scheduled advance to 4")
        late_ms = s.now_ms - armed_at
        assert -100 <= late_ms <= 400, f"advance observed {late_ms} ms after schedule"
        assert not s.error, s.error
        # Timed element 4 (4 s, speed 2) hands over to 5 on time as well.
        s = await self.wait(lambda s: s.active == 4, 8, "timed advance to 5")

    async def run(self) -> bool:
        await self.check("startup", self.startup())
        await self.check("transport", self.transport())
        await self.check("manual take / next / prev", self.manual_take())
        await self.check("element actions", self.element_actions())
        await self.check("playlist modes", self.modes())
        await self.check("transitions", self.transitions())
        await self.check("editing", self.editing())
        if not self.fast:
            await self.check("scheduled advance", self.scheduled_advance())
        failed = [r for r in self.results if not r[1]]
        print(f"{len(self.results) - len(failed)}/{len(self.results)} checks passed")
        return not failed


async def main_async(args) -> int:
    if args.local:
        from playlist import InMemoryBackend, PlaylistController
        from server import default_clips
        now = lambda: time.monotonic_ns() // 1_000_000  # noqa: E731
        ctl = PlaylistController(InMemoryBackend(clock=now), default_clips(Path("/media/playlist")))
        ctl.play()
        client = LocalClient(ctl)
    else:
        client = RemoteClient(args.host, args.port)
        await client.connect()
    reg = Regression(client, args.fast, (await client.status()).clips[0]["url"])
    ok = await reg.run()
    await client.disconnect()
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7778)
    p.add_argument("--fast", action="store_true", help="skip checks that wait for scheduled transitions")
    p.add_argument("--local", action="store_true", help="run against the in-memory backend")
    return asyncio.run(main_async(p.parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
