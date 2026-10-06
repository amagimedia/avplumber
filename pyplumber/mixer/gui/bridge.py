"""Control transport, shared state polling and ordered takes; independent of HTTP."""
from __future__ import annotations

import asyncio
import collections
import json
import threading
import time

from pyplumber.mixer.control import AvpConnection, mixer_command

TAKE_COMMANDS = ("cut", "fade", "wipe", "preview", "interrupt")
PROGRAM_TAKES = ("cut", "fade", "wipe")
AUX_COMMANDS = ("aux", "aux_layout", "aux_page")
STATE_TTL_S = 0.2


class MixerBridge:
    """Serialized access to the mixer's control connection from HTTP threads."""

    def __init__(self, host: str, port: int, mixer: str, timeout: float = 10.0,
                 transition: str | None = None):
        self.mixer = mixer
        self.port = port
        self._lock = threading.Lock()
        self.timeout = timeout
        self.transition = transition
        self._connection = AvpConnection(host, port)
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True, name="mixer-bridge").start()
        self._init_sharing()

    def _init_sharing(self) -> None:
        """State every HTTP thread shares: the cached /api/state reply and the take queue."""
        self._state_lock = threading.Lock()
        self._state: dict | None = None
        self._state_at = self._changed_at = float("-inf")
        self._takes = threading.Condition()
        self._take_arrivals = 0
        self._take_queue: collections.deque[int] = collections.deque()   # waiting, in arrival order
        self._waiting_program: int | None = None
        self._take_running = False

    def _run(self, coro, timeout: float | None = None):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(self.timeout if timeout is None else timeout + 1.0)
        except Exception:
            future.cancel()
            raise

    def command(self, line: str, timeout: float | None = None) -> str | None:
        """Send one command, reconnecting once if the mixer was restarted. `timeout` bounds
        both the exchange and the wait for it, for replies that outgrow the default."""
        budget = self.timeout if timeout is None else timeout
        with self._lock:
            try:
                if not self._connection.connected:
                    self._run(self._connection.connect())
                return self._run(self._connection.command(line, budget), budget)
            except Exception:
                self._run(self._connection.disconnect())
                self._run(self._connection.connect())
                return self._run(self._connection.command(line, budget), budget)

    def status(self, timeout: float | None = None) -> dict:
        return json.loads(self.command(f"mixer.status {self.mixer}", timeout) or "{}")

    def state(self, timeout: float | None = None) -> dict:
        """Everything the page redraws from, in one poll. Polls within STATE_TTL_S of each
        other share one reply, and a poll arriving while another is in flight waits for it,
        so any number of tabs costs one set of commands; a take or key change expires it.
        A caller polling a mixer that is still building its graph passes a longer `timeout`:
        the mixer runs control commands on the thread that is busy starting nodes, so replies
        stall for seconds during startup even though they take a millisecond once it is running."""
        with self._state_lock:
            now = time.monotonic()
            if self._state is None or now - self._state_at >= STATE_TTL_S or self._state_at <= self._changed_at:
                self._state, self._state_at = self._query_state(timeout), now
            return dict(self._state)   # callers add their own top-level keys

    def _query_state(self, timeout: float | None) -> dict:
        state: dict = {"mixer": self.mixer}
        state["status"] = self.status(timeout)
        state["scenes"] = json.loads(self.command(f"mixer.scenes {self.mixer}", timeout) or "[]")
        try:
            state["settings"] = json.loads(self.command(f"mixer.settings {self.mixer}", timeout) or "{}")
        except Exception:
            state["settings"] = {}   # older mixers, or one started without a config
        if self.transition is not None:
            state["settings"]["transition"] = self.transition
        if state["settings"].get("aux_buses"):
            state["aux_buses"] = json.loads(self.command("mixer.aux_status", timeout) or "[]")
        if state["settings"].get("dsk_keys"):
            state["dsk"] = json.loads(self.command("mixer.dsk_status", timeout) or "[]")
        return state

    def take(self, request: dict):
        """Returns the mixer's answer to an aux, aux_layout, aux_page or dsk command; for a take, whether it was sent
        (False when a newer program take superseded it unsent)."""
        command = request.get("command")
        if command in ("dsk", *AUX_COMMANDS):
            payload = {k: v for k, v in request.items() if k != "command"}
            try:
                result = json.loads(self.command(f"mixer.{command} " + json.dumps(payload)) or "{}")
            finally:
                self._changed_at = time.monotonic()
            if isinstance(result, dict) and result.get("error"):
                raise ValueError(result["error"])
            return result
        if command not in TAKE_COMMANDS:
            raise ValueError(f"command must be one of {', '.join(TAKE_COMMANDS)}")
        payload = {k: v for k, v in request.items() if k not in ("command", "mixer")}
        if command != "interrupt" and not payload.get("scene"):
            raise ValueError(f"{command} needs a scene")
        return self._send_in_order(mixer_command(command, self.mixer, **payload), command in PROGRAM_TAKES)

    def _send_in_order(self, line: str, program: bool) -> bool:
        """Takes reach the mixer one at a time, in arrival order. A cut, fade or wipe sets the
        whole program state, so a newer one supersedes the one still waiting: a burst collapses
        to its last take, which is never dropped or overtaken. A preview or interrupt changes
        only part of it; it is never superseded and never supersedes a program take."""
        with self._takes:
            self._take_arrivals += 1
            mine = self._take_arrivals
            if program:
                if self._waiting_program is not None:
                    self._take_queue.remove(self._waiting_program)
                    self._takes.notify_all()   # the superseded take answers now
                self._waiting_program = mine
            self._take_queue.append(mine)
            while mine in self._take_queue and (self._take_running or self._take_queue[0] != mine):
                self._takes.wait()
            if mine not in self._take_queue:
                return False
            self._take_queue.popleft()
            if self._waiting_program == mine:
                self._waiting_program = None
            self._take_running = True
        try:
            self.command(line)
        finally:
            with self._takes:
                self._take_running = False
                self._changed_at = time.monotonic()
                self._takes.notify_all()
        return True
