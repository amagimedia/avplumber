"""Control transport, shared state polling and ordered takes; independent of HTTP."""
from __future__ import annotations

import asyncio
import collections
import json
import math
import threading
import time

from pyplumber.mixer.control import AvpConnection, AvpProtocolError, mixer_command

TAKE_COMMANDS = ("cut", "fade", "wipe", "preview", "interrupt")
PROGRAM_TAKES = ("cut", "fade", "wipe")
AUX_COMMANDS = ("aux", "aux_layout", "aux_page")
STATE_TTL_S = 0.2
# What a show fixes when it is built (its settings, its scenes, the layouts its AUX buses offer) is
# asked for this often, not on every poll: a new show is seen within that time of its start.
STATIC_TTL_S = 5.0
LEVER_END = 0.002   # this close to an end the lever is there (src/mixer/primitives/Lever.hpp)
LEVER_CLIENTS = 16   # controllers whose lever bookkeeping is kept; every page load is a new one


def lever_end(position: float) -> int | None:
    """The end (0 or 1) the lever is at, None between them."""
    return 0 if position <= LEVER_END else 1 if position >= 1 - LEVER_END else None


def remember_client(table: dict, client, value) -> None:
    """`table[client] = value` as its newest entry, forgetting the controllers heard from least
    recently beyond LEVER_CLIENTS."""
    table.pop(client, None)
    table[client] = value
    while len(table) > LEVER_CLIENTS:
        del table[next(iter(table))]


class LeverPacer:
    """Paces one controller's lever positions against network jitter. Each position carries the
    controller's time of sampling (`t_ms`, its own clock); sent minus sampled is the delay to the
    mixer plus a constant, and the smallest of the last WINDOW_S is an undelayed position's. A
    position sent `late` ms after that reaches the picture over its spacing from the previous one,
    shortened by `late` and lengthened by the recent peak of lateness (decaying with HALF_LIFE_S):
    every position then lands its sampling time plus that peak later, so the operator's own motion
    is replayed whatever the network did to it. On a clean network the peak decays to 0 and the
    span is the spacing alone, as without pacing."""

    WINDOW_S = 5.0
    HALF_LIFE_S = 1.0
    MAX_JITTER_MS = 100.0
    STROKE_GAP_MS = 100.0   # a longer pause between positions starts a new stroke

    def __init__(self) -> None:
        self._offsets: collections.deque[tuple[float, float]] = collections.deque()   # ascending offsets
        self._jitter_ms = 0.0
        self._jitter_at: float | None = None
        self._sent_t_ms: float | None = None

    def span_ms(self, t_ms: float, now: float) -> float:
        """The ramp for a position sampled at `t_ms`, sent at monotonic `now` (seconds)."""
        offset = now * 1000 - t_ms
        while self._offsets and self._offsets[-1][1] >= offset:
            self._offsets.pop()
        self._offsets.append((now, offset))
        while self._offsets[0][0] < now - self.WINDOW_S:
            self._offsets.popleft()
        late = offset - self._offsets[0][1]
        decay = 0.5 ** ((now - self._jitter_at) / self.HALF_LIFE_S) if self._jitter_at is not None else 0.0
        self._jitter_ms = min(self.MAX_JITTER_MS, max(late, self._jitter_ms * decay))
        self._jitter_at = now
        gap = t_ms - self._sent_t_ms if self._sent_t_ms is not None else math.inf
        self._sent_t_ms = t_ms
        if not 0 <= gap <= self.STROKE_GAP_MS:
            gap = 0.0
        # A position later than the peak of lateness lands at once: the mixer ramps over a frame.
        return max(1.0, gap + self._jitter_ms - late)


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
        # Lever positions have a connection of their own: the mixer serves each on its own thread,
        # so a state poll never holds a position back.
        self._lever_connection = AvpConnection(host, port)
        self._lever_connection_lock = threading.Lock()
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True, name="mixer-bridge").start()
        self._init_sharing()

    def _init_sharing(self) -> None:
        """State every HTTP thread shares: the cached /api/state reply and the take queue."""
        self._state_lock = threading.Lock()
        self._state: dict | None = None
        self._state_at = self._changed_at = float("-inf")
        self._static: dict | None = None
        self._static_at = float("-inf")
        self._takes = threading.Condition()
        self._take_arrivals = 0
        self._take_queue: collections.deque[int] = collections.deque()   # waiting, in arrival order
        self._waiting_program: int | None = None
        self._take_running = False
        self._keys_lock = threading.Lock()
        self._keys_at = float("-inf")
        self._keys = {"dsk": [], "ftb": {}}
        self._audio_gain_lock = threading.Lock()
        self._audio_gain_at = float("-inf")
        self._audio_gain: list = []
        self._lever_lock = threading.Lock()
        # Positions waiting while one is being sent: the newest, behind the end positions on its
        # way there, each of which only the same end again replaces (an end completes or cancels
        # the take).
        self._lever_waiting: collections.deque[tuple[dict, str | None]] = collections.deque()
        self._lever_sending = False
        self._lever_sent_end = True   # the previous position sent was at an end: the next starts a stroke
        self._lever_seq: dict[str, int] = {}   # newest position number per controller
        self._lever_pacers: dict[str | None, LeverPacer] = {}

    def keys(self, timeout: float | None = None) -> dict:
        """Small, shared poll for frame-derived key and fade-to-black tally, independent of the large show state."""
        with self._keys_lock:
            now = time.monotonic()
            if now - self._keys_at >= STATE_TTL_S or self._keys_at <= self._changed_at:
                self._keys = {"dsk": json.loads(self.command("mixer.dsk_status", timeout) or "[]"),
                              "ftb": json.loads(self.command("mixer.ftb_status", timeout) or "{}")}
                self._keys_at = now
            return {**self._keys, "dsk_sampled_at": self._keys_at}

    def audio_gain(self, timeout: float | None = None) -> list:
        """Small, shared poll for per-source audio gain, independent of the large show state."""
        with self._audio_gain_lock:
            now = time.monotonic()
            if now - self._audio_gain_at >= STATE_TTL_S or self._audio_gain_at <= self._changed_at:
                self._audio_gain = json.loads(self.command("mixer.audio_gain_status", timeout) or "[]")
                self._audio_gain_at = now
            return list(self._audio_gain)

    def lever(self, request: dict) -> bool:
        """queue_lever, sending on the caller's thread when no position was being sent."""
        if self.queue_lever(request):
            self.send_lever()
        return True

    def queue_lever(self, request: dict) -> bool:
        """The lever arm. A fader reports positions faster than they matter: the newest waits while
        one is being sent and replaces any older one waiting, so positions never queue up; an end
        position is replaced only by the same end again (an end completes or cancels the take).
        With the controller's `t_ms` each position carries the `span_ms` LeverPacer gives it.
        True when none was being sent: the caller is then the one to send_lever, and must."""
        position = request.get("position")
        if isinstance(position, bool) or not isinstance(position, (int, float)) or not 0 <= position <= 1:
            raise ValueError("lever needs a position from 0 to 1")
        # A page streams positions without waiting for answers; numbered per controller, one that
        # arrives after a newer one is dropped.
        client, seq, t_ms = request.get("client"), request.get("seq"), request.get("t_ms")
        fields = {k: v for k, v in request.items() if k not in ("command", "mixer", "client", "seq", "t_ms")}
        if isinstance(t_ms, (int, float)) and not isinstance(t_ms, bool) and math.isfinite(t_ms):
            fields["t_ms"] = t_ms   # stripped again when sent
        with self._lever_lock:
            if isinstance(client, str) and isinstance(seq, int):
                if seq <= self._lever_seq.get(client, -1):
                    return False
                remember_client(self._lever_seq, client, seq)
            end = lever_end(position)
            while self._lever_waiting and lever_end(self._lever_waiting[-1][0]["position"]) in (None, end):
                self._lever_waiting.pop()
            self._lever_waiting.append((fields, client if isinstance(client, str) else None))
            if self._lever_sending:
                return False
            self._lever_sending = True
            return True

    def send_lever(self) -> None:
        """Sends the waiting lever positions until none is left."""
        try:
            while True:
                with self._lever_lock:
                    if not self._lever_waiting:
                        self._lever_sending = False
                        return
                    fields, client = self._lever_waiting.popleft()
                    t_ms = fields.pop("t_ms", None)
                    if t_ms is not None:
                        pacer = self._lever_pacers.get(client) or LeverPacer()
                        remember_client(self._lever_pacers, client, pacer)
                        fields["span_ms"] = round(pacer.span_ms(t_ms, time.monotonic()), 3)
                    # The state a page polls changes where a stroke starts and at its ends.
                    at_end = lever_end(fields["position"]) is not None
                    changes = self._lever_sent_end or at_end
                    self._lever_sent_end = at_end
                self.command(mixer_command("lever", self.mixer, **fields))
                if changes:
                    self._changed_at = time.monotonic()
        except BaseException:
            with self._lever_lock:
                self._lever_waiting.clear()
                self._lever_sending = False
            raise

    def _run(self, coro, timeout: float | None = None):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(self.timeout if timeout is None else timeout + 1.0)
        except Exception:
            future.cancel()
            raise

    def command(self, line: str, timeout: float | None = None) -> str | None:
        """Send one command, reconnecting once if the mixer was restarted; a lever position on the
        lever's own connection. A command the mixer answered with an error is not sent again: it
        reached the mixer, and the error is the answer. `timeout` bounds both the exchange and the
        wait for it, for replies that outgrow the default."""
        budget = self.timeout if timeout is None else timeout
        lever = line.startswith("mixer.lever ")
        connection = self._lever_connection if lever else self._connection
        with self._lever_connection_lock if lever else self._lock:
            try:
                if not connection.connected:
                    self._run(connection.connect())
                return self._run(connection.command(line, budget), budget)
            except AvpProtocolError:
                raise
            except Exception:
                self._run(connection.disconnect())
                self._run(connection.connect())
                return self._run(connection.command(line, budget), budget)

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

    def _query_static(self, timeout: float | None) -> dict:
        static: dict = {"scenes": json.loads(self.command(f"mixer.scenes {self.mixer}", timeout) or "[]")}
        try:
            static["settings"] = json.loads(self.command(f"mixer.settings {self.mixer}", timeout) or "{}")
        except Exception:
            static["settings"] = {}   # older mixers, or one started without a config
        static["aux_layouts"] = {}
        if static["settings"].get("aux_buses"):
            try:
                static["aux_layouts"] = json.loads(self.command("mixer.aux_layouts", timeout) or "{}")
            except Exception:
                pass   # a mixer that still sends each bus's layouts with its status
        return static

    def _static_part(self, timeout: float | None) -> dict:
        """The show's fixed part, asked of the mixer every STATIC_TTL_S. Call with _state_lock held."""
        now = time.monotonic()
        if self._static is None or now - self._static_at >= STATIC_TTL_S:
            try:
                self._static, self._static_at = self._query_static(timeout), now
            except Exception:
                self._static = None   # the mixer went away: ask again for the next one
                raise
        return self._static

    def settings(self, timeout: float | None = None) -> dict:
        """The running show's settings alone, for a caller that needs nothing live (a player page)."""
        with self._state_lock:
            settings = dict(self._static_part(timeout)["settings"])
        if self.transition is not None:
            settings["transition"] = self.transition
        return settings

    def _query_state(self, timeout: float | None) -> dict:
        static = self._static_part(timeout)
        state: dict = {"mixer": self.mixer}
        try:
            state["status"] = self.status(timeout)
        except Exception:
            self._static = None
            raise
        state["scenes"] = static["scenes"]
        state["settings"] = dict(static["settings"])
        if self.transition is not None:
            state["settings"]["transition"] = self.transition
        if state["settings"].get("aux_buses"):
            buses = json.loads(self.command("mixer.aux_status", timeout) or "[]")
            # The page reads each bus's layouts beside its state; they come from the fixed part.
            state["aux_buses"] = [{**bus, "layouts": static["aux_layouts"][bus["id"]]}
                                  if bus.get("id") in static["aux_layouts"] else bus for bus in buses]
        if state["settings"].get("dsk_keys"):
            state.update(self.keys(timeout))
        if state["settings"].get("audio_sources"):
            state["audio_gain"] = self.audio_gain(timeout)
        return state

    def take(self, request: dict):
        """Returns the mixer's answer to an aux, aux_layout, aux_page or dsk command; for a take, whether it was sent
        (False when a newer program take superseded it unsent)."""
        command = request.get("command")
        if command == "lever":
            return self.lever(request)
        if command in ("dsk", "ftb", "audio_gain", *AUX_COMMANDS):
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
