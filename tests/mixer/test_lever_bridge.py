"""The lever through the web UI bridge: what reaches mixer.lever, end positions behind a slow mixer,
pacing against network jitter, the lever's own control connection and no resend after an error."""
from __future__ import annotations

import json
import socketserver
import threading
import time

import pytest

from pyplumber.mixer.control import AvpProtocolError
from pyplumber.mixer.gui.bridge import LeverPacer, MixerBridge
from test_webui import FakeBridge


def sent_lever(bridge) -> list[dict]:
    return [json.loads(line.split(" ", 1)[1]) for line in bridge.sent if line.startswith("mixer.lever ")]


def test_lever_request_reaches_the_mixer_without_the_pages_bookkeeping():
    bridge = FakeBridge()
    bridge.take({"command": "lever", "position": 0.25, "seq": 1, "client": "page", "t_ms": 1000.0,
                 "wipe_file": "/clips/wipe.mov", "duration_sec": 1, "curve": "linear", "color": None})
    bridge.take({"command": "lever", "position": 0.3, "wipe_file": "/clips/wipe.mov"})
    paced, plain = sent_lever(bridge)
    assert paced["wipe_file"] == "/clips/wipe.mov" and paced["mixer"] == "mixer" and paced["position"] == 0.25
    assert {"seq", "client", "t_ms", "command"}.isdisjoint(paced) and paced["span_ms"] >= 1
    assert "span_ms" not in plain and plain["wipe_file"] == "/clips/wipe.mov"
    bridge.take({"command": "lever", "position": 0.35, "t_ms": True})   # not a time: no pacing
    assert "span_ms" not in sent_lever(bridge)[-1]


def test_an_end_position_is_never_replaced_behind_a_slow_mixer():
    bridge = FakeBridge()
    bridge.gates["mixer.lever"] = gate = threading.Event()
    first = threading.Thread(target=bridge.take, args=({"command": "lever", "position": 0.5},))
    first.start()
    while not bridge.sent:
        time.sleep(0.01)
    for position in (1.0, 0.9, 0.8, 0.0, 0.1, 0.2):
        assert bridge.take({"command": "lever", "position": position}) is True
    gate.set()
    first.join(5)
    assert [line["position"] for line in sent_lever(bridge)] == [0.5, 1.0, 0.0, 0.2]


def test_an_end_held_behind_a_slow_mixer_is_sent_once():
    # Hand jitter at the end of the slider reports that end over and over: each repeat replaces
    # the one waiting, so the operator's next real position waits behind one end, not hundreds.
    bridge = FakeBridge()
    bridge.gates["mixer.lever"] = gate = threading.Event()
    first = threading.Thread(target=bridge.take, args=({"command": "lever", "position": 0.5},))
    first.start()
    while not bridge.sent:
        time.sleep(0.01)
    for seq, position in enumerate((1.0, 0.999, 0.9, 1.0, 0.9995, *[1.0] * 100, 0.6), start=1):
        assert bridge.take({"command": "lever", "position": position, "seq": seq, "client": "page"}) is True
    gate.set()
    first.join(5)
    assert [line["position"] for line in sent_lever(bridge)] == [0.5, 1.0, 0.6]


def test_state_expires_where_a_stroke_starts_and_at_its_ends_only():
    bridge = FakeBridge()
    changes = []
    for position in (0.3, 0.4, 0.6, 1.0, 0.9, 0.7):
        before = bridge._changed_at
        bridge.take({"command": "lever", "position": position})
        changes.append(bridge._changed_at != before)
    assert changes == [True, False, False, True, True, False]


class Clock:
    """Positions sampled every `period_ms` on the page's clock, sent `delay_ms` later."""

    def __init__(self, pacer: LeverPacer, period_ms: float = 16.0, offset_ms: float = 123456.0):
        self.pacer, self.period_ms, self.offset_ms = pacer, period_ms, offset_ms
        self.t_ms = 0.0

    def send(self, delay_ms: float = 0.0, gap_ms: float | None = None) -> tuple[float, float]:
        """Returns (span_ms, where that ramp ends minus the sampling time)."""
        self.t_ms += self.period_ms if gap_ms is None else gap_ms
        sent_ms = self.t_ms + self.offset_ms + delay_ms
        span = self.pacer.span_ms(self.t_ms, sent_ms / 1000)
        return span, sent_ms + span - self.t_ms


def test_clean_network_spans_are_the_controllers_own_spacing():
    clock = Clock(LeverPacer())
    assert clock.send()[0] == 1.0   # a stroke's first position: at once
    for _ in range(50):
        assert clock.send()[0] == pytest.approx(16.0)


def test_jitter_is_paced_so_positions_land_at_their_spacing():
    pacer = LeverPacer()
    clock = Clock(pacer)
    clock.send()
    clock.send(delay_ms=40)   # one late position raises the peak of lateness
    landings = [clock.send(delay_ms=delay)[1] for delay in (0, 25, 5, 38, 0, 12, 30) * 4]
    # Each ramp ends its sampling time plus a constant, whatever its own delay: within the decay
    # of the peak over the burst (half-life 1 s, a burst of 0.45 s).
    assert max(landings) - min(landings) < 40 * (1 - 0.5 ** 0.45) + 1
    # The peak decays: after 10 s of clean positions the span is the spacing again.
    for _ in range(625):
        span, _ = clock.send()
    assert span == pytest.approx(16.0, abs=0.1)


def test_lateness_is_capped_and_a_pause_starts_a_new_stroke():
    clock = Clock(LeverPacer())
    clock.send()
    clock.send()
    span, _ = clock.send(delay_ms=500)   # far beyond the cap: lands at once
    assert span == 1.0
    # The next position, sent 100 ms later and on time after a pause: the peak is the cap, decayed.
    span, _ = clock.send(gap_ms=600)
    assert span == pytest.approx(100.0 * 0.5 ** 0.1, abs=0.01)
    fresh = Clock(LeverPacer())
    fresh.send()
    span, _ = fresh.send(gap_ms=400)   # the operator paused: no ramp over the pause
    assert span == 1.0


class ControlServer(socketserver.ThreadingTCPServer):
    """AVPlumber's line protocol: a greeting, then one reply per command line, each recorded with
    the connection it came on; `refuse` prefixes are answered with an error."""
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, refuse=()):
        self.lines: list[tuple[int, str]] = []
        self.connections = 0
        self.refuse = refuse
        super().__init__(("127.0.0.1", 0), self.Handler)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            server = self.server
            server.connections += 1
            connection = server.connections
            self.wfile.write(b"100 avplumber\n")
            for raw in self.rfile:
                line = raw.decode().rstrip("\n")
                server.lines.append((connection, line))
                refused = any(line.startswith(prefix) for prefix in server.refuse)
                self.wfile.write(b"400 mixer: the lever wipes only cached clips\n" if refused else b"200 OK\n")


@pytest.fixture
def control_server():
    servers = []

    def start(**kwargs):
        server = ControlServer(**kwargs)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server
    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def test_lever_positions_have_a_connection_of_their_own(control_server):
    server = control_server()
    bridge = MixerBridge("127.0.0.1", server.server_address[1], "mixer")
    bridge.command("mixer.status mixer")
    bridge.lever({"position": 0.4})
    bridge.command("mixer.scenes mixer")
    bridge.lever({"position": 0.5})
    status, lever, scenes, lever_again = server.lines
    assert status[0] == scenes[0] and lever[0] == lever_again[0] and status[0] != lever[0]
    assert lever[1].startswith("mixer.lever ") and server.connections == 2


def test_a_command_the_mixer_refused_is_not_sent_again(control_server):
    server = control_server(refuse=("mixer.lever ",))
    bridge = MixerBridge("127.0.0.1", server.server_address[1], "mixer")
    with pytest.raises(AvpProtocolError, match="cached clips"):
        bridge.lever({"position": 0.4, "wipe_file": "/clips/uncached.mov"})
    assert [line for _, line in server.lines] == [
        'mixer.lever {"position":0.4,"wipe_file":"/clips/uncached.mov","mixer":"mixer"}']
    assert server.connections == 1
    # The lever is free for the next position, on the same connection.
    with pytest.raises(AvpProtocolError):
        bridge.lever({"position": 0.5, "wipe_file": "/clips/uncached.mov"})
    assert len(server.lines) == 2 and server.connections == 1
