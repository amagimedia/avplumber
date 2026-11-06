from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from collections import OrderedDict, deque
import json
import math
import os
import re
import selectors
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


JANUS_REST = os.environ.get(
    "JANUS_REST",
    f"http://127.0.0.1:{os.environ.get('JANUS_HTTP_PORT', '8088')}/janus",
)
JANUS_WS = os.environ.get("JANUS_WS", f"ws://127.0.0.1:{os.environ.get('JANUS_WS_PORT', '8188')}")
MIXER_STATE_URL = os.environ.get("MIXER_STATE_URL", "")
PREVIEW_PORT = int(os.environ.get("JANUS_PREVIEW_PORT", "8080"))
# The mixer's state is served from its last good reply for this long while the mixer restarts
# (a deploy or a new setup), so players keep their outputs and latency instead of 502/503s.
MIXER_STATE_STALE_S = float(os.environ.get("MIXER_STATE_STALE_S", "120"))
PREVIEW_BIND = os.environ.get("JANUS_PREVIEW_BIND", "0.0.0.0")


class ReceiverSamples:
    """Bounded, ephemeral playback counters; no addresses, SDP or media content."""
    fields = set("""epoch fps jitterBufferMs decodeMs rttMs framesDecoded framesRendered
        framesDropped freezeCount pauseCount totalFreezesDuration totalPausesDuration
        packetsReceived packetsLost retransmittedPacketsReceived nackCount pliCount
        bytesReceived frameWidth frameHeight presentationStalls maxPresentationGapMs
        frameAgeMs""".split())

    def __init__(self, clock=time.time):
        self.clock = clock
        self.receivers = OrderedDict()
        self.lock = threading.Lock()

    def _prune(self, now):
        while self.receivers:
            _, (seen, _) = next(iter(self.receivers.items()))
            if seen >= now - 300 and len(self.receivers) <= 64:
                break
            self.receivers.popitem(last=False)

    def add(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("expected an object")
        receiver, sample = payload.get("receiver"), payload.get("sample")
        if (not isinstance(receiver, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", receiver)
                or not isinstance(sample, dict)):
            raise ValueError("invalid receiver sample")
        clean = {k: v for k, v in sample.items() if k in self.fields
                 and (v is None or type(v) in (int, float) and math.isfinite(v) and abs(v) < 1e16)}
        for key in ("visible", "paused"):
            if type(sample.get(key)) is bool:
                clean[key] = sample[key]
        if isinstance(sample.get("codec"), str):
            clean["codec"] = sample["codec"][:64]
        if not clean:
            raise ValueError("no playback counters")
        with self.lock:
            now = self.clock()
            self._prune(now)
            history = self.receivers.get(receiver, (0, deque(maxlen=300)))[1]
            history.append({"at": now, **clean})
            self.receivers[receiver] = (now, history)
            self.receivers.move_to_end(receiver)
            self._prune(now)

    def snapshot(self):
        with self.lock:
            self._prune(self.clock())
            return [{"receiver": receiver, "samples": list(history)}
                    for receiver, (_, history) in self.receivers.items()]


receiver_samples = ReceiverSamples()


class LastGood:
    """The last successful reply of an upstream, for riding out its restarts."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.reply = None                       # (at, content type, payload)

    def keep(self, content_type, payload):
        with self.lock:
            self.reply = (self.clock(), content_type, payload)

    def get(self, max_age):
        with self.lock:
            if self.reply is None or self.clock() - self.reply[0] > max_age:
                return None
            return self.reply[1], self.reply[2], self.clock() - self.reply[0]


mixer_state = LastGood()


class PreviewHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path == "/mixer-state" and MIXER_STATE_URL:
            self._mixer_state()
            return
        if self.path == "/metrics.json" and MIXER_STATE_URL:
            # Embedded in the mixer: no graph-UI measurements to configure. No content tells the
            # player to read the mixer's state instead, without a 404 on every page load.
            self.send_response(204)
            self.end_headers()
            return
        if self.path == "/receiver-stats":
            payload = json.dumps(receiver_samples.snapshot(), allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path.split("?", 1)[0] == "/janus" or self.path.startswith("/janus/"):
            if self.headers.get("Upgrade", "").lower() == "websocket":
                self._tunnel_janus()
            else:
                self._proxy_janus()
            return
        super().do_GET()

    def do_POST(self):
        if self.path == "/receiver-stats":
            try:
                length = int(self.headers.get("content-length", "0"))
                if not 0 < length <= 4096:
                    self.send_error(413, "Receiver sample must be at most 4096 bytes")
                    return
                receiver_samples.add(json.loads(self.rfile.read(length)))
            except (ValueError, TypeError):
                self.send_error(400, "Invalid receiver sample")
                return
            self.send_response(204)
            self.end_headers()
            return
        if self.path.split("?", 1)[0] == "/janus" or self.path.startswith("/janus/"):
            self._proxy_janus()
            return
        self.send_error(404, "File not found")

    def _mixer_state(self):
        """The mixer's state, or while it restarts its last good state marked stale."""
        try:
            with urllib.request.urlopen(MIXER_STATE_URL, timeout=3) as response:
                payload = response.read()
                content_type = response.headers.get("content-type", "application/json")
                mixer_state.keep(content_type, payload)
                self._send_payload(response.status, content_type, payload)
                return
        except urllib.error.HTTPError as error:
            failure = (error.code, error.headers.get("content-type", "text/plain"), error.read())
        except Exception as error:
            failure = (502, "text/plain", str(error).encode())
        kept = mixer_state.get(MIXER_STATE_STALE_S) if failure[0] >= 500 else None
        if kept is None:
            self._send_payload(*failure)
            return
        content_type, payload, age = kept
        self._send_payload(200, content_type, payload, {"X-Mixer-State": f"stale; age={age:.0f}"})

    def _send_payload(self, status, content_type, payload, headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)

    def _proxy_janus(self):
        suffix = self.path[len("/janus"):]
        upstream_url = f"{JANUS_REST}{suffix}"
        self._proxy(upstream_url)

    def _tunnel_janus(self):
        """The player's WebSocket: replays its handshake to JANUS_WS, then relays the bytes of both
        directions, Janus's reply to the handshake included, until either side closes."""
        self.close_connection = True
        janus = urllib.parse.urlsplit(JANUS_WS)
        try:
            upstream = socket.create_connection((janus.hostname, janus.port or 80), timeout=10)
        except OSError as error:
            self.send_error(502, str(error))
            return
        handshake = [f"GET {janus.path or '/'} HTTP/1.1", f"Host: {janus.netloc}",
                     "Upgrade: websocket", "Connection: Upgrade"]
        handshake += [f"{name}: {value}" for name, value in self.headers.items()
                      if name.lower().startswith("sec-websocket-") or name.lower() == "origin"]
        with upstream, selectors.DefaultSelector() as selector:
            # Bound a stalled write without treating an otherwise idle socket as a failure.
            self.connection.settimeout(10)
            # self.connection, not rfile: a WebSocket client sends nothing before the reply to its
            # handshake, so rfile has buffered no more than the request.
            selector.register(self.connection, selectors.EVENT_READ, upstream)
            selector.register(upstream, selectors.EVENT_READ, self.connection)
            try:
                upstream.sendall("\r\n".join(handshake + ["", ""]).encode("latin-1"))
                while True:
                    for key, _ in selector.select():
                        data = key.fileobj.recv(65536)
                        if not data:
                            return
                        key.data.sendall(data)
            except OSError:
                pass

    def _proxy(self, upstream_url, timeout=65):
        body = None
        if self.command in ("POST", "PUT", "PATCH"):
            length = int(self.headers.get("content-length", "0") or "0")
            body = self.rfile.read(length) if length else b""
        headers = {
            "content-type": self.headers.get("content-type", "application/json"),
        }
        request = urllib.request.Request(
            upstream_url,
            data=body,
            headers=headers,
            method=self.command,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
                self.send_response(response.status)
                self.send_header(
                    "Content-Type",
                    response.headers.get("content-type", "application/json"),
                )
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as error:
            payload = error.read()
            self.send_response(error.code)
            self.send_header("Content-Type", error.headers.get("content-type", "text/plain"))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        except Exception as error:
            payload = str(error).encode()
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)


if __name__ == "__main__":
    ThreadingHTTPServer((PREVIEW_BIND, PREVIEW_PORT), PreviewHandler).serve_forever()
