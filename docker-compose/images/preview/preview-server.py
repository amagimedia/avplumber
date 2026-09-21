from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from collections import OrderedDict, deque
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request


JANUS_REST = os.environ.get(
    "JANUS_REST",
    f"http://127.0.0.1:{os.environ.get('JANUS_HTTP_PORT', '8088')}/janus",
)
PREVIEW_PORT = int(os.environ.get("JANUS_PREVIEW_PORT", "8080"))


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
            if seen >= now - 300 and len(self.receivers) <= 16:
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


class PreviewHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path == "/receiver-stats":
            payload = json.dumps(receiver_samples.snapshot(), allow_nan=False).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path.startswith("/janus"):
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
        if self.path.startswith("/janus"):
            self._proxy_janus()
            return
        self.send_error(404, "File not found")

    def _proxy_janus(self):
        suffix = self.path[len("/janus"):]
        upstream_url = f"{JANUS_REST}{suffix}"
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
            with urllib.request.urlopen(request, timeout=65) as response:
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
    ThreadingHTTPServer(("0.0.0.0", PREVIEW_PORT), PreviewHandler).serve_forever()
