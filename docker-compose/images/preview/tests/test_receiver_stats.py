import importlib.util
import math
from pathlib import Path
import json
import threading
import urllib.error
import urllib.request

import pytest


spec = importlib.util.spec_from_file_location("preview_server", Path(__file__).parents[1] / "preview-server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


def test_receiver_history_is_bounded_and_expires():
    now = [1000]
    samples = server.ReceiverSamples(clock=lambda: now[0])
    for i in range(350):
        samples.add({"receiver": "viewer", "sample": {"framesDecoded": i}})
    history = samples.snapshot()[0]["samples"]
    assert len(history) == 300 and history[0]["framesDecoded"] == 50
    for i in range(20):
        samples.add({"receiver": f"viewer-{i}", "sample": {"freezeCount": 0}})
    assert len(samples.snapshot()) == 16
    assert samples.snapshot()[0]["receiver"] == "viewer-4"
    now[0] += 301
    assert samples.snapshot() == []


def test_only_finite_playback_metrics_are_retained():
    samples = server.ReceiverSamples(clock=lambda: 1000)
    samples.add({"receiver": "viewer", "sample": {
        "fps": 30, "visible": True, "paused": False, "codec": "video/H265",
        "freezeCount": None, "packetsLost": -1, "jitterBufferMs": math.inf,
        "decodeMs": math.nan, "bytesReceived": "bad", "at": 9,
        "address": "unretained", "sdp": "unretained", "media": "unretained",
    }})
    assert samples.snapshot()[0]["samples"] == [{
        "at": 1000, "fps": 30, "visible": True, "paused": False, "codec": "video/H265",
        "freezeCount": None, "packetsLost": -1,
    }]


@pytest.mark.parametrize("payload", [None, [], {}, {"receiver": "bad id", "sample": {"fps": 30}},
    {"receiver": "v", "sample": []}, {"receiver": "v", "sample": {"unknown": 1}}])
def test_bad_receiver_samples_are_rejected(payload):
    with pytest.raises(ValueError):
        server.ReceiverSamples().add(payload)


def test_receiver_http_endpoint(monkeypatch):
    monkeypatch.setattr(server, "receiver_samples", server.ReceiverSamples())
    http = server.ThreadingHTTPServer(("127.0.0.1", 0), server.PreviewHandler)
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{http.server_port}/receiver-stats"
    try:
        body = json.dumps({"receiver": "probe", "sample": {"fps": 30, "freezeCount": 0}}).encode()
        with urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=2) as response:
            assert response.status == 204
        with urllib.request.urlopen(url, timeout=2) as response:
            assert response.headers["Cache-Control"] == "no-store"
            snapshot = json.load(response)
        assert snapshot[0]["samples"][0]["fps"] == 30
        for body, status in [(b"invalid", 400), (b"x" * 4097, 413)]:
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=2)
            assert error.value.code == status
    finally:
        http.shutdown()
        http.server_close()
        worker.join(timeout=2)
