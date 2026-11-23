import importlib.util
from pathlib import Path
import sys


_MODULE_PATH = Path(__file__).resolve().parents[1] / "pyplumber" / "rtcp_feedback.py"
_SPEC = importlib.util.spec_from_file_location("rtcp_feedback", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

build_rtcp_fir = _MODULE.build_rtcp_fir
build_rtcp_pli = _MODULE.build_rtcp_pli
rtcp_keyframe_requests = _MODULE.rtcp_keyframe_requests


def test_detects_psfb_pli():
    packet = build_rtcp_pli(sender_ssrc=1, media_ssrc=0x41565001)

    assert rtcp_keyframe_requests(packet) == ["pli"]


def test_detects_psfb_fir():
    packet = build_rtcp_fir(sender_ssrc=1, media_ssrc=0x41565001, sequence=7)

    assert rtcp_keyframe_requests(packet) == ["fir"]


def test_detects_compound_pli_and_fir():
    packet = (
        build_rtcp_pli(sender_ssrc=1, media_ssrc=0x41565001)
        + build_rtcp_fir(sender_ssrc=1, media_ssrc=0x41565001, sequence=8)
    )

    assert rtcp_keyframe_requests(packet) == ["pli", "fir"]


def test_ignores_malformed_packet():
    assert rtcp_keyframe_requests(b"\x80\xce\x00") == []


def _send_pli(listener):
    import socket
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sender.sendto(build_rtcp_pli(sender_ssrc=1, media_ssrc=listener.media_ssrc), ("127.0.0.1", listener.local_port))
    finally:
        sender.close()


def test_listeners_share_one_thread_and_each_gets_its_requests():
    import threading
    import time

    janus = __import__("socket").socket(__import__("socket").AF_INET, __import__("socket").SOCK_DGRAM)
    janus.bind(("127.0.0.1", 0))
    janus.settimeout(2)
    requests = {}
    listeners = [
        _MODULE.RtcpFeedbackListener(bind_host="127.0.0.1", bind_port=0, janus_host="127.0.0.1",
                                     janus_rtcp_port=janus.getsockname()[1], media_ssrc=0x41565000 + i,
                                     on_keyframe_request=lambda request, i=i: requests.setdefault(i, []).append(request))
        for i in range(3)]
    try:
        for listener in listeners:
            listener.start()
        pollers = [t for t in threading.enumerate() if t.name == "janus-rtcp-feedback"]
        assert len(pollers) == 1
        janus.recvfrom(2048)   # the announcements still go out
        for listener in listeners:
            _send_pli(listener)
        deadline = time.monotonic() + 2
        while len(requests) < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert requests == {0: ["pli"], 1: ["pli"], 2: ["pli"]}
        assert [listener.pli_requests for listener in listeners] == [1, 1, 1]
    finally:
        for listener in listeners:
            listener.stop()
        janus.close()
    time.sleep(0.1)
    assert not [t for t in threading.enumerate() if t.name == "janus-rtcp-feedback"]
    # A listener started after the last one stopped gets a new poller.
    again = _MODULE.RtcpFeedbackListener(bind_host="127.0.0.1", bind_port=0, janus_host="127.0.0.1",
                                         janus_rtcp_port=9, media_ssrc=1, on_keyframe_request=lambda _r: None)
    again.start()
    try:
        assert [t.name for t in threading.enumerate()].count("janus-rtcp-feedback") == 1
    finally:
        again.stop()
