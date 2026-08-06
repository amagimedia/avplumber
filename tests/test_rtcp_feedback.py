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
