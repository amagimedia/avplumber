import sys
from pathlib import Path

GRAPH_DIR = Path(__file__).resolve().parents[1] / "graph"
sys.path.insert(0, str(GRAPH_DIR))

from dmabuf_output_config import resolve_output_config


def test_rtp_defaults_keep_janus_options():
    assert resolve_output_config({}, "rtp://127.0.0.1:5004", 96, 0x41565001) == (
        "rtp",
        "rtp://127.0.0.1:5004",
        {"payload_type": 96, "rtpflags": "skip_rtcp", "ssrc": 0x41565001},
    )


def test_file_output_does_not_inherit_rtp_options():
    assert resolve_output_config(
        {"OUTPUT_FORMAT": "mpegts", "OUTPUT_URL": "/capture/grid.ts"},
        "rtp://127.0.0.1:5004",
        96,
        0x41565001,
    ) == ("mpegts", "/capture/grid.ts", {})
