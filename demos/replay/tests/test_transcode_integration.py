import importlib.util
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from replay import (
    JanusVideoConfig,
    PlayerConfig,
    ReplaySlotConfig,
    build_player_application,
    read_seek_table,
)


def _gpu_prerequisites():
    missing = [name for name in ("ffmpeg", "ffprobe", "nvidia-smi") if shutil.which(name) is None]
    if importlib.util.find_spec("_avplumber") is None:
        missing.append("CUDA/NVENC _avplumber Python module")
    if missing:
        pytest.skip("requires NVIDIA integration host with " + ", ".join(missing))
    probe = subprocess.run(["nvidia-smi"], capture_output=True, check=False)
    if probe.returncode:
        pytest.skip("requires a working NVIDIA driver")


def _transcode_test_source(tmp_path, frame_count):
    source = tmp_path / "source.mp4"
    output = tmp_path / "replay.ts"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
        "-i", "testsrc2=size=640x360:rate=30", "-frames:v", str(frame_count),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
    ], check=True)
    subprocess.run([
        sys.executable, str(Path(__file__).parents[1] / "transcode.py"),
        "--input", str(source), "--output", str(output), "--fps", "30",
    ], check=True)
    return output


@pytest.mark.parametrize("frame_count", [30, 60, 73])
def test_finite_transcode_publishes_every_all_intra_packet(tmp_path, frame_count):
    _gpu_prerequisites()
    output = _transcode_test_source(tmp_path, frame_count)

    packet_data = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=flags", "-of", "json", str(output),
    ], capture_output=True, text=True, check=True)
    packets = json.loads(packet_data.stdout)["packets"]
    seek_entries = read_seek_table(Path(f"{output}+seek"))

    assert len(packets) == frame_count
    assert len(seek_entries) == len(packets)
    assert all("K" in packet["flags"] for packet in packets)


def test_player_emits_configured_rtp_without_janus(tmp_path):
    _gpu_prerequisites()
    output = _transcode_test_source(tmp_path, 90)
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(5)
    video_port = receiver.getsockname()[1]
    application = build_player_application(PlayerConfig(
        ReplaySlotConfig(output),
        JanusVideoConfig(
            host="127.0.0.1",
            video_port=video_port,
            payload_type=97,
            ssrc=0x12345678,
        ),
    ))
    try:
        application.start()
        packet, _peer = receiver.recvfrom(65535)
    finally:
        application.stop()
        receiver.close()

    assert len(packet) >= 12
    assert packet[0] >> 6 == 2
    assert packet[1] & 0x7f == 97
    assert int.from_bytes(packet[8:12], "big") == 0x12345678
