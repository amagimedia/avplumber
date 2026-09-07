import json
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
from playback_backend import backend_api


def _transcode_test_source(tmp_path, frame_count, fps=30, backend="nvidia", source_gop="interframe"):
    source = tmp_path / "source.mp4"
    output = tmp_path / "replay.ts"
    codec_options = (["-g", "1", "-bf", "0"] if source_gop == "intra" else
                     ["-g", "60", "-bf", "3", "-x264-params", "b-adapt=0"])
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi",
        "-i", f"testsrc2=size=640x360:rate={fps}", "-frames:v", str(frame_count),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", *codec_options, str(source),
    ], check=True)
    source_probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=pict_type", "-of", "json", str(source),
    ], capture_output=True, text=True, check=True)
    frame_types = {frame["pict_type"] for frame in json.loads(source_probe.stdout)["frames"]}
    if source_gop == "intra":
        assert frame_types == {"I"}
    else:
        assert "B" in frame_types
    if backend == "cpu":
        runner = [sys.executable, "-c", (
            "import sys; "
            f"sys.path[:0] = {[str(Path(__file__).parent), str(Path(__file__).parents[1])]!r}; "
            "import transcode; from playback_backend import backend_api; "
            "build = transcode.build_transcode_application; "
            "transcode.build_transcode_application = lambda config: build(config, api=backend_api('cpu')); "
            "raise SystemExit(transcode.main(sys.argv[1:]))"
        )]
    else:
        runner = [sys.executable, str(Path(__file__).parents[1] / "transcode.py")]
    subprocess.run(runner + [
        "--input", str(source), "--output", str(output), "--fps", str(fps),
    ], check=True)
    return output


@pytest.mark.parametrize("frame_count", [30, 60, 73])
@pytest.mark.parametrize("source_gop", ["intra", "interframe"])
def test_finite_transcode_publishes_every_all_intra_packet(tmp_path, frame_count, source_gop, playback_backend):
    output = _transcode_test_source(tmp_path, frame_count, backend=playback_backend, source_gop=source_gop)

    packet_data = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "packet=flags", "-of", "json", str(output),
    ], capture_output=True, text=True, check=True)
    packets = json.loads(packet_data.stdout)["packets"]
    seek_entries = read_seek_table(Path(f"{output}+seek"))

    assert len(packets) == frame_count
    assert len(seek_entries) == len(packets)
    assert all("K" in packet["flags"] for packet in packets)


def test_player_emits_configured_rtp_without_janus(tmp_path, playback_backend):
    output = _transcode_test_source(tmp_path, 90, backend=playback_backend)
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
    ), api=backend_api(playback_backend))
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
