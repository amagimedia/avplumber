"""TCP loopback through the native fabric nodes, including finite delayed playout."""
import json
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest


def round_trip(binary, tmp_path, source, mode, alignment, codec, layout):
    width, height = 320, 240
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    output = tmp_path / "received.nut"
    script = tmp_path / "graph.avplumber"
    commands = ["queue.plan_capacity * 32"]

    def node(kind, group, **params):
        config = dict(type=kind, group=group, name=params.pop("name", kind), auto_restart="off", **params)
        commands.append("node.add_create " + json.dumps(config))

    options = {} if codec == "h264_intra" else {"codec": codec}
    node("fabric_source", "rx", dst="received", provider="tcp", bind_host="127.0.0.1",
         bind_port=port, idle_timeout_ms=750, max_payload_bytes=262144,
         width=width, height=height, real_pixel_format=layout, frame_rate="25/1", **options)
    node("redundancy_selector", "rx", src="received", dst="selected", mode=mode,
         alignment_mode=alignment, playout_delay_frames=4)
    node("fabric_unpack", "rx", src="selected", dst="packets")
    node("mux", "rx", src="packets", dst="muxed", format="nut")
    node("output", "rx", src="muxed", url=str(output))
    node("input", "tx", url=str(source), dst="input_packets")
    node("demux", "tx", src="input_packets", routing={"v:0": "video"})
    node("packet_relay", "tx", src="video", dst="relay")
    # Omit codec on the sender so the relay supplies its AVCodecParameters.
    node("fabric_sink", "tx", src="relay", provider="tcp", remote_host="127.0.0.1",
         remote_port=port, stream_id="loop", replica_id=1, width=width, height=height, pixel_format=layout,
         max_payload_bytes=262144)
    commands += ["event.on.node.finished done output", "group.start rx", "group.start tx", "event.wait done"]
    script.write_text("\n".join(commands) + "\n")
    log = tmp_path / "graph.log"
    with log.open("w") as stream:
        proc = subprocess.Popen([str(binary), "-s", str(script)], stdout=stream, stderr=stream)
        try:
            deadline = time.monotonic() + 20
            while "Finished parsing file" not in log.read_text(errors="replace"):
                assert proc.poll() is None and time.monotonic() < deadline, log.read_text(errors="replace")
                time.sleep(0.05)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def frames(path):
        hashes = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
                                          "-f", "framemd5", "-"], text=True)
        return [line.split(",")[-1].strip() for line in hashes.splitlines() if not line.startswith("#")]

    expected = frames(source)
    assert len(expected) == 16
    assert frames(output) == expected, log.read_text(errors="replace")


@pytest.fixture
def fabric_binary():
    binary = Path(__file__).resolve().parents[1] / "avplumber"
    if not binary.is_file() or not shutil.which("ldd") or not shutil.which("ffmpeg"):
        pytest.skip("requires a Linux avplumber binary and ffmpeg")
    if "libfabric.so" not in subprocess.check_output(["ldd", str(binary)], text=True):
        pytest.skip("requires HAVE_LIBFABRIC=1")
    return binary


@pytest.mark.parametrize(("mode", "alignment"), [
    ("active_active", "strict"), ("active_active", "wallclock"), ("active_standby", "strict"),
])
def test_delayed_fabric_eof(tmp_path, mode, alignment, fabric_binary):
    source = tmp_path / "source.nut"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25",
                    "-frames:v", "16", "-c:v", "mjpeg", "-pix_fmt", "yuvj444p", "-threads", "1",
                    str(source)], check=True)
    round_trip(fabric_binary, tmp_path, source, mode, alignment, "jpeg", "yuvj444p")


@pytest.mark.parametrize(("encoder", "codec", "layout"), [
    ("h264_nvenc", "h264_intra", "yuv420p"),
    ("hevc_nvenc", "hevc_intra", "yuv444p"),
    ("hevc_nvenc", "hevc_intra", "yuv444p10le"),
])
def test_nvenc_fabric_round_trip(tmp_path, fabric_binary, encoder, codec, layout):
    if not shutil.which("nvidia-smi") or not shutil.which("ffprobe"):
        pytest.skip("requires an NVIDIA GPU and ffprobe")
    source = tmp_path / "source.ts"
    profile = ["-profile:v", "rext", "-tune", "lossless"] if codec == "hevc_intra" else []
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25",
                    "-frames:v", "16", "-pix_fmt", layout, "-c:v", encoder, *profile,
                    "-preset", "p1", "-rc", "constqp", "-qp", "0", "-g", "25", "-bf", "0",
                    "-forced-idr", "1", "-force_key_frames", "expr:gte(n,0)", "-f", "mpegts", str(source)],
                   check=True)
    formats = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                       "-show_entries", "stream=pix_fmt", "-of", "csv=p=0", str(source)], text=True)
    assert layout in [row.rstrip(",") for row in formats.splitlines()]
    round_trip(fabric_binary, tmp_path, source, "active_active", "strict", codec, layout)


def test_key_stream_metadata_before_first_packet(fabric_binary, cpp_library_binary):
    subprocess.run([cpp_library_binary("test_fabric_metadata")], check=True, timeout=15)
