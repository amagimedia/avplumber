"""TCP loopback through the native fabric nodes, including finite delayed playout."""
import json
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest


def free_port(kind=socket.SOCK_STREAM):
    with socket.socket(type=kind) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FabricGraph:
    def __init__(self):
        self.commands = ["queue.plan_capacity * 32"]

    def node(self, kind, group, **params):
        config = dict(type=kind, group=group, name=params.pop("name", kind), auto_restart="off", **params)
        self.commands.append("node.add_create " + json.dumps(config))

    def run(self, binary, tmp_path):
        script = tmp_path / "graph.avplumber"
        script.write_text("\n".join(self.commands) + "\n")
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
        return log.read_text(errors="replace")


def round_trip(binary, tmp_path, source, mode, alignment, codec, layout, container="nut"):
    width, height = 320, 240
    port = free_port()
    output = tmp_path / f"received.{container}"
    graph = FabricGraph()
    node = graph.node
    options = {} if codec == "h264_intra" else {"codec": codec}
    node("fabric_source", "rx", dst="received", provider="tcp", bind_host="127.0.0.1",
         bind_port=port, idle_timeout_ms=750, max_payload_bytes=262144,
         width=width, height=height, real_pixel_format=layout, frame_rate="25/1", **options)
    node("redundancy_selector", "rx", src="received", dst="selected", mode=mode,
         alignment_mode=alignment, playout_delay_frames=4)
    node("fabric_unpack", "rx", src="selected", dst="packets")
    node("mux", "rx", src="packets", dst="muxed", format=container)
    node("output", "rx", src="muxed", url=str(output))
    node("input", "tx", url=str(source), dst="input_packets")
    node("demux", "tx", src="input_packets", routing={"v:0": "video"})
    node("packet_relay", "tx", src="video", dst="relay")
    # Omit codec on the sender so the relay supplies its AVCodecParameters.
    node("fabric_sink", "tx", src="relay", provider="tcp", remote_host="127.0.0.1",
         remote_port=port, stream_id="loop", replica_id=1, width=width, height=height, pixel_format=layout,
         max_payload_bytes=262144)
    graph.commands += ["event.on.node.finished done output", "group.start rx", "group.start tx", "event.wait done"]
    log = graph.run(binary, tmp_path)

    def frames(path):
        hashes = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
                                          "-f", "framemd5", "-"], text=True)
        return [line.split(",")[-1].strip() for line in hashes.splitlines() if not line.startswith("#")]

    expected = frames(source)
    assert len(expected) == 16
    assert frames(output) == expected, log
    return output


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


def test_prores_4444_mov_preserves_profile_and_alpha(tmp_path, fabric_binary):
    if not shutil.which("ffprobe"):
        pytest.skip("requires ffprobe")
    source = tmp_path / "source.mov"
    pattern = ("testsrc2=size=320x240:rate=25,format=yuva444p10le,"
               "geq=lum='p(X,Y)':cb='p(X,Y)':cr='p(X,Y)':a='mod(X+N,256)*4'")
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", pattern, "-frames:v", "16",
                    "-c:v", "prores_ks", "-profile:v", "4444", "-threads", "1", str(source)], check=True)
    output = round_trip(fabric_binary, tmp_path, source, "active_active", "strict", "prores4444",
                        "yuva444p10le", container="mov")
    tag = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                   "-show_entries", "stream=codec_tag_string", "-of", "csv=p=0",
                                   str(output)], text=True)
    assert tag.strip() == "ap4h"


@pytest.mark.parametrize("mode", ["active_active", "active_standby"])
def test_hevc_key_fill_failover(tmp_path, fabric_binary, mode):
    if not shutil.which("nvidia-smi"):
        pytest.skip("requires an NVIDIA GPU and ffmpeg with HEVC NVENC")

    # Encode the alpha from the same source frame as the fill. Changing the mask
    # every frame makes an off-by-one pairing observable after decoding.
    source = tmp_path / "source.ts"
    ffmpeg = ["ffmpeg", "-nostdin", "-v", "error"]
    pattern = ("testsrc2=size=320x240:rate=25:duration=0.96,format=yuva444p,"
               "geq=lum='p(X,Y)':cb='p(X,Y)':cr='p(X,Y)':a='mod(X+7*N,256)'")
    split = "[0:v]split[a][b];[a]format=yuv444p[fill];[b]alphaextract"
    fill, key = tmp_path / "fill.ts", tmp_path / "key.ts"
    for path, vf in [(fill, "format=yuv444p"), (key, "alphaextract,scale=in_range=full:out_range=full,format=yuv444p")]:
        subprocess.run([*ffmpeg, "-f", "lavfi", "-i", pattern, "-vf", vf, "-c:v", "hevc_nvenc",
                        "-profile:v", "rext", "-preset:v", "p1", "-tune:v", "lossless", "-rc:v", "constqp",
                        "-qp:v", "0", "-g:v", "1", "-bf:v", "0", "-muxdelay", "0", str(path)], check=True)
    subprocess.run([*ffmpeg, "-i", str(fill), "-i", str(key), "-map", "0:v", "-map", "1:v",
                    "-c", "copy", "-muxdelay", "0", str(source)], check=True)

    # Only the primary has fill 12. It fails before key 12. The backup has
    # the earlier duplicate frames, key 12, and all subsequent frames.
    replicas = [tmp_path / "primary.ts", tmp_path / "backup.ts"]
    for path, filters in zip(replicas, [
        ["-bsf:v:0", "noise=drop='gt(n,12)'", "-bsf:v:1", "noise=drop='gte(n,12)'"],
        ["-bsf:v:0", "noise=drop='eq(n,12)'"],
    ]):
        subprocess.run([*ffmpeg, "-i", str(source), "-map", "0:v", "-c", "copy",
                        *filters, "-muxdelay", "0", str(path)], check=True)

    port = free_port()
    standby = mode == "active_standby"
    control_port = free_port(socket.SOCK_DGRAM) if standby else 0
    graph = FabricGraph()
    node = graph.node
    node("fabric_source", "rx", dst="fill", alpha_dst="key", bind_host="127.0.0.1",
         bind_port=port, provider="tcp", codec="hevc_intra", width=320, height=240,
         real_pixel_format="yuv444p", alpha_pixel_format="yuv444p", time_base="1/90000",
         frame_rate="25/1", idle_timeout_ms=1500, max_payload_bytes=262144)
    for plane in ("fill", "key"):
        node("redundancy_selector", "rx", name=f"select_{plane}", src=plane, dst=f"selected_{plane}",
             mode=mode, alignment_mode="strict", playout_delay_frames=2, active_timeout_ms=150, repair_grace_ms=120,
             standby_control_port=control_port)
        node("fabric_unpack", "rx", name=f"unpack_{plane}", src=f"selected_{plane}", dst=f"out_{plane}")
    output = tmp_path / "received.nut"
    node("mux", "rx", src=["out_fill", "out_key"], dst="muxed", format="nut")
    node("output", "rx", src="muxed", url=str(output))

    for replica_id, path in enumerate(replicas, 1):
        group = path.stem
        node("input", group, name=f"in_{group}", url=str(path), dst=f"ts_{group}")
        node("realtime<av::Packet>", group, name=f"pace_{group}", src=f"ts_{group}",
             dst=f"paced_{group}", set_pts=False, forward_eof=True)
        node("demux", group, name=f"demux_{group}", src=f"paced_{group}",
             routing={"v:0": f"fill_{group}", "v:1": f"key_{group}"})
        for plane in ("fill", "key"):
            node("packet_relay", group, name=f"relay_{plane}_{group}", src=f"{plane}_{group}",
                 dst=f"encoded_{plane}_{group}")
        sender = {"redundancy_mode": "standby", "control_port": control_port} if standby and replica_id == 2 else {}
        node("fabric_sink", group, name=f"send_{group}", src=f"encoded_fill_{group}",
             alpha_src=f"encoded_key_{group}", provider="tcp", remote_host="127.0.0.1",
             remote_port=port, stream_id="key_fill", replica_id=replica_id, generation=1, width=320, height=240,
             pixel_format="yuv444p", alpha_pixel_format="yuv444p", max_payload_bytes=262144, **sender)
    graph.commands += ["event.on.node.finished done output", "group.start rx", "group.start primary",
                       "group.start backup", "event.wait done"]
    log = graph.run(fabric_binary, tmp_path)

    def frame_hashes(*inputs_and_maps):
        md5 = subprocess.check_output([*ffmpeg, "-copyts", *inputs_and_maps,
                                       "-fps_mode", "passthrough", "-enc_time_base", "1:25",
                                       "-f", "framemd5", "-"], text=True)
        planes = [[], []]
        for line in md5.splitlines():
            if not line.startswith("#"):
                stream, _, pts, _, _, digest = line.split(",")
                planes[int(stream)].append((int(pts), digest.strip()))
        return planes

    expected = frame_hashes("-f", "lavfi", "-i", pattern, "-filter_complex", split + "[key]",
                            "-map", "[fill]", "-map", "[key]")
    for plane in expected:
        assert [pts for pts, _ in plane] == list(range(24))
    assert len({digest for _, digest in expected[1]}) == 24
    assert frame_hashes("-i", str(output), "-map", "0:v", "-filter:v:1", "extractplanes=y") == expected
    live_log = log.split("fabric_source session ended")[0]
    if standby:
        assert "fabric_sink repair sent" in live_log
        assert "fabric_sink promoted" in live_log
        assert live_log.count("elected active replica_id=2") >= 2
    else:
        assert "selected standby replica_id=2" in live_log
