"""Encoded-output qualification: build the real mixer, record a few seconds, probe
what NVENC wrote.

Run A: SDR NV12 canvas, static SDR bars (v210) -> H.264 reference.
Run B: HLG P210 canvas, the same SDR bars fullscreen + static HLG bars as a PIP ->
       HLG HEVC, PQ HEVC (HDR10 static metadata) and a mobius-0.9 SDR H.264 rendition.

Asserts codec, profile, pixel format and VUI on every leg, the HDR10 mastering
display / MaxCLL SEIs on the PQ leg, and that the SDR round trip through the HLG
canvas (SDR -> HLG -> SDR) reproduces run A within a few codes of luma and
saturation over the region without the PIP. Needs the CUDA avplumber module and
the custom ffmpeg (hevc/h264 software decoders for probing).
"""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v210_fixture import write_fixture  # noqa: E402

W, H, FPS = 1280, 720, 60
PIP = {"x": 760, "y": 40, "w": 480, "h": 270}
CROP = f"crop={W // 2}:{H}:0:0"          # left half: never covered by the PIP


def run_mixer(repo, cfg_path, outputs, seconds, timeout):
    """Run mixer.py until every output has grown past *seconds* of content, then kill it."""
    env = {**os.environ, "PYTHONPATH": f"{repo}:{os.environ.get('PYTHONPATH', '')}"}
    proc = subprocess.Popen([sys.executable, str(repo / "demos/mixer/mixer.py"), "--config", str(cfg_path),
                             "--output", str(Path(cfg_path).with_suffix(".unused.ts")), "--remote-control-port", "0"],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + timeout
    started = None
    try:
        while time.monotonic() < deadline:
            time.sleep(1)
            if proc.poll() is not None:
                raise AssertionError("mixer exited early:\n" + proc.stdout.read()[-3000:])
            if all(p.exists() and p.stat().st_size > 0 for p in outputs):
                started = started or time.monotonic()
                if time.monotonic() - started >= seconds:
                    return
        raise AssertionError(f"mixer did not produce {outputs} within {timeout}s")
    finally:
        proc.send_signal(signal.SIGKILL)   # finite-graph shutdown may hang; the files are complete streams
        proc.wait()


def probe(ffmpeg, path):
    out = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    line = next((l for l in out.splitlines() if "Video:" in l), "")
    assert line, out
    return line


def side_data(ffmpeg, path):
    out = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path), "-frames:v", "1", "-vf", "showinfo", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    return "\n".join(l for l in out.splitlines() if "side data" in l.lower() or "MaxCLL" in l)


def stats(ffmpeg, path, seconds=1.0):
    out = subprocess.run([ffmpeg, "-hide_banner", "-ss", "1", "-t", f"{seconds}", "-i", str(path),
                          "-vf", f"{CROP},signalstats,metadata=print:file=-", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    y = [float(v) for v in re.findall(r"signalstats\.YAVG=([0-9.]+)", out)]
    s = [float(v) for v in re.findall(r"signalstats\.SATAVG=([0-9.]+)", out)]
    assert y and s, out[-1500:]
    return sum(y) / len(y), sum(s) / len(s)


def config(root, canvas, renditions):
    return {
        "canvas": {"width": W, "height": H, "fps": FPS, **canvas},
        "sources": [
            {"id": "sdr", "kind": "v210", "path": str(root / "sdr.v210"), "width": W, "height": H, "color": "sdr"},
            {"id": "hlg", "kind": "v210", "path": str(root / "hlg.v210"), "width": W, "height": H, "color": "hlg"},
        ],
        "scenes": [{"id": "full", "items": [
            {"source": "sdr", "dst": {"x": 0, "y": 0, "w": W, "h": H}},
            *([{"source": "hlg", "dst": PIP}] if canvas["color"] != "sdr" else [])]}],
        "renditions": renditions,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ffmpeg", default="/usr/local/bin/ffmpeg")
    p.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    p.add_argument("--seconds", type=float, default=4)
    p.add_argument("--timeout", type=float, default=120)
    args = p.parse_args()
    repo = Path(args.repo)
    with tempfile.TemporaryDirectory(prefix="avp-qual-") as tmp:
        root = Path(tmp)
        # Static content: one pattern repeated, so the two runs compare without frame alignment.
        write_fixture(root / "sdr.v210", W, H, 1, family="sdr8")
        write_fixture(root / "hlg.v210", W, H, 1, family="hlg")
        for name in ("sdr", "hlg"):
            data = (root / f"{name}.v210").read_bytes()
            (root / f"{name}.v210").write_bytes(data * 120)

        ref = root / "ref.ts"
        cfg_a = root / "a.json"
        cfg_a.write_text(json.dumps(config(root, {"working_format": "nv12", "color": "sdr"}, [
            {"id": "ref", "target": str(ref), "codec": "h264_nvenc", "bitrate_kbps": 6000}])))
        run_mixer(repo, cfg_a, [ref], args.seconds, args.timeout)
        line = probe(args.ffmpeg, ref)
        assert "h264" in line and "yuv420p(tv, bt709" in line, line
        print("PASS SDR canvas -> H.264 BT.709:", line.split("Video: ")[1][:60], flush=True)

        legs = {k: root / f"{k}.ts" for k in ("hlg", "pq", "sdr")}
        cfg_b = root / "b.json"
        cfg_b.write_text(json.dumps(config(root, {"working_format": "p210le", "color": "hlg"}, [
            {"id": "hlg", "target": str(legs["hlg"]), "codec": "hevc_nvenc", "bitrate_kbps": 8000},
            {"id": "pq", "target": str(legs["pq"]), "codec": "hevc_nvenc", "color": "pq", "max_fall": 400,
             "bitrate_kbps": 8000},
            {"id": "sdr", "target": str(legs["sdr"]), "codec": "h264_nvenc", "tonemap": "mobius",
             "tonemap_param": 0.9, "bitrate_kbps": 6000}])))
        run_mixer(repo, cfg_b, list(legs.values()), args.seconds, args.timeout)

        line = probe(args.ffmpeg, legs["hlg"])
        assert "hevc (Main 10)" in line and "yuv420p10le(tv, bt2020nc/bt2020/arib-std-b67" in line, line
        print("PASS HLG canvas -> HEVC Main 10, BT.2020 HLG VUI", flush=True)
        line = probe(args.ffmpeg, legs["pq"])
        assert "hevc (Main 10)" in line and "yuv420p10le(tv, bt2020nc/bt2020/smpte2084" in line, line
        sd = side_data(args.ffmpeg, legs["pq"])
        assert "Mastering display" in sd and "MaxCLL=1000" in sd and "MaxFALL=400" in sd, sd
        print("PASS PQ rendition -> HEVC Main 10, PQ VUI, HDR10 mastering display + MaxCLL/MaxFALL SEIs", flush=True)
        line = probe(args.ffmpeg, legs["sdr"])
        assert "h264" in line and "yuv420p(tv, bt709" in line, line

        y_ref, s_ref = stats(args.ffmpeg, ref)
        y_out, s_out = stats(args.ffmpeg, legs["sdr"])
        assert abs(y_out - y_ref) <= 3 and abs(s_out - s_ref) <= 3, \
            f"SDR round trip drifted: luma {y_ref:.1f} -> {y_out:.1f}, saturation {s_ref:.1f} -> {s_out:.1f}"
        print(f"PASS SDR -> HLG canvas -> SDR round trip: luma {y_ref:.1f}->{y_out:.1f} "
              f"saturation {s_ref:.1f}->{s_out:.1f}", flush=True)


if __name__ == "__main__":
    main()
