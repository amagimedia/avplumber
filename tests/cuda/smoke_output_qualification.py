"""Encoded-output qualification: build the real mixer, record a few seconds, probe
what NVENC wrote.

Run A: SDR NV12 canvas, static SDR bars (v210) -> H.264 reference.
Run B: HLG P210 canvas, the same SDR bars fullscreen + static HLG bars as a PIP ->
       HLG HEVC, PQ HEVC (HDR10 static metadata) and a mobius-0.9 SDR H.264 rendition.

Asserts codec, profile, pixel format and VUI on every leg, the HDR10 mastering
display / MaxCLL SEIs on the PQ leg, and that the SDR round trip through the HLG
canvas (SDR -> HLG -> SDR) reproduces run A's bar colours within a few codes. Needs the CUDA avplumber module and
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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v210_fixture import pack_v210, write_fixture  # noqa: E402

W, H, FPS = 1280, 720, 60
BAR = W // 8                              # eight vertical colour bars
PIP = {"x": 760, "y": 40, "w": 480, "h": 270}
PIP_BOTTOM = PIP["y"] + PIP["h"] + 40     # bar centres are sampled below the HLG insert
TOLERANCE = 8                             # codes, per channel, at a bar centre


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


def bar_centres(ffmpeg, path):
    """Mean (Y, U, V) of the centre of each colour bar, sampled below the PIP so the
    HLG insert never contributes; edges are excluded because every chroma resample
    on the HLG path blurs them a little, which is expected and not a colour error."""
    centres = []
    for i in range(8):
        crop = f"crop={BAR // 4}:{H - PIP_BOTTOM}:{i * BAR + BAR // 2 - BAR // 8}:{PIP_BOTTOM}"
        run = subprocess.run([ffmpeg, "-hide_banner", "-ss", "1", "-i", str(path), "-frames:v", "1",
                              "-vf", f"{crop},signalstats,metadata=print:file=-", "-f", "null", "-"],
                             capture_output=True, text=True)
        out = run.stdout + run.stderr          # metadata=print:file=- writes to stdout
        found = [float(re.search(rf"signalstats\.{k}AVG=([0-9.]+)", out).group(1)) for k in "YUV"]
        centres.append(found)
    return centres


def write_bars(path, width, height, frames):
    """Flat 75% BT.709 colour bars as limited-range 10-bit v210: flat regions survive
    the chroma resampling of the HLG round trip, unlike the pixel-frequency fixtures."""
    rgb = np.array([(1, 1, 1), (1, 1, 0), (0, 1, 1), (0, 1, 0), (1, 0, 1), (1, 0, 0), (0, 0, 1), (0, 0, 0)],
                   dtype=np.float64) * 0.75
    cols = np.repeat(rgb, width // 8, axis=0)[:width]
    r, g, b = (np.tile(cols[:, i], (height, 1)) for i in range(3))
    yp = 0.2126 * r + 0.7152 * g + 0.0722 * b
    y = np.rint(64 + 876 * yp).astype("<u2")
    u = np.rint(896 * (b - yp) / 1.8556 + 512).astype("<u2")[:, 0::2]
    v = np.rint(896 * (r - yp) / 1.5748 + 512).astype("<u2")[:, 0::2]
    Path(path).write_bytes(pack_v210((y, u, v)) * frames)


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
    p.add_argument("--sdr-tonemap", default="mobius", help="operator for the SDR rendition off the HLG canvas")
    p.add_argument("--knee", type=float, default=0.9, help="tonemap_param for that rendition (0 = filter default)")
    p.add_argument("--keep", help="directory to copy the recordings into for inspection")
    args = p.parse_args()
    repo = Path(args.repo)
    with tempfile.TemporaryDirectory(prefix="avp-qual-") as tmp:
        root = Path(tmp)
        # Static content: one frame repeated, so the two runs compare without frame alignment.
        write_bars(root / "sdr.v210", W, H, 120)
        write_fixture(root / "hlg.v210", W, H, 1, family="hlg")
        (root / "hlg.v210").write_bytes((root / "hlg.v210").read_bytes() * 120)

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
            {"id": "sdr", "target": str(legs["sdr"]), "codec": "h264_nvenc", "tonemap": args.sdr_tonemap,
             "tonemap_param": args.knee, "bitrate_kbps": 6000}])))
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

        if args.keep:
            import shutil
            Path(args.keep).mkdir(parents=True, exist_ok=True)
            for f in (ref, *legs.values()):
                shutil.copy(f, Path(args.keep) / f.name)
        ref_bars, out_bars = bar_centres(args.ffmpeg, ref), bar_centres(args.ffmpeg, legs["sdr"])
        worst = max(abs(a - b) for r, o in zip(ref_bars, out_bars) for a, b in zip(r, o))
        detail = "; ".join(f"{'/'.join(f'{v:.0f}' for v in r)} -> {'/'.join(f'{v:.0f}' for v in o)}"
                           for r, o in zip(ref_bars, out_bars))
        assert worst <= TOLERANCE, f"SDR round trip drifted by {worst:.1f} codes: {detail}"
        print(f"PASS SDR -> HLG canvas -> SDR round trip ({args.sdr_tonemap}): every bar centre within "
              f"{worst:.1f} codes", flush=True)


if __name__ == "__main__":
    main()
