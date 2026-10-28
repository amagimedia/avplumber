#!/usr/bin/env python3
"""Turn the two Janus recordings of one capture into elementary streams, frames and a timing report.

Runs in a container that has python3, numpy and ffmpeg; works in the current directory, which holds
capture.json, pgm.mjr (H.264) and preview.mjr (H.265)."""
import collections, json, struct, subprocess, sys
from pathlib import Path
import numpy as np

FPS = 30
FRAMES = 10 * FPS
root = Path(".")
capture = json.loads((root / "capture.json").read_text())
START = b"\x00\x00\x00\x01"


def read_mjr(path):
    packets = []
    with path.open("rb") as f:
        assert f.read(8) == b"MJR00002"
        header = json.loads(f.read(struct.unpack("!H", f.read(2))[0]))
        while record := f.read(10):
            assert len(record) == 10 and record[:4] == b"MEET"
            elapsed, size = struct.unpack("!IH", record[4:])
            rtp = f.read(size)
            assert len(rtp) == size
            seq, stamp = struct.unpack("!HI", rtp[2:8])
            packets.append((elapsed, seq, stamp, rtp))
    return header, packets


def payload(rtp):
    pos = 12 + 4 * (rtp[0] & 15)
    if rtp[0] & 16:
        pos += 4 + 4 * struct.unpack("!H", rtp[pos + 2:pos + 4])[0]
    end = len(rtp) - (rtp[-1] if rtp[0] & 32 else 0)
    return rtp[pos:end]


def h264_units(data):
    """(bytes for the elementary stream, NAL types that start here) of one RTP payload."""
    kind = data[0] & 31
    if kind == 24:                                   # STAP-A
        out, kinds, pos = b"", [], 1
        while pos + 2 <= len(data):
            size = struct.unpack("!H", data[pos:pos + 2])[0]
            nal = data[pos + 2:pos + 2 + size]
            out += START + nal; kinds.append(nal[0] & 31); pos += 2 + size
        return out, kinds
    if kind == 28:                                   # FU-A
        if data[1] & 128:
            return START + bytes([(data[0] & 0xE0) | (data[1] & 31)]) + data[2:], [data[1] & 31]
        return data[2:], []
    return START + data, [kind]


def h265_units(data):
    kind = (data[0] >> 1) & 63
    if kind == 48:                                   # aggregation packet
        out, kinds, pos = b"", [], 2
        while pos + 2 <= len(data):
            size = struct.unpack("!H", data[pos:pos + 2])[0]
            nal = data[pos + 2:pos + 2 + size]
            out += START + nal; kinds.append((nal[0] >> 1) & 63); pos += 2 + size
        return out, kinds
    if kind == 49:                                   # fragmentation unit
        fu = data[2]
        if fu & 128:
            return START + bytes([(data[0] & 0x81) | ((fu & 63) << 1), data[1]]) + data[3:], [fu & 63]
        return data[3:], []
    return START + data, [kind]


def is_key(name, kinds):
    return 5 in kinds if name == "pgm" else any(16 <= k <= 23 for k in kinds)


report = {}
for name, units, ext in (("pgm", h264_units, "h264"), ("preview", h265_units, "hevc")):
    header, packets = read_mjr(root / f"{name}.mjr")
    gaps = [(a[1], b[1]) for a, b in zip(packets, packets[1:]) if (b[1] - a[1]) & 0xFFFF != 1]
    frames = collections.OrderedDict()               # RTP timestamp -> [bytes, NAL kinds, arrival ms]
    for elapsed, seq, stamp, rtp in packets:
        data, kinds = units(payload(rtp))
        frame = frames.setdefault(stamp, [bytearray(), [], elapsed])
        frame[0] += data; frame[1] += kinds
    stamps = list(frames)
    deltas = collections.Counter((b - a) & 0xFFFFFFFF for a, b in zip(stamps, stamps[1:]))
    keys = [i for i, s in enumerate(stamps) if is_key(name, frames[s][1])]
    first_key = keys[0]
    # The first packet of the recording was written at header["u"] (microseconds); frames follow the RTP clock.
    wall = [header["u"] / 1e6 + ((s - stamps[0]) & 0xFFFFFFFF) / 90000 for s in stamps]
    start = next(i for i, w in enumerate(wall) if w >= capture["demo_start_wall"])
    assert start >= first_key and start + FRAMES <= len(stamps), (first_key, start, len(stamps))
    (root / f"{name}.{ext}").write_bytes(b"".join(bytes(frames[s][0]) for s in stamps[first_key:]))
    report[name] = {"codec": header["c"], "packets": len(packets), "rtp_frames": len(stamps),
                    "frame_tick_deltas": {str(k): v for k, v in deltas.items()}, "packet_sequence_gaps": len(gaps),
                    "first_keyframe": first_key, "keyframes": len(keys), "demo_start_frame": start,
                    "demo_start_in_stream": start - first_key,
                    "window_tick_deltas": {str(k): v for k, v in collections.Counter(
                        (b - a) & 0xFFFFFFFF for a, b in zip(stamps[start:start + FRAMES], stamps[start + 1:start + FRAMES])).items()}}
    print(name, json.dumps(report[name]), flush=True)


def ffmpeg(*args, **kw):
    return subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True, **kw)


for name, ext in (("pgm", "h264"), ("preview", "hevc")):
    skip = report[name]["demo_start_in_stream"]
    out = root / "frames" / name
    out.mkdir(parents=True, exist_ok=True)
    pick = f"select='between(n\\,{skip}\\,{skip + FRAMES - 1})'"
    ffmpeg("-framerate", str(FPS), "-i", f"{name}.{ext}", "-vf", f"{pick},scale=760:428:flags=lanczos",
           "-fps_mode", "passthrough", "-start_number", "1", str(out / "%03d.png"))
    print(name, "frames", len(list(out.glob("*.png"))), flush=True)

skip = report["pgm"]["demo_start_in_stream"]
pick = f"select='between(n\\,{skip}\\,{skip + FRAMES - 1})'"
# Where the picture changes: the frames a cut lands on.
raw = ffmpeg("-framerate", str(FPS), "-i", "pgm.h264", "-vf", f"{pick},scale=160:90,format=gray", "-fps_mode", "passthrough",
             "-f", "rawvideo", "pipe:1", stdout=subprocess.PIPE).stdout
small = np.frombuffer(raw, np.uint8).reshape(-1, 90, 160).astype(np.int16)
change = [float(np.mean(np.abs(b - a))) for a, b in zip(small, small[1:])]
order = sorted(range(len(change)), key=lambda i: -change[i])
cuts = sorted(i + 1 for i in order[:6])
report["cut_frames"] = cuts
report["cut_seconds"] = [round(c / FPS, 3) for c in cuts]
report["cut_change"] = [round(change[c - 1], 1) for c in cuts]
report["largest_other_change"] = round(change[order[6]], 1)
# The ticker band, full size: how far its text moves between consecutive frames.
raw = ffmpeg("-framerate", str(FPS), "-i", "pgm.h264", "-vf", f"{pick},crop=1500:72:250:972,format=gray", "-fps_mode", "passthrough",
             "-f", "rawvideo", "pipe:1", stdout=subprocess.PIPE).stdout
band = np.frombuffer(raw, np.uint8).reshape(-1, 72, 1500) > 160
shifts, errors = [], []
for a, b in zip(band, band[1:]):
    scores = []
    for dx in range(-16, 9):
        left = a[:, max(0, -dx):min(1500, 1500 - dx)]; right = b[:, max(0, dx):min(1500, 1500 + dx)]
        scores.append((float(np.mean(left != right)), dx))
    score, dx = min(scores); shifts.append(dx); errors.append(score)
report["ticker"] = {"frame_pairs": len(shifts), "horizontal_steps_px": {str(k): v for k, v in collections.Counter(shifts).items()},
                    "median_match_error": float(np.median(errors))}
ffmpeg("-framerate", str(FPS), "-i", "pgm.h264", "-c", "copy", "-movflags", "+faststart", "mixer-pgm-instance-1080p30.mp4")
(root / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: report[k] for k in ("cut_frames", "cut_seconds", "cut_change", "largest_other_change", "ticker")}))
