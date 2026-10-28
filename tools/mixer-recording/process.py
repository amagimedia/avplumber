#!/usr/bin/env python3
"""Decode the two host RTP captures and validate frame cadence and ticker motion.

Runs in a container that has python3, numpy and ffmpeg; works in the current directory, which holds
capture.json and native.pcap (Ethernet/IPv4), or pgm.mjr and preview.mjr (H.264 or H.265)."""

import collections, json, struct, subprocess, sys
from pathlib import Path
import numpy as np

root = Path(".")
capture = json.loads((root / "capture.json").read_text())
FPS = capture["fps"]
FRAMES = capture["frames"]
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
        pos += 4 + 4 * struct.unpack("!H", rtp[pos + 2 : pos + 4])[0]
    end = len(rtp) - (rtp[-1] if rtp[0] & 32 else 0)
    return rtp[pos:end]


def h264_units(data):
    """(bytes for the elementary stream, NAL types that start here) of one RTP payload."""
    kind = data[0] & 31
    if kind == 24:  # STAP-A
        out, kinds, pos = b"", [], 1
        while pos + 2 <= len(data):
            size = struct.unpack("!H", data[pos : pos + 2])[0]
            nal = data[pos + 2 : pos + 2 + size]
            out += START + nal
            kinds.append(nal[0] & 31)
            pos += 2 + size
        return out, kinds
    if kind == 28:  # FU-A
        if data[1] & 128:
            return START + bytes([(data[0] & 0xE0) | (data[1] & 31)]) + data[2:], [
                data[1] & 31
            ]
        return data[2:], []
    return START + data, [kind]


def h265_units(data):
    kind = (data[0] >> 1) & 63
    if kind == 48:  # aggregation packet
        out, kinds, pos = b"", [], 2
        while pos + 2 <= len(data):
            size = struct.unpack("!H", data[pos : pos + 2])[0]
            nal = data[pos + 2 : pos + 2 + size]
            out += START + nal
            kinds.append((nal[0] >> 1) & 63)
            pos += 2 + size
        return out, kinds
    if kind == 49:  # fragmentation unit
        fu = data[2]
        if fu & 128:
            return START + bytes([(data[0] & 0x81) | ((fu & 63) << 1), data[1]]) + data[
                3:
            ], [fu & 63]
        return data[3:], []
    return START + data, [kind]


def is_key(codec, kinds):
    return 5 in kinds if codec == "h264" else any(16 <= k <= 23 for k in kinds)


if (root / "native.pcap").exists():
    b = Path("native.pcap").read_bytes()
    endian = "<" if b[:4] in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1") else ">"
    nano = b[:4] in (b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d")
    assert struct.unpack_from(endian + "I", b, 20)[0] == 1
    pos = 24
    streams = {5004: [], 5008: []}
    while pos < len(b):
        sec, sub, size, orig = struct.unpack_from(endian + "IIII", b, pos)
        pos += 16
        p = b[pos : pos + size]
        pos += size
        if p[12:14] != b"\x08\x00":
            continue
        ip = p[14:]
        ihl = (ip[0] & 15) * 4
        if ip[9] != 17:
            continue
        udp = ip[ihl:]
        src, dst, length = struct.unpack_from("!HHH", udp)
        if dst in streams:
            streams[dst].append(
                (sec * 1000000 + int(sub / (1000 if nano else 1)), udp[8:length])
            )
    for port, name in [(5004, "pgm"), (5008, "preview")]:
        packets = streams[port]
        start = packets[0][0]
        header = json.dumps({"c": "h264", "t": "v", "u": start, "s": start}).encode()
        with open(name + ".mjr", "wb") as f:
            f.write(b"MJR00002" + struct.pack("!H", len(header)) + header)
            for wall, rtp in packets:
                f.write(
                    b"MEET" + struct.pack("!IH", (wall - start) // 1000, len(rtp)) + rtp
                )
        print(name, len(packets))

report = {"fps": FPS, "frames": FRAMES}
for name in ("pgm", "preview"):
    header, packets = read_mjr(root / f"{name}.mjr")
    units, ext = {"h264": (h264_units, "h264"), "h265": (h265_units, "hevc")}[
        header["c"]
    ]
    gaps = [
        (a[1], b[1])
        for a, b in zip(packets, packets[1:])
        if (b[1] - a[1]) & 0xFFFF != 1
    ]
    frames = (
        collections.OrderedDict()
    )  # RTP timestamp -> [bytes, NAL kinds, arrival ms]
    for elapsed, seq, stamp, rtp in packets:
        data, kinds = units(payload(rtp))
        frame = frames.setdefault(stamp, [bytearray(), [], elapsed, False])
        frame[0] += data
        frame[1] += kinds
        frame[3] = bool(rtp[1] & 128)
    # Stopping a recording may split its final access unit. Do not decode that partial tail.
    incomplete_tail = not next(reversed(frames.values()))[3]
    if incomplete_tail:
        frames.popitem()
    assert all(frame[3] for frame in frames.values()), "Incomplete RTP access unit"
    assert not gaps, f"{name}: RTP packet sequence gaps"
    stamps = list(frames)
    deltas = collections.Counter(
        (b - a) & 0xFFFFFFFF for a, b in zip(stamps, stamps[1:])
    )
    assert set(deltas) == {90000 // FPS}, f"{name}: discontinuous RTP frame clock"
    keys = [i for i, s in enumerate(stamps) if is_key(header["c"], frames[s][1])]
    first_key = keys[0]
    # The first packet of the recording was written at header["u"] (microseconds); frames follow the RTP clock.
    wall = [header["u"] / 1e6 + ((s - stamps[0]) & 0xFFFFFFFF) / 90000 for s in stamps]
    start = next(i for i, w in enumerate(wall) if w >= capture["demo_start_wall"])
    assert start >= first_key and start + FRAMES <= len(stamps), (
        first_key,
        start,
        len(stamps),
    )
    (root / f"{name}.{ext}").write_bytes(
        b"".join(bytes(frames[s][0]) for s in stamps[first_key:])
    )
    report[name] = {
        "codec": header["c"],
        "extension": ext,
        "first_frame_wall": wall[start],
        "packets": len(packets),
        "rtp_frames": len(stamps),
        "incomplete_tail_discarded": incomplete_tail,
        "frame_tick_deltas": {str(k): v for k, v in deltas.items()},
        "packet_sequence_gaps": len(gaps),
        "first_keyframe": first_key,
        "keyframes": len(keys),
        "demo_start_frame": start,
        "demo_start_in_stream": start - first_key,
        "window_tick_deltas": {
            str(k): v
            for k, v in collections.Counter(
                (b - a) & 0xFFFFFFFF
                for a, b in zip(
                    stamps[start : start + FRAMES], stamps[start + 1 : start + FRAMES]
                )
            ).items()
        },
    }
    print(name, json.dumps(report[name]), flush=True)


def ffmpeg(*args, **kw):
    return subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True, **kw
    )


for name in ("pgm", "preview"):
    ext = report[name]["extension"]
    skip = report[name]["demo_start_in_stream"]
    out = root / "frames" / name
    out.mkdir(parents=True, exist_ok=True)
    pick = f"select='between(n\\,{skip}\\,{skip + FRAMES - 1})'"
    ffmpeg(
        "-framerate",
        str(FPS),
        "-i",
        f"{name}.{ext}",
        "-vf",
        f"{pick},scale=760:428:flags=lanczos",
        "-fps_mode",
        "passthrough",
        "-start_number",
        "1",
        str(out / "%03d.png"),
    )
    print(name, "frames", len(list(out.glob("*.png"))), flush=True)

skip = report["pgm"]["demo_start_in_stream"]
pick = f"select='between(n\\,{skip}\\,{skip + FRAMES - 1})'"
# The ticker band, full size: how far its text moves between consecutive frames.
raw = ffmpeg(
    "-framerate",
    str(FPS),
    "-i",
    "pgm.h264",
    "-vf",
    f"{pick},crop=1500:72:250:972,format=gray",
    "-fps_mode",
    "passthrough",
    "-f",
    "rawvideo",
    "pipe:1",
    stdout=subprocess.PIPE,
).stdout
band = np.frombuffer(raw, np.uint8).reshape(-1, 72, 1500) > 160
shifts, errors = [], []
for a, b in zip(band, band[1:]):
    scores = []
    for dx in range(-16, 9):
        left = a[:, max(0, -dx) : min(1500, 1500 - dx)]
        right = b[:, max(0, dx) : min(1500, 1500 + dx)]
        scores.append((float(np.mean(left != right)), dx))
    score, dx = min(scores)
    shifts.append(dx)
    errors.append(score)
report["ticker"] = {
    "frame_pairs": len(shifts),
    "horizontal_steps_px": {str(k): v for k, v in collections.Counter(shifts).items()},
    "median_match_error": float(np.median(errors)),
}
ffmpeg(
    "-framerate",
    str(FPS),
    "-i",
    "pgm.h264",
    "-c",
    "copy",
    "-movflags",
    "+faststart",
    "mixer-pgm-instance-1080p30.mp4",
)
(root / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report["ticker"]))
