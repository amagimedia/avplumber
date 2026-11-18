#!/usr/bin/env python3
"""rtp_latency.py CAPTURE.pcap [--x 1584 --y 32]

Live-input -> wire latency of an H.264 RTP stream, measured the same way for any producer:
  * the input is a Chromium page (tspage.html) that paints the wall clock into a 32-block barcode each frame;
  * the capture is tcpdump on the host (`tcpdump -i lo -s 0 -w f.pcap udp dst port N`): kernel arrival time
    of every RTP packet, outside the producer;
  * each frame is rebuilt from its RTP packets (single NAL, STAP-A, FU-A), its arrival time is that of its
    marker (last) packet, it is decoded (nvh264dec) and the barcode is read at the box position;
  * latency = arrival - painted time, both on the host clock.
Prints p50/p90/p99/max and how many frames carried a valid barcode.
"""
import argparse, json, struct, sys
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

P = argparse.ArgumentParser()
P.add_argument("pcap")
P.add_argument("--x", type=int, default=1584)
P.add_argument("--y", type=int, default=32)
P.add_argument("--block", type=int, default=38)
A = P.parse_args()

# ---- pcap -> RTP packets (Ethernet / Linux cooked v1 / v2 link types, IPv4 UDP)
raw = open(A.pcap, "rb").read()
magic, = struct.unpack("<I", raw[:4])
endian, nano = {0xa1b2c3d4: ("<", False), 0xa1b23c4d: ("<", True)}.get(magic, (">", False))
linktype, = struct.unpack(endian + "I", raw[20:24])
l2 = {1: 14, 113: 16, 276: 20}[linktype]
pos, pkts = 24, []
while pos + 16 <= len(raw):
    sec, frac, incl, _ = struct.unpack(endian + "IIII", raw[pos:pos + 16]); pos += 16
    data = raw[pos:pos + incl]; pos += incl
    ip = data[l2:]
    if len(ip) < 28 or ip[0] >> 4 != 4 or ip[9] != 17:
        continue
    udp = ip[(ip[0] & 15) * 4:]
    rtp = udp[8:]
    if len(rtp) < 12 or rtp[0] >> 6 != 2:
        continue
    cc, ext = rtp[0] & 15, rtp[0] & 16
    marker, ts = rtp[1] >> 7, struct.unpack(">I", rtp[4:8])[0]
    off = 12 + 4 * cc
    if ext:
        off += 4 + 4 * struct.unpack(">H", rtp[off + 2:off + 4])[0]
    pkts.append((sec + frac / (1e9 if nano else 1e6), ts, marker, rtp[off:]))

# ---- RTP -> Annex-B access units with the arrival time of their last packet
SC = b"\x00\x00\x00\x01"
frames, cur, fu = [], [], None
for t, ts, marker, pl in pkts:
    if not pl:
        continue
    typ = pl[0] & 31
    if typ == 24:                                   # STAP-A
        i = 1
        while i + 2 <= len(pl):
            n, = struct.unpack(">H", pl[i:i + 2]); cur.append(SC + pl[i + 2:i + 2 + n]); i += 2 + n
    elif typ == 28:                                 # FU-A
        s, e = pl[1] & 128, pl[1] & 64
        if s:
            fu = bytearray(SC + bytes([(pl[0] & 0xe0) | (pl[1] & 31)]))
        if fu is not None:
            fu += pl[2:]
            if e:
                cur.append(bytes(fu)); fu = None
    else:
        cur.append(SC + pl)
    if marker:
        frames.append((t, b"".join(cur))); cur = []
if not frames:
    sys.exit("no complete frames in the capture")

# ---- decode in order and read the barcode
Gst.init(None)
pipe = Gst.parse_launch("appsrc name=src format=time caps=video/x-h264,stream-format=byte-stream,alignment=au ! "
                        "h264parse ! nvh264dec ! video/x-raw,format=NV12 ! appsink name=sink sync=false "
                        "max-buffers=0 emit-signals=false")
src, sink = pipe.get_by_name("src"), pipe.get_by_name("sink")
pipe.set_state(Gst.State.PLAYING)
for k, (_, au) in enumerate(frames):
    b = Gst.Buffer.new_wrapped(au); b.pts = k * Gst.SECOND // 30; b.duration = Gst.SECOND // 30
    src.emit("push-buffer", b)
src.emit("end-of-stream")

B = A.block
lat, bad, decoded, seen = [], 0, 0, set()
while True:
    s = sink.emit("try-pull-sample", 5 * Gst.SECOND)
    if s is None:
        break
    decoded += 1
    buf = s.get_buffer(); st = s.get_caps().get_structure(0)
    w = st.get_value("width")
    k = round(buf.pts * 30 / Gst.SECOND)
    ok, info = buf.map(Gst.MapFlags.READ)
    y = info.data
    bits = 0
    for i in range(32):
        cx, cy = A.x + (i % 8) * B + B // 2, A.y + (i // 8) * B + B // 2
        v = sum(y[(cy + dy) * w + cx + dx] for dy in (-6, 0, 6) for dx in (-6, 0, 6)) / 9
        bits = (bits << 1) | (v > 128)
    buf.unmap(info)
    t24, chk = bits >> 8, bits & 255
    if chk != (((t24 >> 16) + ((t24 >> 8) & 255) + (t24 & 255)) & 255) ^ 0xa5 or k >= len(frames):
        bad += 1; continue
    arr_ms = frames[k][0] * 1000
    lat.append((arr_ms - t24) % 16777216)
    seen.add(t24)
pipe.set_state(Gst.State.NULL)

lat.sort()
pct = lambda p: round(lat[min(len(lat) - 1, int(p / 100 * len(lat)))], 1) if lat else None
print(json.dumps({"frames": len(frames), "decoded": decoded, "valid_barcodes": len(lat), "invalid": bad,
                  "seconds": round(frames[-1][0] - frames[0][0], 1),
                  "page_updates_per_s": round(len(seen) / max(0.1, frames[-1][0] - frames[0][0]), 1),
                  "latency_ms": {"min": pct(0), "p50": pct(50), "p90": pct(90), "p99": pct(99), "max": pct(100)}}))
