"""Shared scaffolding for the CUDA smokes: AVPlumber setup, the packed-v210
ingest chain, plane extraction and the frame drain loop."""

import time

import numpy as np

EOF = -(1 << 63)


def make_avp(hwaccel, capacity=3):
    """AVPlumber with a CUDA hwaccel and an error sink; returns (avp, errors)."""
    from pyplumber import AVPlumber
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *e: errors.append(tuple(map(str, e)))
    avp.edges.planCapacity("*", capacity)
    avp.executeCommandsFromString(f'hwaccel.init {{"name":"{hwaccel}","type":"cuda"}}')
    return avp, errors


def v210_chain(nodes, tag, path, *, width, height, stride, fmt, hwaccel, color, **unpack):
    """Input(rawvideo gray, stride x height packets) -> Demux -> V210ToCuda; returns the edge.
    The gray demuxer only frames bytes, which also permits nonstandard row strides."""
    from pyplumber.node import Demux, Input, V210ToCuda
    nodes += [
        Input({"name": f"in_{tag}", "url": str(path), "format": "rawvideo", "dst": f"pkt_{tag}",
               "options": {"pixel_format": "gray", "video_size": f"{stride}x{height}",
                           "framerate": "60"}}),
        Demux({"name": f"demux_{tag}", "src": f"pkt_{tag}", "routing": {"v:0": f"packed_{tag}"}}),
        V210ToCuda({"name": f"unpack_{tag}", "src": f"packed_{tag}", "dst": f"gpu_{tag}",
                    "hwaccel": hwaccel, "width": width, "height": height, "stride": stride,
                    "fps": "60/1", "timebase": "1/90000", "format": fmt, **color, **unpack}),
    ]
    return f"gpu_{tag}"


def frame_planes(frame, fmt):
    """Downloaded frame -> (Y, U, V) logical 10-bit planes."""
    if fmt in ("p010le", "p210le"):
        heights = (frame.height, (frame.height + 1) // 2 if fmt == "p010le" else frame.height)
        y, uv = [np.frombuffer(d, "<u2").reshape(h, p // 2)
                 for d, p, h in zip(frame.data[:2], frame.linesize[:2], heights)]
        y, uv = y[:, :frame.width], uv[:, :frame.width]
        assert not np.any(y & 63) and not np.any(uv & 63), f"{fmt} low bits must be zero"
        return y >> 6, uv[:, 0::2] >> 6, uv[:, 1::2] >> 6
    widths = [frame.width, frame.width // 2, frame.width // 2] if fmt != "yuv444p10le" \
        else [frame.width] * 3
    return tuple(np.frombuffer(d, "<u2").reshape(frame.height, p // 2)[:, :w]
                 for d, p, w in zip(frame.data[:3], frame.linesize[:3], widths))


def start(avp, nodes, group, edge):
    """Register nodes under one group with auto_restart off, start them, return the edge."""
    for node in nodes:
        node.parameters.update({"group": group, "auto_restart": "off"})
        avp.addNode(node)
    out = avp.getEdge(edge, "VideoFrame")
    avp.group(group).startNodes()
    return out


def drain(edge, errors, timeout, limit=None, state=None):
    """Yield frames until EOF, *limit* frames, an error or the deadline.
    ``state`` (a dict) receives ``eof`` and ``count`` for callers that care."""
    deadline = time.monotonic() + timeout
    count, eof = 0, False
    while time.monotonic() < deadline and not errors and (limit is None or count < limit):
        frame = edge.tryGet(100)
        if frame is None:
            continue
        if frame.pts.timestamp == EOF:
            eof = True
            break
        count += 1
        yield frame
    if state is not None:
        state.update(eof=eof, count=count)


def finish(avp, nodes, timeout=10):
    """Tear the instance down. avplumber's finite-graph shutdown can hang in the
    stop/start race of a group being torn down; a smoke must not stall on it,
    so the instance is abandoned (leaked) after *timeout* seconds."""
    import threading
    nodes.clear()
    worker = threading.Thread(target=avp.shutdown, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        print(f"WARNING: avp.shutdown() hung for {timeout}s; leaking the instance", flush=True)
