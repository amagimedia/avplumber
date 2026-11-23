"""Verify packed_to_cuda: packed UYVY and RGB frames unpacked on the device to NV12 or BGRA.

UYVY to NV12 is byte for byte: luma is the packed picture's and a 4:2:0 chroma row the rounded mean
of its two picture rows, which are of one field in an interlaced frame. RGB to NV12 and UYVY to
BGRA follow the kernels' own definition (BT.709, limited range) within one code, and RGB to NV12
is also compared with swscale's conversion. With storage bgra, bgra passes untouched, alpha
included, and frames without alpha come out opaque. Widths that end inside a kernel thread's
8 pixels, inside a block and on an odd column are covered. GPU readback is solely for verification.
"""
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from _harness import drain, finish, make_avp
from pyplumber import node as api
from pyplumber.mixer.inputs import _pace, _raw_file_packets

KR, KB = 0.2126, 0.0722
KG = 1 - KR - KB
BYTES = {"uyvy422": 2, "bgra": 4, "bgr0": 4, "rgba": 4, "rgb0": 4}


def code(values):
    return np.clip(np.floor(values + 0.5), 0, 255).astype(np.uint8)


def chroma_rows(rows, interlaced):
    """The two picture rows of each 4:2:0 chroma row: of one field when interlaced."""
    step = 2 if interlaced else 1
    first = [2 * c - (c & 1 if interlaced else 0) for c in range((rows + 1) // 2)]
    return first, [f + step if f + step < rows else f for f in first]


def rgb_of(picture, fmt):
    """(R, G, B, A) planes of packed RGB rows; A is 255 where the format has none."""
    p = picture.reshape(len(picture), -1, 4).astype(np.float32)
    r, b = (p[..., 0], p[..., 2]) if fmt.startswith("rgb") else (p[..., 2], p[..., 0])
    return r, p[..., 1], b, p[..., 3] if fmt.endswith("a") else np.full_like(r, 255)


def nv12_of(picture, fmt, interlaced):
    """NV12 planes (luma, chroma) the node makes of a packed picture."""
    first, second = chroma_rows(len(picture), interlaced)
    if fmt == "uyvy422":
        chroma = picture[:, 0::2].astype(np.uint16)
        return picture[:, 1::2], ((chroma[first] + chroma[second] + 1) >> 1).astype(np.uint8)
    r, g, b, _ = rgb_of(picture, fmt)
    luma = code(16 + 219 / 255 * (KR * r + KG * g + KB * b))
    if r.shape[1] % 2:   # an odd width repeats its last column
        r, g, b = (np.concatenate((c, c[:, -1:]), axis=1) for c in (r, g, b))
    r, g, b = ((c[first] + c[second]).reshape(len(first), -1, 2).sum(axis=2) / 4 for c in (r, g, b))
    y = KR * r + KG * g + KB * b
    chroma = np.stack((code(128 + 112 / 255 * (b - y) / (1 - KB)), code(128 + 112 / 255 * (r - y) / (1 - KR))), axis=-1)
    return luma, chroma.reshape(len(first), -1)


def bgra_of(picture, fmt):
    """BGRA rows the node makes of a packed picture: opaque where the format has no alpha."""
    if fmt == "uyvy422":
        y = 255 / 219 * (picture[:, 1::2].astype(np.float32) - 16)
        cb, cr = (np.repeat(picture[:, o::4].astype(np.float32) - 128, 2, axis=1) for o in (0, 2))
        b, r = y + 2 * 255 / 224 * (1 - KB) * cb, y + 2 * 255 / 224 * (1 - KR) * cr
        r, g, b, a = code(r), code((y - KR * r - KB * b) / KG), code(b), np.full(y.shape, 255, np.uint8)
    else:
        r, g, b, a = (c.astype(np.uint8) for c in rgb_of(picture, fmt))
    return np.stack((b, g, r, a), axis=-1).reshape(len(picture), -1)


def swscale_nv12(path, fmt, width, height, interlaced):
    """One picture of the file as swscale converts it to BT.709 limited-range NV12."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", fmt, "-s", f"{width}x{height}", "-i", str(path),
                          "-frames:v", "1", "-vf", f"scale=out_color_matrix=bt709:out_range=tv:interl={int(interlaced)},format=nv12",
                          "-f", "rawvideo", "-"], check=True, capture_output=True).stdout
    planes = np.frombuffer(out, np.uint8)
    return planes[:width * height].reshape(height, width), planes[width * height:].reshape((height + 1) // 2, -1)


def check(fmt, storage, width, interlaced, tolerance):
    height, fps = 64, 30
    row, col = np.indices((height, width * BYTES[fmt]))
    if fmt == "uyvy422":
        pictures = [((col * 7 + row * 13 + i * 31 + (col % 2) * row * 5) % 256).astype(np.uint8) for i in range(6)]
    else:   # smooth and different in each channel, alpha from clear to solid
        channel, x = col % 4, col // 4
        pictures = [np.choose(channel, (x * 255 // max(width - 1, 1), row * 255 // (height - 1),
                                        255 - (x + row + i * 8) * 255 // (width + height + 48),
                                        (x + row * 3 + i * 40) % 256)).astype(np.uint8) for i in range(6)]
    expected = [(bgra_of(p, fmt),) if storage == "bgra" else nv12_of(p, fmt, interlaced) for p in pictures]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "packed.raw"
        path.write_bytes(b"".join(picture.tobytes() for picture in pictures))
        avp, errors = make_avp("packed_gpu", capacity=2)
        nodes = []
        try:
            packets = _raw_file_packets(avp, api, "packed", str(path), pixel_format=fmt,
                                       video_size=f"{width}x{height}", group="probe",
                                       fps=fps, fps_den=1, loop=True)
            chain = [api.DecVideo({"name": "decode", "src": packets, "dst": "decoded",
                                   "codec": "rawvideo", "pixel_format": fmt, "group": "probe"})]
            if interlaced:
                chain.append(api.FilterVideo({"name": "fields", "src": "decoded", "dst": "woven",
                                              "graph": "setfield=tff", "threads": 1, "group": "probe"}))
            chain.append(api.PackedToCuda({"name": "upload", "src": "woven" if interlaced else "decoded", "dst": "uploaded",
                                           "hwaccel": "packed_gpu", "storage": storage,
                                           "frame_color": {"colorspace": "bt709", "color_range": "tv"}, "group": "probe"}))
            for node in chain:
                nodes.append(node)
                avp.addNode(node)
            paced = _pace(avp, api, "packed", "uploaded", fps=fps, fps_den=1, group="probe", event_loop=None)
            verify = api.FilterVideo({"name": "verify", "src": paced, "dst": "result",
                                      "graph": f"hwdownload,format={storage}", "hwaccel": "packed_gpu", "threads": 1, "group": "probe"})
            nodes.append(verify)
            avp.addNode(verify)
            output = avp.getEdge("result", "VideoFrame")
            avp.group("probe").startNodes()
            seen, worst = [], 0
            for frame in drain(output, errors, timeout=10, limit=24):
                planes = [np.frombuffer(frame.data[p], np.uint8).reshape(-1, frame.linesize[p])[:len(e), :e.shape[1]].astype(int)
                          for p, e in enumerate(expected[0])]
                off = [max(int(np.abs(got - want).max()) for got, want in zip(planes, e)) for e in expected]
                assert min(off) <= tolerance, f"{fmt} to {storage}: no picture within {tolerance}: {off}"
                seen.append(off.index(min(off)))
                worst = max(worst, min(off))
            assert not errors, errors
            assert len(seen) == 24 and len(set(seen)) >= 3, seen
            against = ""
            if storage == "nv12" and fmt != "uyvy422":
                diffs = [np.abs(got.astype(int) - want) for got, want in zip(swscale_nv12(path, fmt, width, height, interlaced), expected[0])]
                assert diffs[0].max() <= 2 and diffs[1].max() <= 6 and diffs[1].mean() < 1, [(d.max(), d.mean()) for d in diffs]
                against = f"; swscale within {diffs[0].max()} (luma) and {diffs[1].max()} (chroma, mean {diffs[1].mean():.2f})"
            print(f"PASS: {fmt} {width}x{height} to {storage}, interlaced={interlaced}, 24 frames within {worst}{against}", flush=True)
        finally:
            finish(avp, nodes)


if __name__ == "__main__":
    for width in (50, 258, 1920):
        for interlaced in (False, True):
            check("uyvy422", "nv12", width, interlaced, 0)
    for fmt in ("bgra", "bgr0", "rgba", "rgb0"):
        for width in (50, 51, 1920):
            for interlaced in (False, True):
                check(fmt, "nv12", width, interlaced, 1)
    for fmt, tolerance in (("bgra", 0), ("bgr0", 0), ("rgba", 0), ("rgb0", 0), ("uyvy422", 1)):
        for width in (50, 1920) + ((51,) if fmt != "uyvy422" else ()):
            check(fmt, "bgra", width, False, tolerance)
