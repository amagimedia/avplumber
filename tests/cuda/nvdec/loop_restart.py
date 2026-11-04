"""A looping clip decodes the same on every pass on NVDEC, progressive and interlaced, on the NVIDIA host.

Each clip plays twice through input_rec with `loop`: as it is (a pass begins with a keyframe: the
decoder is never reset) and with its first packet cut off (a pass begins with a predicted picture:
input_rec marks it, and the decoder drains and resets). Both must give the same pictures on every
pass, from a decoder that never ends. Progressive clips decode as the mixer's CUarray sources do
(parking on, so an H.264 reset takes its own session back), and on linear CUDA; an interlaced
(field-coded or MBAFF) clip decodes on linear CUDA and is deinterlaced as a mixer `deinterlace`
source is.

    python3 tests/cuda/nvdec/loop_restart.py --clip h264.ts --clip hevc.ts --interlaced mbaff.ts \
        [--passes 3] [--ffmpeg ffmpeg] --work <dir>
"""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from _harness import EOF, make_avp
from pyplumber import node as api
from pyplumber.mixer.config import DEINTERLACE_FILTER

GPU = "loop_gpu"


def pixels(frame):
    digest = hashlib.sha256()
    for plane, rows in enumerate((frame.height, (frame.height + 1) // 2)):
        view, pitch = memoryview(frame.data[plane]), frame.linesize[plane]
        width = frame.width if plane == 0 else ((frame.width + 1) // 2) * 2
        for row in range(rows):
            digest.update(view[row * pitch:row * pitch + width])
    return digest.hexdigest()


def nodes(path, storage, interlaced, size):
    decoder = {"name": "decode", "src": "video", "dst": "decoded", "hwaccel": GPU}
    if storage == "cuarray":
        decoder.update(pixel_format="cuarray", park=True, park_limit=2,
                       options={"threads": 1, "extra_hw_frames": 3,
                                "hwaccel_flags": "unsafe_output+extra_hw_frames_only+private_stream"})
    else:
        decoder.update(pixel_format="cuda", options={"threads": 1})
    chain = [api.InputRec({"name": "input", "url": str(path), "dst": "packets", "loop": True}),
             api.Demux({"name": "demux", "src": "packets", "routing": {"v:0": "video"}}),
             api.DecVideo(decoder)]
    if storage == "cuarray":   # CUarray frames are read by the rect compositor, as in the mixer
        width, height = size
        chain.append(api.CudaRectOverlay({"name": "compose", "src": ["decoded"], "dst": "readable", "hwaccel": GPU,
                                          "width": width, "height": height, "sw_format": "nv12",
                                          "layers": [{"input": 0, "dst_x": 0, "dst_y": 0, "dst_w": width, "dst_h": height}]}))
        graph = "hwdownload,format=nv12"
    else:
        chain.append(api.FilterVideo({"name": "deinterlace", "src": "decoded", "dst": "readable", "hwaccel": GPU,
                                      "threads": 1, "defer_preliminary_init": True,
                                      "graph": DEINTERLACE_FILTER if interlaced else "null"}))
        graph = "hwdownload,format=nv12"
    chain.append(api.FilterVideo({"name": "download", "src": "readable", "dst": "result", "hwaccel": GPU,
                                  "threads": 1, "defer_preliminary_init": True, "graph": graph}))
    return chain


def passes(path, storage, interlaced, size, count, timeout):
    """(pts, hash) of every picture of `count` passes, and the decoder's status after them."""
    avp, errors = make_avp(GPU, capacity=4)
    for node in nodes(path, storage, interlaced, size):
        node.parameters.update(group="clip", auto_restart="off")
        avp.addNode(node)
    out = avp.getEdge("result", "VideoFrame")
    result, last = [[]], None
    try:
        avp.group("clip").startNodes()
        deadline = time.monotonic() + timeout
        while len(result) <= count:
            assert not errors, errors
            assert time.monotonic() < deadline, f"{path.name} {storage}: {sum(map(len, result))} pictures in {timeout} s"
            frame = out.tryGet(10)
            if frame is None or frame.pts.timestamp == EOF:
                continue
            seconds = frame.pts.timestamp * frame.pts.timebase.num / frame.pts.timebase.den
            if last is not None and seconds < last - 0.5:   # the clip began again
                result.append([])
            last = seconds
            result[-1].append((round(seconds, 4), pixels(frame)))
        status = avp.node("decode").getObject("status")
    finally:
        worker = threading.Thread(target=avp.shutdown, daemon=True)
        worker.start()
        worker.join(30)
        assert not worker.is_alive(), "shutdown hung"
    return result[:count], status


def probe(ffprobe, path):
    out = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", str(path)], check=True, capture_output=True, text=True).stdout
    width, height = (int(v) for v in out.strip().split(",")[:2])
    return width, height


def cut(ffmpeg, path, work):
    """The clip without its first packet, as a transport stream: it begins with a predicted picture."""
    out = work / f"{path.stem}_cut.ts"
    subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(path), "-map", "0:v:0", "-c", "copy",
                    "-bsf:v", "noise=drop=eq(n\\,0)", "-f", "mpegts", str(out)], check=True, capture_output=True, timeout=120)
    return out


def check(name, got, status, restarted, count, warmup):
    """Every pass shows the same pictures at the same times. `warmup`: pictures at the start whose
    content may differ on the first pass alone (the deinterlacer reads the picture before, which the
    first pass does not have); their times must still match."""
    first = got[0]
    assert first, f"{name}: the first pass gave no picture"
    for number, later in enumerate(got[1:], start=2):
        assert [t for t, _ in later] == [t for t, _ in first], \
            f"{name}: pass {number} shows {len(later)} pictures at other times than the first's {len(first)}"
        differ = sum(a != b for a, b in zip(first[warmup:], later[warmup:]))
        assert not differ, f"{name}: pass {number} differs from the first in {differ} of {len(first)} pictures"
    restarts = status.get("stream_restarts", 0)
    if restarted:
        assert restarts >= count - 1, f"{name}: {restarts} decoder resets over {count} passes"
    else:
        assert restarts == 0, f"{name}: a pass that begins with a keyframe reset the decoder {restarts} times"
    assert not status.get("degraded"), f"{name}: decoder degraded: {status}"
    print(f"{name}: {count} passes of {len(first)} pictures match, {restarts} resets, "
          f"{status.get('decode_errors', 0)} decode errors", flush=True)


def run(args):
    args.work.mkdir(parents=True, exist_ok=True)
    if not args.interlaced:
        raise SystemExit("an interlaced clip is required (--interlaced): it is tested on every run")
    cases = [(Path(c), storage, False) for c in args.clip for storage in ("cuarray", "cuda")]
    cases += [(Path(c), "cuda", True) for c in args.interlaced]
    for path, storage, interlaced in cases:
        size = probe(args.ffprobe, path)
        for clip, restarted in ((path, False), (cut(args.ffmpeg, path, args.work), True)):
            name = f"{path.name} {'interlaced ' if interlaced else ''}{storage}{' cut' if restarted else ''}"
            got, status = passes(clip, storage, interlaced, size, args.passes, args.timeout)
            check(name, got, status, restarted, args.passes, warmup=2 if interlaced else 0)
    print("PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", action="append", default=[], help="a progressive H.264 or HEVC clip")
    parser.add_argument("--interlaced", action="append", default=[], help="a field-coded or MBAFF H.264 clip")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--work", type=Path, required=True)
    run(parser.parse_args())
