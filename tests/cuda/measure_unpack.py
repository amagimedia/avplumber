"""Isolate v210_to_cuda unpack compute cost: N sources unpacked to P210 and
dropped (no compositor, no tonemap, no encode). Run detached while
`nvidia-smi dmon` samples SM/mem to attribute the mixer's compute.
"""
import argparse
from pathlib import Path
import time

from v210_fixture import COLOR, frame_stride, write_fixture

W, H = 1920, 1080


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", type=int, default=4)
    p.add_argument("--seconds", type=float, default=25)
    p.add_argument("--fixtures", default="/repo/out/fixtures")
    args = p.parse_args()
    from pyplumber import AVPlumber
    from pyplumber.node import Demux, FilterVideo, InputRec, V210ToCuda
    stride = frame_stride(W)
    avp = AVPlumber()
    avp.edges.planCapacity("*", 3)
    avp.executeCommandsFromString('hwaccel.init {"name":"g","type":"cuda"}')
    outs = []
    nodes = []
    for s in range(args.sources):
        path = Path(args.fixtures) / f"hlgmotion{s % 4}_1920x1080_300f.v210"
        nodes += [
            InputRec({"name": f"in{s}", "url": str(path), "format": "rawvideo", "dst": f"p{s}",
                      "loop": True, "options": {"pixel_format": "gray",
                      "video_size": f"{stride}x{H}", "framerate": "60"}}),
            Demux({"name": f"dx{s}", "src": f"p{s}", "routing": {"v:0": f"pk{s}"}}),
            V210ToCuda({"name": f"up{s}", "src": f"pk{s}", "dst": f"gpu{s}", "hwaccel": "g",
                        "width": W, "height": H, "stride": stride, "fps": "60/1",
                        "timebase": "1/90000", "format": "p210le", **COLOR["hlg"]}),
            FilterVideo({"name": f"dl{s}", "src": f"gpu{s}", "dst": f"out{s}", "hwaccel": "g",
                         "graph": "hwdownload,format=p210le"}),
        ]
        outs.append(f"out{s}")
    for n in nodes:
        n.parameters.update({"group": "g", "auto_restart": "off"})
        avp.addNode(n)
    del n
    edges = [avp.getEdge(o, "VideoFrame") for o in outs]
    avp.group("g").startNodes()
    print(f"unpacking {args.sources} sources for {args.seconds}s", flush=True)
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        for e in edges:
            e.tryGet(5)
    print("done", flush=True)


if __name__ == "__main__":
    main()
