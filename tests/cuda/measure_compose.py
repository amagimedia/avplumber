"""Profile the compositor + tonemap compute path (the real SM consumers).
N v210 sources -> cuda_rect_overlay P210 grid -> scale_cuda=p010 ->
[tonemap_cuda] -> hwdownload/drain. Run under nsys --stats; read the
cuda_gpu_kern_sum report (kernel time is separate from the drain memcpy).
"""
import argparse
from pathlib import Path
import time

from v210_fixture import COLOR, frame_stride

W, H = 1920, 1080


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", type=int, default=4)
    p.add_argument("--seconds", type=float, default=14)
    p.add_argument("--tonemap", action="store_true")
    p.add_argument("--fixtures", default="/repo/out/fixtures")
    args = p.parse_args()
    from pyplumber import AVPlumber
    from pyplumber.node import CudaRectOverlay, Demux, FilterVideo, InputRec, V210ToCuda
    stride = frame_stride(W)
    avp = AVPlumber()
    avp.edges.planCapacity("*", 3)
    avp.executeCommandsFromString('hwaccel.init {"name":"g","type":"cuda"}')
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
        ]
    cols = 4 if args.sources > 1 else 1
    tw, th = W // cols, H // cols
    graph = "scale_cuda=format=p010le"
    if args.tonemap:
        graph += ",tonemap_cuda=transfer=hlg:tonemap=hable:peak=10"
    graph += ",hwdownload,format=" + ("nv12" if args.tonemap else "p010le")
    nodes += [
        CudaRectOverlay({"name": "comp", "src": [f"gpu{s}" for s in range(args.sources)],
                         "dst": "scene", "hwaccel": "g", "width": W, "height": H,
                         "sw_format": "p210le", "scale": True, "fps": "60/1",
                         "active_inputs": (1 << args.sources) - 1,
                         "layers": [{"dst_x": (s % cols) * tw, "dst_y": (s // cols) * th,
                                     "dst_w": tw, "dst_h": th} for s in range(args.sources)]}),
        FilterVideo({"name": "out", "src": "scene", "dst": "res", "hwaccel": "g", "graph": graph}),
    ]
    for n in nodes:
        n.parameters.update({"group": "g", "auto_restart": "off"})
        avp.addNode(n)
    del n
    res = avp.getEdge("res", "VideoFrame")
    avp.group("g").startNodes()
    print(f"compose {args.sources} src tonemap={args.tonemap} for {args.seconds}s", flush=True)
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        res.tryGet(5)
    print("done", flush=True)


if __name__ == "__main__":
    main()
