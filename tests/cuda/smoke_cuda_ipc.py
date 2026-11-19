"""Exercise the AVPlumber VMM nodes in separate processes, optionally containers."""
import argparse
from pathlib import Path
import time

from pyplumber import AVPlumber, node as api
from pyplumber.mixer.color import Color
from pyplumber.mixer.inputs import build_raw420_input

WIDTH, HEIGHT, FPS = 160, 96, 30
GROUP = "cuda_ipc_test"
DEVICE = "gpu"


class CudaIpcSink(api.InternalNode):
    TYPE = "cuda_ipc_sink"


class CudaIpcSource(api.InternalNode):
    TYPE = "cuda_ipc_source"


def make_avp():
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(tuple(map(str, error)))
    avp.edges.planCapacity("*", 2)
    avp.executeCommandsFromString(f'hwaccel.init {{"name":"{DEVICE}","type":"cuda"}}')
    return avp, errors


def drain(edge, errors, timeout):
    finish_at = time.monotonic() + timeout
    while time.monotonic() < finish_at and not errors:
        frame = edge.tryGet(100)
        if frame is not None:
            yield frame


def pattern(luma, fmt):
    def plane(value, count):
        pixel = bytes([value]) if fmt == "nv12" else (value << 8).to_bytes(2, "little")
        return pixel * count

    return plane(luma, WIDTH * HEIGHT) + plane(128, WIDTH * HEIGHT // 2)


def produce(socket, raw, fmt, luma, duration, export):
    Path(raw).write_bytes(pattern(luma, fmt) * 20)
    avp, errors = make_avp()
    edge = build_raw420_input(avp, api, "pattern", raw, width=WIDTH, height=HEIGHT,
                              pixel_format=fmt, fps=FPS, group=GROUP, hwaccel=DEVICE,
                              loop=True, color=Color())
    nodes = [
        api.MixerCompositor({
            "name": "export_canvas", "src": [edge], "dst": "canvas", "group": GROUP,
            "hwaccel": DEVICE, "width": WIDTH, "height": HEIGHT, "sw_format": fmt, "color": "sdr",
            "fps": f"{FPS}/1", "active_inputs": 1, "layers": [{"input": 0}],
            "vmm_slots": 6 if export else 0, "gpu_handoff": False,
        }),
        CudaIpcSink({"name": "export", "group": GROUP, "src": "canvas", "socket": socket, "hwaccel": DEVICE}),
    ]
    for node in nodes:
        avp.addNode(node)
    try:
        avp.group(GROUP).startNodes()
        until = time.monotonic() + duration
        while time.monotonic() < until and not errors:
            time.sleep(.02)
        if export:
            assert not errors, errors
        else:
            assert any("requires uncropped frames from the VMM pool" in " ".join(error) for error in errors), errors
            print("PASS: ordinary CUDA frames rejected", flush=True)
    finally:
        avp.shutdown()


def consume(socket, fmt, expected):
    avp, errors = make_avp()
    nodes = [
        CudaIpcSource({"name": "import", "group": GROUP, "dst": "imported", "socket": socket, "hwaccel": DEVICE}),
        api.FilterVideo({"name": "check", "group": GROUP, "src": "imported", "dst": "pixels",
                         "hwaccel": DEVICE, "graph": f"hwdownload,format={fmt}"}),
    ]
    for node in nodes:
        avp.addNode(node)
    try:
        output = avp.getEdge("pixels", "VideoFrame")
        avp.group(GROUP).startNodes()
        wanted = list(expected)
        consecutive = 0
        previous = None
        for frame in drain(output, errors, timeout=25):
            assert (frame.width, frame.height) == (WIDTH, HEIGHT)
            value = bytes(frame.data[0])[int(fmt != "nv12")]
            if value != wanted[0]:
                consecutive = 0
                continue
            pts = frame.pts
            seconds = pts.timestamp * pts.timebase.num / pts.timebase.den
            assert previous is None or seconds > previous, (previous, seconds)
            previous = seconds
            consecutive += 1
            if consecutive == 8:
                print(f"PASS: {fmt} luma {value} and increasing PTS after import", flush=True)
                wanted.pop(0)
                consecutive = 0
                if not wanted:
                    return
        raise AssertionError(f"missing expected luma values: {wanted}")
    finally:
        avp.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("producer", "consumer"))
    parser.add_argument("socket")
    parser.add_argument("--raw", default="/tmp/cuda-ipc-pattern.raw")
    parser.add_argument("--format", choices=("nv12", "p010le"), default="nv12")
    parser.add_argument("--luma", type=int, nargs="+", default=[72])
    parser.add_argument("--duration", type=float, default=3)
    parser.add_argument("--reject-ordinary", action="store_true")
    args = parser.parse_args()
    if args.role == "producer":
        produce(args.socket, args.raw, args.format, args.luma[0], args.duration, not args.reject_ordinary)
    else:
        consume(args.socket, args.format, args.luma)
