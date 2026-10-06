"""Real NVDEC frames through normalization and the unclocked CUDA compositor.

Compare every CPU-boundary pixel hash and PTS against linear CUDA, with two
outputs sharing frame references. Wiretaps inspect handles, never GPU pixels;
unchanged handles prove the format-only scale_cuda filter remains passthrough.
Opaque decoders use private producer streams; the unclocked compositors use the
shared device stream. --filter-linear also exercises producer ordering after
CUDA pad/crop convert opaque frames into ordinary linear CUDA frames.

Run on the NVIDIA host with its built module on PYTHONPATH and --input CLIP.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _harness import EOF, make_avp
from pyplumber import node as api


def timestamp(frame):
    pts = frame.pts
    return (pts.timestamp, pts.timebase.num, pts.timebase.den)


def pixels(frame):
    assert frame.format.name == "nv12", frame.format
    digest = hashlib.sha256()
    planes, pitches = frame.data, frame.linesize
    for plane, rows in enumerate((frame.height, (frame.height + 1) // 2)):
        view = memoryview(planes[plane])
        width = frame.width if plane == 0 else ((frame.width + 1) // 2) * 2
        for row in range(rows):
            digest.update(view[row * pitches[plane]:row * pitches[plane] + width])
    return (*timestamp(frame), digest.hexdigest())


def shutdown(avp, nodes, timeout=15):
    # The general smoke helper tolerates a leaked instance; this test must prove
    # repeated decoder/pool teardown, so a hung shutdown is a test failure.
    nodes.clear()
    worker = threading.Thread(target=avp.shutdown, daemon=True)
    worker.start()
    worker.join(timeout)
    assert not worker.is_alive(), "AVP shutdown did not release the decoder and compositor pools"


def run(args, storage):
    avp, errors = make_avp("decode_gpu", capacity=3)
    decoded, normalized = [], []
    nodes = []
    options = {"threads": 1}
    if storage == "cuarray":
        options.update(hwaccel_flags="unsafe_output", extra_hw_frames=3)
    graph = ("setparams=color_trc=bt709:color_primaries=bt709:colorspace=bt709:range=tv,"
             "scale_cuda=format=nv12")
    if args.filter_linear:
        graph += ",pad_cuda=w=iw+32:h=ih+16:x=16:y=8,crop_cuda=w=iw-32:h=ih-16:x=16:y=8"
    nodes += [
        api.Input({"name": "input", "url": str(args.input), "dst": "packets"}),
        api.Demux({"name": "demux", "src": "packets", "routing": {"v:0": "video_packets"}}),
        api.DecVideo({"name": "decode", "src": "video_packets", "dst": "decoded",
                      "hwaccel": "decode_gpu", "pixel_format": storage, "options": options}),
        api.FilterVideo({"name": "normalize", "src": "decoded", "dst": "normalized",
                         "hwaccel": "decode_gpu", "threads": 1, "defer_preliminary_init": True,
                         "graph": graph}),
        api.Split({"name": "split", "src": "normalized", "dst": ["source_0", "source_1"],
                   "data_type": "video"}),
    ]
    geometry = [(args.width, args.height), (args.width // 2, args.height // 2)]
    for index, (width, height) in enumerate(geometry):
        layers = [{"input": 0, "dst_x": 0, "dst_y": 0, "dst_w": width, "dst_h": height}]
        if index:
            # Reuse the same decoded array twice in a draw, with crop and z-order.
            layers.append({"input": 0, "crop_x": 16, "crop_y": 8,
                           "crop_w": args.width - 32, "crop_h": args.height - 16,
                           "dst_x": width // 4, "dst_y": height // 4,
                           "dst_w": width // 2, "dst_h": height // 2, "z": 1})
        nodes += [
            api.CudaRectOverlay({"name": f"compose_{index}", "src": [f"source_{index}"],
                                 "dst": f"canvas_{index}", "hwaccel": "decode_gpu",
                                 "width": width, "height": height, "sw_format": "nv12", "layers": layers}),
            api.FilterVideo({"name": f"verify_{index}", "src": f"canvas_{index}",
                             "dst": f"result_{index}", "hwaccel": "decode_gpu",
                             "graph": "hwdownload,format=nv12", "threads": 1, "defer_preliminary_init": True}),
        ]

    def observe(records, expected):
        def callback(frame):
            if frame.pts.timestamp != EOF:
                assert frame.format.name == expected, (expected, frame.format)
                records.append((*timestamp(frame), frame.data_ptr[0]))
        return callback

    result = [[], []]
    outputs = []
    try:
        for node in nodes:
            node.parameters.update(group="check", auto_restart="off")
            avp.addNode(node)
        del node
        avp.getEdge("decoded", "VideoFrame").addWiretapCallback(observe(decoded, storage))
        avp.getEdge("normalized", "VideoFrame").addWiretapCallback(
            observe(normalized, "cuda" if args.filter_linear else storage))
        outputs = [avp.getEdge(f"result_{index}", "VideoFrame") for index in range(2)]
        avp.group("check").startNodes()
        ended = [False, False]
        deadline = time.monotonic() + args.timeout
        while not all(ended) and time.monotonic() < deadline and not errors:
            for index, output in enumerate(outputs):
                if ended[index]:
                    continue
                frame = output.tryGet(10)
                if frame is None:
                    continue
                if frame.pts.timestamp == EOF:
                    ended[index] = True
                else:
                    result[index].append(pixels(frame))
        assert not errors, errors
        assert all(ended), f"missing EOF: {ended}; output counts {[len(items) for items in result]}"
        assert all(len(items) == args.frames for items in result), [len(items) for items in result]
        assert len(decoded) == args.frames and len(normalized) == args.frames
        assert [item[:3] for item in decoded] == [item[:3] for item in normalized], "normalization changed PTS"
        if not args.filter_linear:
            assert decoded == normalized, "identity normalization changed frame handles"
        assert all(len({item[:3] for item in items}) == args.frames for items in result), "duplicate output PTS"
        surfaces = len({item[3] for item in decoded})
        if storage == "cuarray":
            assert surfaces < args.frames, "CUarray surface pool did not reuse any array"
        return {"storage": storage, "filter_linear": args.filter_linear,
                "frames": args.frames, "unique_surfaces": surfaces,
                "outputs": result}
    finally:
        frame = output = None
        outputs.clear()
        shutdown(avp, nodes)
        del avp
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--filter-linear", action="store_true", help="exercise private-stream linear pad/crop output")
    parser.add_argument("--rounds", type=int, default=2, help="CUarray recreate rounds; 0 runs the FFmpeg 8 baseline only")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    baseline = run(args, "cuda")
    print(f"PASS linear CUDA: {args.frames} frames × 2 outputs, EOF and shutdown", flush=True)
    records = [baseline]
    for iteration in range(args.rounds):
        actual = run(args, "cuarray")
        assert actual["outputs"] == baseline["outputs"], f"CUarray pixels/PTS differ in round {iteration + 1}"
        records.append(actual)
        path = "linear pad/crop output" if args.filter_linear else "identity normalization without copies"
        print(f"PASS round {iteration + 1}: {args.frames} frames × 2 outputs, exact pixels/PTS, "
              f"{actual['unique_surfaces']} array handles reused, {path}", flush=True)
    if args.report:
        args.report.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
