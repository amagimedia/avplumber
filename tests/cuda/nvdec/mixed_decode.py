"""Compare one mixed NVDEC graph against its all-linear CUDA baseline.

Three simultaneous inputs share a CUDA device: HEVC switches to CUARRAY while
H264 and AV1 remain linear CUDA. Full source images of different sizes are
scaled into three canvas panels. Only the final verification boundary downloads
pixels. Run on the NVIDIA host with the built module on PYTHONPATH.
"""

import argparse
from fractions import Fraction
import gc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def run(args, mixed):
    from compositor_decode import EOF, api, make_avp, pixels, shutdown, timestamp

    avp, errors = make_avp("mixed_gpu", capacity=3)
    codecs = ("hevc", "h264", "av1")
    sources = {codec: [] for codec in codecs}
    ended = {codec: False for codec in codecs}
    storage = {codec: "cuarray" if mixed and codec == "hevc" else "cuda" for codec in codecs}
    nodes, records = [], []
    output = frame = None
    for codec in codecs:
        options = {"threads": 1, "extra_hw_frames": 3}
        if storage[codec] == "cuarray":
            options["hwaccel_flags"] = "unsafe_output"
        nodes += [
            api.Input({"name": f"input_{codec}", "url": str(getattr(args, codec)),
                       "dst": f"packets_{codec}"}),
            api.Demux({"name": f"demux_{codec}", "src": f"packets_{codec}",
                       "routing": {"v:0": f"video_{codec}"}}),
            api.DecVideo({"name": f"decode_{codec}", "src": f"video_{codec}",
                          "dst": f"decoded_{codec}", "codec": codec,
                          "pixel_format": storage[codec], "hwaccel": "mixed_gpu", "options": options}),
            api.FilterVideo({"name": f"normalize_{codec}", "src": f"decoded_{codec}",
                             "dst": f"source_{codec}", "hwaccel": "mixed_gpu", "threads": 1,
                             "defer_preliminary_init": True,
                             "graph": "settb=1/90000,setpts=PTS-STARTPTS,"
                                      "setparams=color_trc=bt709:color_primaries=bt709:colorspace=bt709:range=tv,"
                                      "scale_cuda=format=nv12"}),
        ]
    nodes += [
        api.CudaRectOverlay({"name": "compose", "src": [f"source_{c}" for c in codecs],
                             "dst": "canvas", "hwaccel": "mixed_gpu", "width": 960, "height": 180,
                             "sw_format": "nv12", "layers": [
                                 {"input": i, "dst_x": i * 320, "dst_y": 0,
                                  "dst_w": 320, "dst_h": 180} for i in range(3)]}),
        api.FilterVideo({"name": "verify", "src": "canvas", "dst": "result",
                         "hwaccel": "mixed_gpu", "threads": 1, "defer_preliminary_init": True,
                         "graph": "hwdownload,format=nv12"}),
    ]

    def observe(codec):
        def callback(frame):
            if frame.pts.timestamp == EOF:
                ended[codec] = True
            else:
                # Inspect metadata only; GPU data pointers are never dereferenced.
                sources[codec].append((*timestamp(frame), frame.width, frame.height, frame.format.name))
        return callback

    eof = False
    try:
        for node in nodes:
            node.parameters.update(group="mixed", auto_restart="off", on_error="off")
            avp.addNode(node)
        del node
        for codec in codecs:
            avp.getEdge(f"source_{codec}", "VideoFrame").addWiretapCallback(observe(codec))
        output = avp.getEdge("result", "VideoFrame")
        avp.group("mixed").startNodes()
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline and not errors:
            frame = output.tryGet(100)
            if frame is None:
                continue
            if frame.pts.timestamp == EOF:
                eof = True
                break
            assert (frame.width, frame.height) == (960, 180)
            records.append(pixels(frame))
            assert len(records) <= args.frames, "unexpected extra compositor frames"
        assert not errors, errors
        assert eof and all(ended.values()), f"missing EOF: output={eof}, inputs={ended}"
        assert len(records) == args.frames, len(records)
        output_times = [Fraction(p * n, d) for p, n, d, _ in records]
        assert output_times == sorted(set(output_times)), "duplicate or nonmonotonic canvas PTS"
        geometries = set()
        for codec, rows in sources.items():
            assert len(rows) == args.frames, (codec, len(rows))
            assert all(row[5] == storage[codec] for row in rows), (codec, rows)
            assert [Fraction(p * n, d) for p, n, d, *_ in rows] == output_times, codec
            geometries.add(rows[0][3:5])
        assert len(geometries) > 1, "supply fixtures with different source dimensions"
    finally:
        frame = output = None
        shutdown(avp, nodes)
        del avp
        gc.collect()
    return {"mixed": mixed, "storage": storage, "sources": sources, "outputs": records,
            "eof": eof, "shutdown_completed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for codec in ("hevc", "h264", "av1"):
        parser.add_argument("--" + codec, type=Path, required=True)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=30, help="decode deadline per graph")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.frames < 1 or args.timeout <= 0:
        parser.error("frames and timeout must be positive")
    if args.worker:
        baseline = run(args, False)
        actual = run(args, True)
        assert actual["outputs"] == baseline["outputs"], "mixed graph pixels/PTS differ from all-linear CUDA"
        args.worker.write_text(json.dumps({"ok": True, "inputs": {c: str(getattr(args, c))
                                               for c in ("hevc", "h264", "av1")},
                                          "baseline": baseline, "mixed": actual}, indent=2) + "\n")
        return 0
    # Cover native stalls and teardown as well as the Python drain deadline.
    with tempfile.TemporaryDirectory(prefix="avp-mixed-decode-") as directory:
        result = Path(directory) / "result.json"
        try:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                      *sys.argv[1:], "--worker", str(result)], timeout=2 * args.timeout + 40)
            report = json.loads(result.read_text()) if result.exists() else {"ok": False}
            if process.returncode:
                report.update(ok=False, failure=f"worker exit {process.returncode}")
        except subprocess.TimeoutExpired:
            report = {"ok": False, "failure": "worker timed out; killed (includes shutdown)"}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print("PASS mixed HEVC CUARRAY/H264 CUDA/AV1 CUDA: exact canvas pixels/PTS, EOF and shutdown"
              if report["ok"] else json.dumps(report), flush=True)
        return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
