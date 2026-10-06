"""Compare pinned and pageable FFmpeg CPU/GPU upload paths on an NVIDIA host.

All modes use the same looped packets, post-upload pacing and CUDA scaler.
The scaler exercises output-pool reuse while GPU reads are queued. This is an
upload microbenchmark, not a full mixer capacity qualification.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import tempfile
import time

from _harness import finish, make_avp
from pyplumber import node as api
from pyplumber.mixer.inputs import _pace, _raw_file_packets


def run(args):
    width, height = args.width, args.height
    sample = 1 if args.format == "nv12" else 2
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "source.raw"
        # Four complete pictures, resident in the page cache before measurement.
        path.write_bytes(bytes([64]) * (width * height * 3 // 2 * sample * 4))
        avp, errors = make_avp("upload_bench", capacity=2)
        nodes, outputs = [], []
        try:
            for i in range(args.inputs):
                tag, group = str(i), "probe"
                packets = _raw_file_packets(avp, api, tag, str(path), pixel_format=args.format,
                                           video_size=f"{width}x{height}", group=group,
                                           fps=args.fps, fps_den=1, loop=True)
                uploaded = f"input_{tag}_cuda"
                avp.edges.planCapacity(uploaded, 1)
                common = {"name": f"upload_{tag}", "dst": uploaded, "hwaccel": "upload_bench", "group": group}
                decoded = f"input_{tag}_decoded"
                chain = [api.DecVideo({"name": f"decode_{tag}", "src": packets, "dst": decoded,
                                       "codec": "rawvideo", "pixel_format": args.format, "group": group}),
                         api.FilterVideo({**common, "src": decoded, "threads": 1,
                                          "graph": "hwupload_cuda=pinned=1" if args.mode == "filter" else "hwupload"})]
                for node in chain:
                    nodes.append(node)
                    avp.addNode(node)
                paced = _pace(avp, api, tag, uploaded, fps=args.fps, fps_den=1, group=group, event_loop=None)
                edge = f"scaled_{tag}"
                for node in (api.FilterVideo({"name": f"scale_{tag}", "src": paced, "dst": edge,
                                              "graph": "scale_cuda=1280:720:passthrough=0", "threads": 1,
                                              "hwaccel": "upload_bench", "group": group}),
                             api.NullSink({"name": f"sink_{tag}", "src": edge, "group": group})):
                    nodes.append(node)
                    avp.addNode(node)
                outputs.append(avp.getEdge(edge, "VideoFrame"))
            avp.group("probe").startNodes()
            time.sleep(args.warmup)
            assert not errors, errors
            if args.ready:
                Path(args.ready).write_text(str(os.getpid()))
            start_counts = [edge.enqueued_total for edge in outputs]
            before = resource.getrusage(resource.RUSAGE_SELF)
            started = time.monotonic()
            time.sleep(args.seconds)
            elapsed = time.monotonic() - started
            after = resource.getrusage(resource.RUSAGE_SELF)
            counts = [edge.enqueued_total - initial for edge, initial in zip(outputs, start_counts)]
            result = dict(mode=args.mode, inputs=args.inputs, format=args.format, width=width, height=height,
                          target_fps=args.fps, seconds=elapsed, frames=counts,
                          fps=[count / elapsed for count in counts],
                          cpu_seconds=(after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime),
                          voluntary_switches=after.ru_nvcsw - before.ru_nvcsw,
                          involuntary_switches=after.ru_nivcsw - before.ru_nivcsw, errors=errors)
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            assert not errors, errors
            assert min(result["fps"]) >= args.fps * .98, "throughput shortfall invalidates a CPU comparison"
        finally:
            finish(avp, nodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("filter", "pageable"), required=True)
    parser.add_argument("--inputs", type=int, default=8)
    parser.add_argument("--format", choices=("nv12", "p010le"), default="p010le")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--warmup", type=float, default=5)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--ready")
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
