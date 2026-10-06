"""Compare camera-motion metadata for linear and CUarray NV12 NVDEC input.

Run on an NVIDIA host with HAVE_NVOF=1 and a finite 8-bit HEVC --input clip.
The gpu_irls backend also needs HAVE_NVCC=1. No image downloads or encodes occur
in this graph. Wiretaps prove that the node forwards the original GPU storage.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _harness import EOF, make_avp
from compositor_decode import shutdown, timestamp
from pyplumber import node as api


class AlternateStorage(api.PythonNode):
    """Keep one decoded frame per PTS, alternating producer devices/streams."""
    index = 0

    def process(self):
        frames = [edge.get() for edge in self._src.values()]
        assert timestamp(frames[0]) == timestamp(frames[1]), "decoder PTS mismatch"
        self._dst.enqueue(frames[self.index % 2])
        self.index += 1


def run(args, storage, backend, *, filter_linear=False, unsupported=False, lookahead=None):
    avp, errors = make_avp("motion_gpu", capacity=2)
    path = args.unsupported_input if unsupported else args.input
    storages = ("cuda", "cuarray") if storage == "mixed" else (storage,)
    nodes, sources = [], []
    for index, pixel_format in enumerate(storages):
        options = {"threads": 1, "extra_hw_frames": (lookahead or 0) + 3}
        if pixel_format == "cuarray":
            options["hwaccel_flags"] = "unsafe_output"
        source = f"decoded_{index}"
        nodes.extend([
            api.Input({"name": f"input_{index}", "url": str(path), "dst": f"packets_{index}"}),
            api.Demux({"name": f"demux_{index}", "src": f"packets_{index}",
                       "routing": {"v:0": f"video_packets_{index}"}}),
            api.DecVideo({"name": f"decode_{index}", "src": f"video_packets_{index}", "dst": source,
                          "hwaccel": "motion_gpu", "pixel_format": pixel_format, "options": options}),
        ])
        sources.append(source)
    if len(sources) > 1:
        source = "alternating"
        nodes.append(AlternateStorage({"name": "alternate", "src": sources, "dst": source}))
    if filter_linear:
        # A CUarray decoder's private stream survives a filter's linear output.
        nodes.append(api.FilterVideo({"name": "linearize", "src": source, "dst": "linear",
                                     "hwaccel": "motion_gpu", "defer_preliminary_init": True,
                                     "graph": "pad_cuda=w=iw+32:h=ih+16:x=16:y=8,"
                                              "crop_cuda=w=iw-32:h=ih-16:x=16:y=8"}))
        source = "linear"
    analysis = {"name": "analysis", "src": source, "dst": "result", "strict_cuda": not unsupported}
    nodes.append(api.CudaCameraMotion({**analysis, "affine_backend": backend}) if lookahead is None
                 else api.LumaDiff({**analysis, "frames_lookahead": lookahead}))
    observed, forwarded, records = [], [], []

    def observe(frame):
        if frame.pts.timestamp != EOF:
            observed.append((timestamp(frame), frame.format.name, frame.data_ptr[0]))

    output = frame = None
    try:
        for node in nodes:
            node.parameters.update(group="check", auto_restart="off", on_error="off")
            avp.addNode(node)
        del node
        avp.getEdge(source, "VideoFrame").addWiretapCallback(observe)
        output = avp.getEdge("result", "VideoFrame")
        avp.group("check").startNodes()
        deadline = time.monotonic() + args.timeout
        ended = False
        while time.monotonic() < deadline and not errors:
            frame = output.tryGet(50)
            if frame is None:
                continue
            if frame.pts.timestamp == EOF:
                ended = True
                break
            index = len(records)
            forwarded.append((timestamp(frame), frame.format.name, frame.data_ptr[0]))
            assert frame.format.name == ("cuda" if filter_linear else storages[index % len(storages)])
            if lookahead is None:
                metadata = json.loads(frame.metadata["camera_motion"])
                assert metadata["frame_index"] == index
                assert metadata["status"] == ("unsupported_sw_format" if unsupported else "ok"), metadata
                assert metadata["has_prev"] == (index > 0 and not unsupported), metadata
            else:
                metadata = [json.loads(frame.metadata["scene_diff" + (f"+{k}" if k else "")])
                            for k in range(lookahead + 1)]
                assert all(item["frame_index"] == index for item in metadata)
            records.append((timestamp(frame), metadata))
        assert not errors, errors
        assert ended, "analysis did not deliver EOF"
        # Queue publication can precede its wiretap callback. Compare after EOF,
        # when all source callbacks preceding that marker have completed.
        assert forwarded == observed, "analysis changed PTS, storage format or source handle"
        assert len(records) == len(observed) and len(records) > 1
        if not unsupported:
            assert len(records) == args.frames, (len(records), args.frames)
            if lookahead is None:
                assert all(r[1]["affine_point_count"] > 0 for r in records[1:]), "no motion was analyzed"
        return records
    finally:
        frame = output = None
        shutdown(avp, nodes)
        del avp
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--backends", nargs="+", default=["tx_median", "cpu_irls", "gpu_irls"])
    parser.add_argument("--unsupported-input", type=Path, help="optional Main10 HEVC fixture: verify explicit skip status")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--linear-only", action="store_true", help="run against an unmodified module")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reference", type=Path, help="compare linear metadata against a saved build")
    args = parser.parse_args()
    report = {}
    for backend in args.backends:
        baseline = run(args, "cuda", backend)
        report[backend] = baseline
        for _ in range(0 if args.linear_only else args.rounds):
            for filter_linear in (False, True):
                actual = run(args, "cuarray", backend, filter_linear=filter_linear)
                assert actual == baseline, f"metadata/PTS mismatch: {backend}, filter_linear={filter_linear}"
        if not args.linear_only:
            assert run(args, "mixed", backend) == baseline, f"stream/storage switching mismatch: {backend}"
        print(f"PASS {backend}: {args.frames} frames, metadata/PTS, original handles, EOF; "
              + ("linear only" if args.linear_only else "CUarray and private-stream linear parity, repeated shutdown"),
              flush=True)
    if args.unsupported_input and not args.linear_only:
        run(args, "cuarray", "tx_median", unsupported=True)
        print("PASS Main10 CUarray is explicitly skipped in non-strict mode", flush=True)
    # JSON normalization accounts only for tuples becoming lists in saved reports.
    if args.reference:
        assert json.loads(json.dumps(report)) == json.loads(args.reference.read_text()), "linear build regression"
        print("PASS exact linear metadata/PTS against reference build", flush=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
