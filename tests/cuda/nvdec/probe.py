"""Finite AVP decode probe; run on the build host with a supplied video fixture.

Compare --action pixels reports using --reference; --action hold keeps frame
references without downloading or reading GPU pixels. CUDA is the copying
baseline; CUARRAY uses native NVDEC unsafe_output. Memory samples are whole-GPU
usage, including unrelated processes. Every run has a subprocess watchdog,
including shutdown; timeout or native failure is never reported as success.
"""

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time


def pixel_hash(frame, fmt):
    # Only called after the explicit CPU verification boundary. Ignore pitch padding.
    assert frame.format.name == fmt, (frame.format.name, fmt)
    sample_bytes = 1 if fmt == "nv12" else 2
    chroma_height = frame.height if fmt == "p210le" else (frame.height + 1) // 2
    widths = (frame.width * sample_bytes, ((frame.width + 1) // 2) * 2 * sample_bytes)
    digest = hashlib.sha256()
    for plane, pitch, width, height in zip(frame.data[:2], frame.linesize[:2], widths,
                                           (frame.height, chroma_height)):
        assert pitch >= width and len(plane) >= pitch * height
        for y in range(height):
            digest.update(plane[y * pitch:y * pitch + width])
    return digest.hexdigest()


def snapshot(gpu):
    threads = Path("/proc/self/task")
    result = {"at": time.monotonic(), "cpu_seconds": time.process_time(),
              "threads": len(list(threads.iterdir())) if threads.exists() else None}
    if gpu:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits",
        ], text=True, timeout=5)
        result["whole_gpu_memory_mib"] = {
            uuid.strip(): int(memory) for uuid, memory in
            (line.split(",") for line in output.splitlines())}
    return result


def worker(args):
    from pyplumber import AVPlumber
    from pyplumber.node import DecVideo, Demux, FilterVideo, Input

    avp, errors = AVPlumber(), []
    if args.verbose:
        import ctypes
        import ctypes.util
        library = "libavutil.so" if sys.platform.startswith("linux") else ctypes.util.find_library("avutil")
        avutil = ctypes.CDLL(library)
        avutil.av_log_set_level.argtypes = [ctypes.c_int]
        avutil.av_log_set_level.restype = None
        avutil.av_log_set_level(40)  # AV_LOG_VERBOSE; uses the worker's library environment.
    avp.on_exception = lambda *error: errors.append(tuple(map(str, error)))
    avp.edges.planCapacity("*", 3)
    avp.edges.planCapacity("decoded", 3)
    gpu = args.mode != "cpu"
    if gpu:
        avp.executeCommandsFromString('hwaccel.init {"name":"probe_gpu","type":"cuda"}')
    options = {"threads": 1}
    decode = {"src": "video_packets", "dst": "decoded", "options": options}
    if args.codec:
        decode["codec"] = args.codec
    if gpu:
        decode.update(hwaccel="probe_gpu", pixel_format="cuarray" if args.mode == "cuarray" else "cuda")
        options["extra_hw_frames"] = args.extra_hw_frames
        if args.mode == "cuarray":
            options["hwaccel_flags"] = "unsafe_output"
    nodes = [Input({"name": "input", "url": str(args.input), "dst": "packets"}),
             Demux({"name": "demux", "src": "packets", "routing": {"v:0": "video_packets"}}),
             DecVideo({"name": "decode", **decode})]
    edge_name = "decoded"
    if args.action == "pixels":
        # Intentional GPU/CPU interop for assertions only; retain mode has no filter/copy.
        graph = ("hwdownload," if gpu else "") + "format=" + args.sw_format
        nodes.append(FilterVideo({"name": "verification", "src": "decoded", "dst": "pixels",
                                  "graph": graph, "threads": 1}))
        edge_name = "pixels"
    for node in nodes:
        node.parameters.update(group="probe", auto_restart="off", on_error="off")
        avp.addNode(node)
    edge = avp.getEdge(edge_name, "VideoFrame")
    retained, records, samples = deque(maxlen=args.hold), [], []
    stopped = threading.Event()
    sample_errors = []

    def sample():
        try:
            samples.append(snapshot(gpu))
        except Exception as error:
            sample_errors.append(str(error))

    def sampler():
        while not stopped.wait(.5):
            sample()

    sample()
    monitor = threading.Thread(target=sampler, daemon=True)
    monitor.start()
    report = {"ok": False, "input": str(args.input), "codec": args.codec,
              "mode": args.mode, "action": args.action,
              "decoder_options": options, "verbose": args.verbose,
              "extra_hw_frames": args.extra_hw_frames if gpu else None,
              "hold": args.hold, "decoded_edge_capacity": 3, "frames": records,
              "samples": samples, "errors": errors, "sampling_errors": sample_errors}
    start = time.monotonic()
    cpu_start = time.process_time()
    eof = False
    try:
        avp.group("probe").startNodes()
        deadline = start + max(1, args.timeout - 5)
        while time.monotonic() < deadline:
            assert not errors, errors
            frame = edge.tryGet(100)
            if frame is None:
                continue
            if frame.pts.timestamp == -(1 << 63):
                eof = True
                del frame
                break
            assert frame.pts.timebase.den > 0 and frame.width > 0 and frame.height > 0
            row = {"pts": frame.pts.timestamp,
                   "timebase": [frame.pts.timebase.num, frame.pts.timebase.den],
                   "width": frame.width, "height": frame.height}
            if args.action == "pixels":
                row["sha256"] = pixel_hash(frame, args.sw_format)
            else:
                if gpu:
                    assert frame.format.name == decode["pixel_format"], frame.format.name
                retained.append(frame)  # AVFrame references only; never access GPU frame.data.
            records.append(row)
            del frame
            assert len(records) <= args.expected_frames, "more frames than expected"
        assert not errors, errors
        assert eof, "no EOF before deadline (possible decode/surface exhaustion)"
        assert len(records) == args.expected_frames, (len(records), args.expected_frames)
        if args.reference:
            reference = json.loads(args.reference.read_text())
            assert reference["ok"] and reference["action"] == "pixels", "invalid pixel reference"
            assert records == reference["frames"], "pixel hashes, dimensions or PTS differ from reference"
        report["decode_passed"] = True
    except Exception as error:
        report["failure"] = str(error)
    finally:
        report.update(eof=eof, count=len(records), retained_refs=len(retained),
                      elapsed_seconds=time.monotonic() - start,
                      cpu_seconds=time.process_time() - cpu_start)
        stopped.set()
        monitor.join(6)
        sample()  # Includes retained frames before release/teardown.
        args.worker.write_text(json.dumps(report, indent=2))
        retained.clear()
        avp.shutdown()  # Parent watchdog fails the run if this hangs.
    report["shutdown_completed"] = True
    report["ok"] = bool(report.get("decode_passed")) and not errors and not sample_errors
    args.worker.write_text(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--mode", choices=("cpu", "cuda", "cuarray"), required=True)
    parser.add_argument("--action", choices=("pixels", "hold"), default="pixels")
    parser.add_argument("--codec", choices=("h264", "hevc", "av1"))
    parser.add_argument("--sw-format", choices=("nv12", "p010le", "p210le"), default="nv12")
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--extra-hw-frames", type=int, default=3)
    parser.add_argument("--hold", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--verbose", action="store_true", help="enable FFmpeg verbose logging")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.expected_frames < 1 or args.hold < 0 or args.extra_hw_frames < -1 or args.timeout <= 5:
        parser.error("frame count must be positive, hold nonnegative, extra frames >=-1, timeout >5 seconds")
    if args.reference and args.action != "pixels":
        parser.error("--reference requires --action pixels")
    if args.worker:
        return worker(args)
    with tempfile.TemporaryDirectory(prefix="avp-nvdec-probe-") as temporary:
        result = Path(temporary) / "result.json"
        try:
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                        *sys.argv[1:], "--worker", str(result)], timeout=args.timeout)
            failure = None if completed.returncode == 0 else f"worker exit {completed.returncode}"
        except subprocess.TimeoutExpired:
            failure = "worker timed out, including startup/decode/shutdown; worker killed"
        report = json.loads(result.read_text()) if result.exists() else {"ok": False}
        if failure:
            report.update(ok=False, worker_failure=failure)
        if not report.get("shutdown_completed"):
            report["ok"] = False
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps({key: value for key, value in report.items() if key not in ("frames", "samples")}))
        return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
