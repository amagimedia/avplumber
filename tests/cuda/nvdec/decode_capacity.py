"""How many decoders of a stream type fit: engine load and memory per real-time stream.

Runs --streams real-time decoders of each clip together and reads NVDEC load and device memory, so
codecs and shapes can be compared on one GPU. Needs an NVIDIA host with nothing else on the GPU.

    python3 tests/cuda/nvdec/decode_capacity.py --clip hevc=<file> --clip h264=<file> --output capacity.json
"""
import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path


def gpu():
    fields = "utilization.decoder,utilization.gpu,memory.used"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return [float(v) for v in out.split(",")]


def measure(ffmpeg, clip, streams, seconds, extra):
    idle = gpu()[2]
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-hwaccel", "nvdec", "-hwaccel_output_format", "cuda",
               "-extra_hw_frames", str(extra), "-re", "-stream_loop", "-1", "-i", clip, "-map", "0:v", "-f", "null", "-"]
    decoders = [subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) for _ in range(streams)]
    try:
        time.sleep(6)
        samples = []
        for _ in range(seconds):
            samples.append(gpu())
            time.sleep(1)
        alive = sum(d.poll() is None for d in decoders)
    finally:
        for d in decoders:
            d.terminate()
        errors = [d.communicate(timeout=10)[1][-200:].decode(errors="replace") for d in decoders]
    load = statistics.mean(s[0] for s in samples)
    memory = statistics.mean(s[2] for s in samples) - idle
    per_stream = load / streams
    return {"streams": streams, "alive": alive, "decoder_load_pct": round(load, 1), "gpu_pct": round(statistics.mean(s[1] for s in samples), 1),
            "load_per_stream_pct": round(per_stream, 3), "mib_per_stream": round(memory / streams, 1),
            "streams_at_90pct": int(90 / per_stream) if per_stream else None,
            "errors": sorted({e.strip() for e in errors if e.strip()})[:3]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", action="append", required=True, help="name=path")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--streams", type=int, default=30)
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--extra-hw-frames", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = {}
    for entry in args.clip:
        name, path = entry.split("=", 1)
        results[name] = measure(args.ffmpeg, path, args.streams, args.seconds, args.extra_hw_frames)
        print(name, json.dumps(results[name]), flush=True)
        time.sleep(3)
    if args.output:
        args.output.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
