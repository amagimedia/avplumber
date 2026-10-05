#!/usr/bin/env python3
"""GPU pixel regressions for CUarray filter inputs, including mixed transitions.

Uploads/downloads are test fixture and comparison boundaries only. Production
filters must consume arrays directly without intermediate linear input images.
Run on the NVIDIA build host after applying the CUarray filter patch.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


FORMATS = {"nv12": (1, 1, 0), "p010le": (2, 1, 6), "p210le": (2, 0, 6)}
MODES = ("fade", "wipe_left", "wipe_right", "wipe_down", "wipe_up", "dip")
WIDTH, HEIGHT, FRAMES = 70, 38, 10


def fixture(fmt: str, variant: int) -> bytes:
    """Nonuniform luma and distinct interleaved U/V expose plane/address errors."""
    size, sub_y, shift = FORMATS[fmt]
    result = array.array("B" if size == 1 else "H")
    mask = (1 << (8 if size == 1 else 10)) - 1
    for frame in range(FRAMES):
        for plane, height in ((0, HEIGHT), (1, HEIGHT >> sub_y)):
            for y in range(height):
                for x in range(WIDTH):
                    value = x * 13 + y * 31 + frame * 43 + variant * 89
                    value += plane * (71 if x % 2 else 193)
                    result.append((value & mask) << shift)
    if size == 2 and sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


def run(ffmpeg: str, inputs: list[Path], fmt: str, graph: str, output_format: str | None = None) -> bytes:
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
           "-init_hw_device", "cuda=gpu:0", "-filter_hw_device", "gpu",
           "-filter_complex_threads", "1"]
    for path in inputs:
        cmd += ["-f", "rawvideo", "-pixel_format", fmt, "-video_size", f"{WIDTH}x{HEIGHT}",
                "-framerate", "25", "-i", str(path)]
    cmd += ["-filter_complex", graph, "-map", "[out]", "-frames:v", str(FRAMES),
            "-threads", "1", "-c:v", "rawvideo", "-pix_fmt", output_format or fmt, "-f", "rawvideo", "pipe:1"]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if completed.returncode:
        raise RuntimeError(completed.stderr.decode(errors="replace"))
    if not completed.stdout:
        raise RuntimeError("FFmpeg produced no comparison frames")
    return completed.stdout


def single_graph(storage: str, operation: str, fmt: str) -> str:
    return f"[0:v]hwupload,format={storage},{operation},hwdownload,format={fmt}[out]"


def transition_graph(first: str, second: str, mode: str, fmt: str) -> str:
    return (f"[0:v]hwupload,format={first}[a];[1:v]hwupload,format={second}[b];"
            f"[a][b]transition_cuda=mode={mode}:alpha='mod(n,5)/4',"
            f"hwdownload,format={fmt}[out]")


def crop_reference(raw: bytes, fmt: str) -> bytes:
    size, sub_y, _ = FORMATS[fmt]
    result = bytearray()
    offset = 0
    for _ in range(FRAMES):
        for plane, height in ((0, HEIGHT), (1, HEIGHT >> sub_y)):
            y = 6 >> (sub_y if plane else 0)
            out_height = 26 >> (sub_y if plane else 0)
            for row in range(y, y + out_height):
                start = offset + (row * WIDTH + 8) * size
                result += raw[start:start + 54 * size]
            offset += WIDTH * height * size
    return bytes(result)


def pad_reference(raw: bytes, fmt: str) -> bytes:
    size, sub_y, shift = FORMATS[fmt]
    result = bytearray()
    source_offset = 0
    for _ in range(FRAMES):
        for plane, height in ((0, HEIGHT), (1, HEIGHT >> sub_y)):
            output_height = 50 >> (sub_y if plane else 0)
            offset_y = 6 >> (sub_y if plane else 0)
            value = (128 if plane else 16) << (shift + (2 if size == 2 else 0))
            output = bytearray(value.to_bytes(size, "little") * (86 * output_height))
            for y in range(height):
                start = ((y + offset_y) * 86 + 8) * size
                source = source_offset + y * WIDTH * size
                output[start:start + WIDTH * size] = raw[source:source + WIDTH * size]
            result += output
            source_offset += WIDTH * height * size
    return bytes(result)


def run_decoded(ffmpeg: str, clip: Path, storage: tuple[str, ...], operation: str,
                preprocess: tuple[str, str] = ("null", "null"), output_format: str = "nv12") -> bytes:
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
           "-init_hw_device", "cuda=gpu:0", "-filter_hw_device", "gpu",
           "-filter_complex_threads", "1"]
    for index, layout in enumerate(storage):
        cmd += ["-hwaccel", "cuda", "-hwaccel_device", "gpu", "-hwaccel_output_format", layout,
                "-extra_hw_frames", "3", "-threads", "1"]
        if layout == "cuarray":
            cmd += ["-hwaccel_flags", "+unsafe_output"]
        if index:
            cmd += ["-ss", "0.4"]
        cmd += ["-i", str(clip)]
    if len(storage) == 1:
        graph = f"[0:v]{operation},hwdownload,format={output_format}[out]"
    else:
        graph = (f"[0:v]setpts=PTS-STARTPTS,{preprocess[0]}[a];"
                 f"[1:v]setpts=PTS-STARTPTS,{preprocess[1]}[b];"
                 f"[a][b]{operation},crop_cuda=320:180:8:8,hwdownload,format={output_format}[out]")
    cmd += ["-filter_complex", graph, "-map", "[out]", "-frames:v", str(FRAMES),
            "-threads", "1", "-c:v", "rawvideo", "-pix_fmt", output_format, "-f", "rawvideo", "pipe:1"]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if completed.returncode or not completed.stdout:
        raise RuntimeError(completed.stderr.decode(errors="replace"))
    return completed.stdout


def decoded_cases(ffmpeg: str, clip: Path):
    for name, operation in (("decode-crop", "crop_cuda=320:180:8:8"),
                            ("decode-pad", "pad_cuda=iw+32:ih+24:16:12,crop_cuda=320:180:0:0")):
        baseline = run_decoded(ffmpeg, clip, ("cuda",), operation)
        yield name, baseline, run_decoded(ffmpeg, clip, ("cuarray",), operation)
    for mode in ("fade", "wipe_left", "dip"):
        operation = f"transition_cuda=mode={mode}:alpha='mod(n,5)/4'"
        # Each decoder's private stream survives pad/crop in the linear frame
        # device. The second transform queues linear work without a CPU fence,
        # exercising transition readiness and overlay lifetime across streams.
        chains = {
            "decode": ("null", "null"),
            "decode-chain": (
                "pad_cuda=iw+32:ih+24:16:12,crop_cuda=iw-32:ih-24:16:12",
                "crop_cuda=iw-32:ih-24:16:12,pad_cuda=iw+32:ih+24:16:12",
            ),
        }
        for name, preprocess in chains.items():
            baseline = run_decoded(ffmpeg, clip, ("cuda", "cuda"), operation, preprocess)
            for layout in (("cuarray", "cuarray"), ("cuda", "cuarray"), ("cuarray", "cuda")):
                actual = run_decoded(ffmpeg, clip, layout, operation, preprocess)
                yield f"{name}-{mode}-{'-'.join(layout)}", baseline, actual


def compare(results: list, output: Path, fmt: str, name: str, baseline: bytes, actual: bytes):
    mismatches = sum(a != b for a, b in zip(actual, baseline)) + abs(len(actual) - len(baseline))
    result = {"format": fmt, "case": name, "bytes": len(actual), "different_bytes": mismatches,
              "sha256": hashlib.sha256(actual).hexdigest()}
    results.append(result)
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if mismatches:
        output.with_suffix(".actual.raw").write_bytes(actual)
        output.with_suffix(".expected.raw").write_bytes(baseline)
        raise RuntimeError(f"{fmt}/{name}: {mismatches} bytes differ from linear CUDA")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--formats", nargs="+", choices=tuple(FORMATS), default=list(FORMATS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip", type=Path, help="optional HEVC 8-bit 4:2:0 fixture, at least 320x180")
    parser.add_argument("--decode-only", action="store_true", help="run only --clip comparisons")
    args = parser.parse_args()
    if args.decode_only and not args.clip:
        parser.error("--decode-only requires --clip")
    results = []
    with tempfile.TemporaryDirectory(prefix="cuarray-filters-") as directory:
        root = Path(directory)
        for fmt in (() if args.decode_only else args.formats):
            paths = [root / f"{fmt}-{i}.raw" for i in range(2)]
            for i, path in enumerate(paths):
                path.write_bytes(fixture(fmt, i))
            operations = {"crop": "crop_cuda=54:26:8:6", "pad": "pad_cuda=86:50:8:6:black"}
            cases = []
            for name, operation in operations.items():
                print(json.dumps({"stage": "baseline", "format": fmt, "case": name}), flush=True)
                baseline = run(args.ffmpeg, paths[:1], fmt, single_graph("cuda", operation, fmt))
                reference = crop_reference if name == "crop" else pad_reference
                if baseline != reference(paths[0].read_bytes(), fmt):
                    raise RuntimeError(f"{fmt} linear {name} differs from CPU reference")
                cases.append((name, paths[:1], baseline, single_graph("cuarray", operation, fmt)))
            for mode in MODES:
                print(json.dumps({"stage": "baseline", "format": fmt, "case": mode}), flush=True)
                baseline = run(args.ffmpeg, paths, fmt, transition_graph("cuda", "cuda", mode, fmt))
                for first, second in (("cuarray", "cuarray"), ("cuda", "cuarray"), ("cuarray", "cuda")):
                    cases.append((f"{mode}-{first}-{second}", paths, baseline,
                                  transition_graph(first, second, mode, fmt)))
            for name, inputs, baseline, graph in cases:
                actual = run(args.ffmpeg, inputs, fmt, graph)
                compare(results, args.output, fmt, name, baseline, actual)
    if args.clip:
        for name, baseline, actual in decoded_cases(args.ffmpeg, args.clip):
            compare(results, args.output, "nv12", name, baseline, actual)
    print(f"PASS: {len(results)} CUarray filter pixel comparisons", flush=True)


if __name__ == "__main__":
    main()
