"""LumaDiff CUarray/linear parity and independent CPU score reference.

Use the same finite 8-bit HEVC fixture as camera_motion.py. CPU decoding is
only the test oracle; the graphs under test forward GPU frames unchanged.
"""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from camera_motion import run


def reference(args):
    info = json.loads(subprocess.check_output([
        args.ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(args.input)], timeout=30))
    width, height = (info["streams"][0][key] for key in ("width", "height"))
    assert width % 2 == height % 2 == 0, "fixture must have even NV12 dimensions"
    raw = subprocess.check_output([
        args.ffmpeg, "-v", "error", "-threads", "1", "-i", str(args.input),
        "-map", "0:v:0", "-pix_fmt", "nv12", "-f", "rawvideo", "pipe:1"], timeout=30)
    size = width * height * 3 // 2
    assert len(raw) == args.frames * size
    return [np.frombuffer(raw, np.uint8, count=width * height, offset=i * size).astype(np.int16)
            for i in range(args.frames)]


def check_scores(records, frames):
    for index, (_, scores) in enumerate(records):
        for k, score in enumerate(scores):
            has_prev = index > 0 and index + k < len(frames)
            status = "no_prev_frame" if index == 0 else "ok" if has_prev else "no_forward_frame"
            assert score["status"] == status and score["has_prev"] == has_prev, (index, k, score)
            assert score["lookahead"] == k
            diff = frames[index + k] - frames[index - 1] if has_prev else np.array([0])
            expected = [np.abs(diff).mean(), abs(diff.mean() / 255), diff.mean()]
            actual = [score[key] for key in ("mean_abs", "mean_norm", "mean_signed")]
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6,
                                       err_msg=f"frame {index}, lookahead {k}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--lookahead", nargs="+", type=int, default=[0, 2, 4, 16])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--unsupported-input", type=Path)
    parser.add_argument("--linear-only", action="store_true", help="also run against an unmodified module")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reference", type=Path, help="compare linear metadata against a saved build")
    args = parser.parse_args()
    frames = reference(args)
    report = {}
    for lookahead in args.lookahead:
        baseline = run(args, "cuda", None, lookahead=lookahead)
        check_scores(baseline, frames)
        report[str(lookahead)] = baseline
        if not args.linear_only:
            for _ in range(args.rounds):
                for filter_linear in (False, True):
                    actual = run(args, "cuarray", None, lookahead=lookahead, filter_linear=filter_linear)
                    assert actual == baseline, f"CUarray score/PTS mismatch at lookahead {lookahead}"
                    check_scores(actual, frames)
            assert run(args, "mixed", None, lookahead=lookahead) == baseline, "stream/storage switching mismatch"
        storage = "linear" if args.linear_only else "linear/CUarray"
        print(f"PASS lookahead={lookahead}: CPU oracle, {storage} metadata, ring wrap, "
              "EOF tail, original handles and PTS", flush=True)
    if args.unsupported_input and not args.linear_only:
        records = run(args, "cuarray", None, lookahead=2, unsupported=True)
        assert all(s["status"].startswith("unsupported_sw_format_") and not s["has_prev"]
                   for _, scores in records for s in scores)
        print("PASS Main10 CUarray: explicit unsupported status", flush=True)
    if args.reference:
        assert json.loads(json.dumps(report)) == json.loads(args.reference.read_text()), "linear build regression"
        print("PASS exact linear metadata/PTS against reference build", flush=True)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
