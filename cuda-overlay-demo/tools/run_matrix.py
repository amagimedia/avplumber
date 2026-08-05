#!/usr/bin/env python3
"""Generate fixtures, run the CUDA graph matrix, compare, and report."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from cpu_reference import compare_frames, compose
from generate_assets import generate_all
from make_contact_sheets import make_result_sheets, make_source_sheet, raw_to_png
from overlay_common import HEIGHT, MAX_OVERLAYS, MODES, WIDTH, read_frame, write_frame


DEMO_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = DEMO_DIR.parent
GRAPH = DEMO_DIR / "graph" / "overlay_case.py"


def _command_text(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"
    output = (result.stdout or result.stderr).strip()
    return output if output else f"exit {result.returncode}"


def _patch_identity() -> dict[str, str]:
    identities: dict[str, str] = {}
    for path in sorted((REPO_DIR / "deps" / "ffmpeg-patches").glob("*.patch")):
        identities[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return identities


def _parse_counts(value: str) -> list[int]:
    counts: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            counts.update(range(start, end + 1))
        else:
            counts.add(int(part))
    if not counts or min(counts) < 1 or max(counts) > MAX_OVERLAYS:
        raise argparse.ArgumentTypeError(f"counts must be within 1..{MAX_OVERLAYS}")
    return sorted(counts)


def _positive_dimension(value: str) -> int:
    dimension = int(value)
    if dimension <= 0:
        raise argparse.ArgumentTypeError("dimensions must be positive")
    return dimension


def _write_report(path: Path, report: dict[str, object]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("/artifacts"))
    parser.add_argument("--ffmpeg", default="/usr/local/bin/ffmpeg")
    parser.add_argument("--counts", type=_parse_counts, default=_parse_counts("1-15"))
    parser.add_argument("--case-timeout", type=float, default=30.0)
    parser.add_argument("--width", type=_positive_dimension, default=WIDTH)
    parser.add_argument("--height", type=_positive_dimension, default=HEIGHT)
    args = parser.parse_args()

    started_at = dt.datetime.now(dt.timezone.utc)
    run_id = started_at.strftime("run-%Y%m%dT%H%M%SZ")
    run_dir = args.artifacts / run_id
    logs_dir = run_dir / "logs"
    gpu_dir = run_dir / "gpu-raw"
    reference_dir = run_dir / "reference-raw"
    result_png_dir = run_dir / "result-png"
    sheet_dir = run_dir / "contact-sheets"
    for path in (logs_dir, gpu_dir, reference_dir, result_png_dir, sheet_dir):
        path.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "status": "running",
        "ffmpeg_tag": "n7.1.5",
        "cuda_toolkit_minimum": "11.7",
        "dimensions": {"width": args.width, "height": args.height},
        "requested_overlay_counts": args.counts,
        "modes": list(MODES),
        "environment": {
            "ffmpeg": _command_text([args.ffmpeg, "-version"]),
            "filters": _command_text([args.ffmpeg, "-hide_banner", "-filters"]),
            "nvcc": _command_text(["nvcc", "--version"]),
            "gpu": _command_text(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,compute_cap",
                    "--format=csv,noheader",
                ]
            ),
            "repository_commit": os.environ.get("AVPLUMBER_REVISION", "workspace"),
            "patches_sha256": _patch_identity(),
        },
        "cases": [],
    }
    report_path = run_dir / "report.json"
    _write_report(report_path, report)

    filter_listing = str(report["environment"]["filters"])  # type: ignore[index]
    if "overlay_many_cuda" not in filter_listing:
        report["status"] = "prerequisite-failed"
        report["error"] = "patched FFmpeg does not expose overlay_many_cuda"
        _write_report(report_path, report)
        return 2
    gpu_listing = str(report["environment"]["gpu"])  # type: ignore[index]
    if gpu_listing.startswith("unavailable:") or gpu_listing.startswith("exit "):
        report["status"] = "prerequisite-failed"
        report["error"] = "no NVIDIA GPU is visible in the container"
        _write_report(report_path, report)
        return 2

    generate_all(run_dir, args.ffmpeg, args.width, args.height)
    raw_dir = run_dir / "assets" / "raw"
    stream_dir = run_dir / "assets" / "input-streams"
    failures = 0
    for mode_name, mode in MODES.items():
        for overlay_count in args.counts:
            case_name = f"{mode_name}-overlays_{overlay_count:02d}"
            gpu_path = gpu_dir / f"{case_name}.yuv"
            reference_path = reference_dir / f"{case_name}.yuv"
            png_path = result_png_dir / f"{case_name}.png"
            log_path = logs_dir / f"{case_name}.log"
            command = [
                sys.executable,
                str(GRAPH),
                "--mode",
                mode_name,
                "--overlays",
                str(overlay_count),
                "--raw-dir",
                str(stream_dir),
                "--output",
                str(gpu_path),
                "--timeout",
                str(args.case_timeout),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
            ]
            case_started = time.monotonic()
            case: dict[str, object] = {
                "name": case_name,
                "mode": mode_name,
                "overlays": overlay_count,
                "total_inputs": overlay_count + 1,
                "status": "running",
                "log": str(log_path.relative_to(run_dir)),
            }
            report["cases"].append(case)  # type: ignore[union-attr]
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    process = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                        env=os.environ.copy(),
                        timeout=args.case_timeout + 10.0,
                    )
                if process.returncode != 0:
                    raise RuntimeError(f"PyPlumber graph exited {process.returncode}")

                expected_planes = compose(
                    mode_name,
                    overlay_count,
                    raw_dir,
                    args.width,
                    args.height,
                )
                write_frame(reference_path, expected_planes)
                actual_planes = read_frame(
                    gpu_path,
                    mode.main_format,
                    args.width,
                    args.height,
                )
                comparisons = compare_frames(actual_planes, expected_planes)
                case["planes"] = dict(zip(("Y", "U", "V"), comparisons))
                mismatch_count = sum(int(plane["mismatch_count"]) for plane in comparisons)
                if mismatch_count:
                    raise AssertionError(f"{mismatch_count} raw YUV samples differ")

                raw_to_png(
                    args.ffmpeg,
                    gpu_path,
                    mode.main_format,
                    png_path,
                    args.width,
                    args.height,
                )
                case["result_png"] = str(png_path.relative_to(run_dir))
                case["status"] = "passed"
            except Exception as error:  # Keep running to report the whole matrix.
                failures += 1
                case["status"] = "failed"
                case["error"] = f"{type(error).__name__}: {error}"
                if gpu_path.is_file():
                    try:
                        raw_to_png(
                            args.ffmpeg,
                            gpu_path,
                            mode.main_format,
                            png_path,
                            args.width,
                            args.height,
                        )
                        case["result_png"] = str(png_path.relative_to(run_dir))
                    except Exception:
                        pass
            finally:
                case["duration_seconds"] = round(time.monotonic() - case_started, 3)
                _write_report(report_path, report)

    make_source_sheet(run_dir / "assets" / "png", sheet_dir)
    make_result_sheets(result_png_dir, sheet_dir)
    report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    report["failure_count"] = failures
    report["status"] = "passed" if failures == 0 else "failed"
    report["contact_sheets"] = [
        str(path.relative_to(run_dir)) for path in sorted(sheet_dir.glob("*.png"))
    ]
    _write_report(report_path, report)
    print(f"CUDA overlay demo {report['status']}: {report_path}", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
