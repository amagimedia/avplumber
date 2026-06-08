#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyplumber.sports_reframe import (  # pyright: ignore[reportMissingImports]
    optional_fps_value,
    parse_size,
    ratio_from_fps,
    run_tracknet_ball_to_raw_json,
)


def default_raw_path(input_path: Path) -> Path:
    return Path("artifacts") / f"{input_path.stem}_ball_tracknet_raw.json"


def default_interp_path(raw_path: Path) -> Path:
    name = raw_path.name
    if name.endswith("_ball_tracknet_raw.json"):
        return raw_path.with_name(name.replace("_ball_tracknet_raw.json", "_ball_interpolated.json"))
    return raw_path.with_name(f"{raw_path.stem}_interpolated.json")


def load_scene_boundaries(path: Path | None) -> list[int]:
    if path is None:
        return []
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return [int(x) for x in data.get("scene_change_boundaries", [])]


def run_existing_ball_interpolation(
    service_repo: Path,
    input_path: Path,
    output_dir: Path,
    raw_json: Path,
    interp_json: Path,
    scene_json: Path | None,
    ball_box_half_px: float,
) -> None:
    for path in (service_repo, service_repo / "src"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

    from pipeline.ball_interpolation import run_ball_interpolation  # pyright: ignore[reportMissingImports]
    from pipeline.config import PipelineConfig  # pyright: ignore[reportMissingImports]

    cfg = PipelineConfig(
        video_path=input_path,
        out_dir=output_dir,
        ball_box_half_px=ball_box_half_px,
    )
    scene_boundaries = load_scene_boundaries(scene_json)
    run_ball_interpolation(cfg, raw_json, interp_json, scene_boundaries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1 sports reframing example: avplumber TrackNet metadata into existing ball raw JSON."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input processing video.")
    parser.add_argument(
        "--engine",
        type=Path,
        default=os.environ.get("AVP_TRACKNET_ENGINE"),
        required=os.environ.get("AVP_TRACKNET_ENGINE") is None,
        help="TrackNet TensorRT plan path. Can also be set with AVP_TRACKNET_ENGINE.",
    )
    parser.add_argument("--ball-raw-json", type=Path, default=None)
    parser.add_argument("--metadata-jsonl", type=Path, default=None)
    parser.add_argument("--metadata-key", default="tracknet_ball")
    parser.add_argument("--target-label", default="basketball")
    parser.add_argument("--conf-thresh", type=float, default=0.04)
    parser.add_argument("--visible-thresh", type=float, default=0.5)
    parser.add_argument("--output-mode", default="detection", choices=["detection", "srs_ball"])
    parser.add_argument("--triplet-alignment", default="latest", choices=["latest", "center"])
    parser.add_argument("--preprocess-mode", default=None, choices=["resize", "srs_affine"])
    parser.add_argument(
        "--tracknet-auto-sample-min-fps",
        default="50/1",
        help="Enable TrackNet frame sampling at or above this input FPS; use off/none to disable.",
    )
    parser.add_argument(
        "--tracknet-auto-sample-every-n",
        type=int,
        default=2,
        help="When auto sampling is active, run TrackNet on every Nth source frame.",
    )
    parser.add_argument("--fps", default="", help="Optional forced processing FPS; empty preserves source cadence.")
    parser.add_argument("--tracknet-scale", type=parse_size, default=None, metavar="WIDTHxHEIGHT")
    parser.add_argument("--contract-width", type=int, default=0)
    parser.add_argument("--contract-height", type=int, default=0)
    parser.add_argument("--include-timing", action="store_true")
    parser.add_argument("--use-cuda-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-log-metadata", action="store_true")
    parser.add_argument("--debug-log-every-n", type=int, default=0)
    parser.add_argument("--queue-capacity", type=int, default=12)
    parser.add_argument("--initial-timeout", type=int, default=20)
    parser.add_argument("--input-timeout", type=int, default=10)
    parser.add_argument("--cuda-device", default=None, help="Optional CUDA device string for av_hwdevice_ctx_create.")
    parser.add_argument("--stop-ts", default="", help="Optional input_rec stop timestamp, for bounded VOD smoke runs.")
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--remote-control-port", type=int, default=0)
    parser.add_argument("--webui-api", default="")
    parser.add_argument("--instance-name", default="sports-reframe-stage1")
    parser.add_argument("--logfile", default="")
    parser.add_argument("--run-ball-interpolation", action="store_true")
    parser.add_argument("--service-repo", type=Path, default=None)
    parser.add_argument("--scene-boundaries-json", type=Path, default=None)
    parser.add_argument("--ball-interpolated-json", type=Path, default=None)
    parser.add_argument("--ball-box-half-px", type=float, default=18.0)
    args = parser.parse_args()

    args.fps = ratio_from_fps(args.fps)
    if args.contract_width < 0 or args.contract_height < 0:
        parser.error("--contract-width and --contract-height must be non-negative")
    if bool(args.contract_width) != bool(args.contract_height):
        parser.error("--contract-width and --contract-height must be provided together")
    args.tracknet_auto_sample_min_fps = optional_fps_value(args.tracknet_auto_sample_min_fps)
    if args.tracknet_auto_sample_every_n < 1:
        parser.error("--tracknet-auto-sample-every-n must be >= 1")
    if (
        args.tracknet_auto_sample_min_fps is not None
        and args.tracknet_auto_sample_every_n > 1
        and args.triplet_alignment != "latest"
    ):
        parser.error("--tracknet-auto-sample-every-n > 1 requires --triplet-alignment latest")
    if args.run_ball_interpolation and args.service_repo is None:
        parser.error("--run-ball-interpolation requires --service-repo")
    return args


def main() -> None:
    args = parse_args()
    raw_json = args.ball_raw_json or default_raw_path(args.input)
    interp_json = args.ball_interpolated_json or default_interp_path(raw_json)

    print(f"Input: {args.input}", flush=True)
    print(f"TrackNet engine: {args.engine}", flush=True)
    print(f"Raw ball JSON: {raw_json}", flush=True)
    if args.metadata_jsonl:
        print(f"Metadata JSONL: {args.metadata_jsonl}", flush=True)

    run_tracknet_ball_to_raw_json(
        input_path=args.input,
        engine_path=args.engine,
        output_path=raw_json,
        metadata_jsonl=args.metadata_jsonl,
        metadata_key=args.metadata_key,
        target_label=args.target_label,
        conf_thresh=args.conf_thresh,
        visible_thresh=args.visible_thresh,
        output_mode=args.output_mode,
        triplet_alignment=args.triplet_alignment,
        preprocess_mode=args.preprocess_mode,
        auto_sample_min_fps=args.tracknet_auto_sample_min_fps,
        auto_sample_every_n=args.tracknet_auto_sample_every_n,
        fps=args.fps,
        tracknet_scale=args.tracknet_scale,
        contract_width=args.contract_width,
        contract_height=args.contract_height,
        include_timing=args.include_timing,
        use_cuda_graph=args.use_cuda_graph,
        debug_log_metadata=args.debug_log_metadata,
        debug_log_every_n=args.debug_log_every_n,
        queue_capacity=args.queue_capacity,
        initial_timeout=args.initial_timeout,
        input_timeout=args.input_timeout,
        cuda_device=args.cuda_device,
        stop_ts=args.stop_ts,
        timeout_s=args.timeout_s,
        remote_control_port=args.remote_control_port,
        webui_api=args.webui_api,
        instance_name=args.instance_name,
        logfile=args.logfile,
    )

    if args.run_ball_interpolation:
        assert args.service_repo is not None
        run_existing_ball_interpolation(
            args.service_repo,
            args.input,
            raw_json.parent,
            raw_json,
            interp_json,
            args.scene_boundaries_json,
            args.ball_box_half_px,
        )


if __name__ == "__main__":
    main()
