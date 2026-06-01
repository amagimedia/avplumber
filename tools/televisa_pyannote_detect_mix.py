#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from televisa_pyannote_enroll import (
    decode_audio,
    extract_embedding,
    l2_normalize,
    load_embedding_model,
    rms_dbfs,
    slice_audio,
    utc_now,
)


DEFAULT_CONTENT_ROOT_ENV = "AVP_TELEVISA_CONTENT_ROOT"
DEFAULT_CONTENT_ROOT = (
    Path(os.environ[DEFAULT_CONTENT_ROOT_ENV])
    if os.environ.get(DEFAULT_CONTENT_ROOT_ENV)
    else None
)
DEFAULT_CENTROIDS = (
    DEFAULT_CONTENT_ROOT / "speaker-enrollment" / "pyannote-non-mix-prefer-loudest" / "speaker-centroids.npz"
    if DEFAULT_CONTENT_ROOT is not None
    else None
)


@dataclass(frozen=True)
class WindowScore:
    start_sec: float
    end_sec: float
    rms_dbfs: float
    predicted_speaker: str
    top_score: float
    second_speaker: str
    second_score: float
    score_margin: float
    accepted: bool
    reject_reason: str = ""

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class Segment:
    speaker: str
    start_sec: float
    end_sec: float
    window_count: int
    mean_top_score: float
    mean_score_margin: float
    clip_path: Path | None = None

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def format_time(seconds: float) -> str:
    return f"{max(0.0, seconds):.6f}"


def slugify(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return cleaned.strip("_") or "speaker"


def load_centroids(path: Path) -> tuple[list[str], np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as data:
        speakers = [str(value) for value in data["speakers"].tolist()]
        centroids = np.asarray(data["centroids"], dtype=np.float32)
        metadata = {
            "model": str(data["model"].tolist()) if "model" in data.files else "",
            "sample_rate": int(data["sample_rate"].tolist()) if "sample_rate" in data.files else 0,
        }
    if not speakers or centroids.size == 0:
        raise RuntimeError(f"no centroids in {path}")
    if centroids.shape[0] != len(speakers):
        raise RuntimeError(f"centroid speaker count mismatch in {path}")
    centroids = np.stack([l2_normalize(row) for row in centroids])
    return speakers, centroids, metadata


def score_embedding(embedding: np.ndarray, speakers: list[str], centroids: np.ndarray) -> tuple[str, float, str, float, float]:
    scores = centroids @ embedding
    order = np.argsort(scores)[::-1]
    top_index = int(order[0])
    second_index = int(order[1]) if len(order) > 1 else -1
    top_score = float(scores[top_index])
    second_score = float(scores[second_index]) if second_index >= 0 else 0.0
    return (
        speakers[top_index],
        top_score,
        speakers[second_index] if second_index >= 0 else "",
        second_score,
        top_score - second_score,
    )


def iter_windows(duration_sec: float, window_sec: float, step_sec: float, min_window_sec: float) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    if duration_sec < min_window_sec:
        return windows
    if duration_sec <= window_sec:
        return [(0.0, duration_sec)]

    start = 0.0
    last_start = duration_sec - window_sec
    while start <= last_start + 1e-6:
        windows.append((start, start + window_sec))
        start += step_sec
    if duration_sec - windows[-1][1] >= min_window_sec:
        windows.append((duration_sec - window_sec, duration_sec))
    return windows


def score_windows(
    audio: np.ndarray,
    sample_rate: int,
    inference,
    speakers: list[str],
    centroids: np.ndarray,
    args: argparse.Namespace,
) -> list[WindowScore]:
    duration_sec = audio.size / sample_rate
    windows = iter_windows(duration_sec, args.window_sec, args.step_sec, args.min_window_sec)
    scored: list[WindowScore] = []
    for index, (start_sec, end_sec) in enumerate(windows, start=1):
        level = rms_dbfs(audio, sample_rate, start_sec, end_sec)
        if level < args.min_rms_db:
            scored.append(
                WindowScore(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    rms_dbfs=level,
                    predicted_speaker="",
                    top_score=0.0,
                    second_speaker="",
                    second_score=0.0,
                    score_margin=0.0,
                    accepted=False,
                    reject_reason="below_min_rms",
                )
            )
            continue

        window_audio = slice_audio(audio, sample_rate, start_sec, end_sec)
        embedding = extract_embedding(inference, window_audio, sample_rate)
        predicted, top_score, second, second_score, margin = score_embedding(embedding, speakers, centroids)
        reject_reason = ""
        if top_score < args.min_score:
            reject_reason = "low_score"
        elif margin < args.min_margin:
            reject_reason = "low_margin"
        scored.append(
            WindowScore(
                start_sec=start_sec,
                end_sec=end_sec,
                rms_dbfs=level,
                predicted_speaker=predicted,
                top_score=top_score,
                second_speaker=second,
                second_score=second_score,
                score_margin=margin,
                accepted=not reject_reason,
                reject_reason=reject_reason,
            )
        )
        if args.progress_every and index % args.progress_every == 0:
            print(f"Scored {index}/{len(windows)} windows", flush=True)
    return scored


def merge_speaker_windows(
    speaker: str,
    windows: list[WindowScore],
    gap_sec: float,
    pad_sec: float,
    min_clip_sec: float,
    max_clip_sec: float,
    input_duration_sec: float,
) -> list[Segment]:
    accepted = [window for window in windows if window.accepted and window.predicted_speaker == speaker]
    if not accepted:
        return []

    segments: list[Segment] = []
    group: list[WindowScore] = []
    for window in sorted(accepted, key=lambda item: (item.start_sec, item.end_sec)):
        if not group or window.start_sec <= group[-1].end_sec + gap_sec:
            group.append(window)
            continue
        segments.extend(build_segments_from_group(speaker, group, pad_sec, min_clip_sec, max_clip_sec, input_duration_sec))
        group = [window]
    segments.extend(build_segments_from_group(speaker, group, pad_sec, min_clip_sec, max_clip_sec, input_duration_sec))
    return segments


def build_segments_from_group(
    speaker: str,
    group: list[WindowScore],
    pad_sec: float,
    min_clip_sec: float,
    max_clip_sec: float,
    input_duration_sec: float,
) -> list[Segment]:
    start_sec = max(0.0, min(window.start_sec for window in group) - pad_sec)
    end_sec = min(input_duration_sec, max(window.end_sec for window in group) + pad_sec)
    if end_sec - start_sec < min_clip_sec:
        return []

    if max_clip_sec <= 0.0 or end_sec - start_sec <= max_clip_sec:
        return [
            Segment(
                speaker=speaker,
                start_sec=start_sec,
                end_sec=end_sec,
                window_count=len(group),
                mean_top_score=float(np.mean([window.top_score for window in group])),
                mean_score_margin=float(np.mean([window.score_margin for window in group])),
            )
        ]

    segments: list[Segment] = []
    current_start = start_sec
    while current_start < end_sec:
        current_end = min(end_sec, current_start + max_clip_sec)
        overlap_windows = [
            window
            for window in group
            if max(0.0, min(current_end, window.end_sec) - max(current_start, window.start_sec)) > 0.0
        ]
        if current_end - current_start >= min_clip_sec and overlap_windows:
            segments.append(
                Segment(
                    speaker=speaker,
                    start_sec=current_start,
                    end_sec=current_end,
                    window_count=len(overlap_windows),
                    mean_top_score=float(np.mean([window.top_score for window in overlap_windows])),
                    mean_score_margin=float(np.mean([window.score_margin for window in overlap_windows])),
                )
            )
        current_start = current_end
    return segments


def run_ffmpeg_copy(input_path: Path, output_path: Path, start_sec: float, end_sec: float, ffmpeg: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_sec = max(0.0, end_sec - start_sec)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        format_time(start_sec),
        "-i",
        str(input_path),
        "-t",
        format_time(duration_sec),
        "-map",
        "0:a:0",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        "-y",
        str(output_path),
    ]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to cut {output_path}: {stderr}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced an empty clip: {output_path}")


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(f"output directory is not empty: {path}; use --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_window_scores(path: Path, scores: list[WindowScore]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "start_sec",
                "end_sec",
                "duration_sec",
                "rms_dbfs",
                "accepted",
                "reject_reason",
                "predicted_speaker",
                "top_score",
                "second_speaker",
                "second_score",
                "score_margin",
            ],
        )
        writer.writeheader()
        for score in scores:
            writer.writerow(
                {
                    "start_sec": f"{score.start_sec:.6f}",
                    "end_sec": f"{score.end_sec:.6f}",
                    "duration_sec": f"{score.duration_sec:.6f}",
                    "rms_dbfs": f"{score.rms_dbfs:.3f}",
                    "accepted": int(score.accepted),
                    "reject_reason": score.reject_reason,
                    "predicted_speaker": score.predicted_speaker,
                    "top_score": f"{score.top_score:.6f}",
                    "second_speaker": score.second_speaker,
                    "second_score": f"{score.second_score:.6f}",
                    "score_margin": f"{score.score_margin:.6f}",
                }
            )


def write_segments(path: Path, segments: list[Segment]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "speaker",
                "start_sec",
                "end_sec",
                "duration_sec",
                "window_count",
                "mean_top_score",
                "mean_score_margin",
                "clip_path",
            ],
        )
        writer.writeheader()
        for segment in sorted(segments, key=lambda item: (item.speaker, item.start_sec, item.end_sec)):
            writer.writerow(
                {
                    "speaker": segment.speaker,
                    "start_sec": f"{segment.start_sec:.6f}",
                    "end_sec": f"{segment.end_sec:.6f}",
                    "duration_sec": f"{segment.duration_sec:.6f}",
                    "window_count": segment.window_count,
                    "mean_top_score": f"{segment.mean_top_score:.6f}",
                    "mean_score_margin": f"{segment.mean_score_margin:.6f}",
                    "clip_path": str(segment.clip_path or ""),
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect enrolled Televisa speakers in Rene's mixed audio track.")
    parser.add_argument(
        "--content-root",
        type=Path,
        default=DEFAULT_CONTENT_ROOT,
        required=DEFAULT_CONTENT_ROOT is None,
        help=f"Televisa content directory; may also be set with {DEFAULT_CONTENT_ROOT_ENV}.",
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--centroids", type=Path, default=DEFAULT_CENTROIDS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model", default="pyannote/embedding")
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="Optional Hugging Face token. Defaults to HF_TOKEN/HUGGINGFACE_HUB_TOKEN or cached login.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--min-window-sec", type=float, default=1.5)
    parser.add_argument("--min-rms-db", type=float, default=-55.0)
    parser.add_argument("--min-score", type=float, default=0.48)
    parser.add_argument("--min-margin", type=float, default=0.08)
    parser.add_argument("--merge-gap-sec", type=float, default=1.25)
    parser.add_argument("--pad-sec", type=float, default=0.25)
    parser.add_argument("--min-clip-sec", type=float, default=2.0)
    parser.add_argument("--max-clip-sec", type=float, default=18.0)
    parser.add_argument("--max-clips-per-speaker", type=int, default=0)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.content_root = args.content_root.resolve()
    if args.input is None:
        args.input = args.content_root / "televisa-Rene_norm.m4a"
    if args.centroids is None:
        args.centroids = args.content_root / "speaker-enrollment" / "pyannote-non-mix-prefer-loudest" / "speaker-centroids.npz"
    if args.output_dir is None:
        args.output_dir = args.content_root / "speaker-detections" / "rene-mix-pyannote"
    args.input = args.input.resolve()
    args.centroids = args.centroids.resolve()
    args.output_dir = args.output_dir.resolve()

    if not args.input.exists():
        print(f"Missing input mix: {args.input}", file=sys.stderr)
        return 2
    if not args.centroids.exists():
        print(f"Missing centroids: {args.centroids}", file=sys.stderr)
        return 2

    prepare_output_dir(args.output_dir, args.overwrite)

    speakers, centroids, centroid_metadata = load_centroids(args.centroids)
    print(f"Loaded centroids for: {', '.join(speakers)}", flush=True)
    print(f"Loading pyannote model: {args.model}", flush=True)
    inference, device = load_embedding_model(args.model, args.hf_token, args.device)
    print(f"Embedding device: {device}", flush=True)

    print(f"Decoding mix: {args.input}", flush=True)
    audio = decode_audio(args.input, args.sample_rate, args.ffmpeg)
    duration_sec = audio.size / args.sample_rate
    print(f"Scoring {duration_sec:.3f}s with {args.window_sec:.3f}s/{args.step_sec:.3f}s windows", flush=True)
    scores = score_windows(audio, args.sample_rate, inference, speakers, centroids, args)

    segments: list[Segment] = []
    for speaker in speakers:
        speaker_segments = merge_speaker_windows(
            speaker=speaker,
            windows=scores,
            gap_sec=args.merge_gap_sec,
            pad_sec=args.pad_sec,
            min_clip_sec=args.min_clip_sec,
            max_clip_sec=args.max_clip_sec,
            input_duration_sec=duration_sec,
        )
        if args.max_clips_per_speaker > 0:
            speaker_segments = sorted(
                speaker_segments,
                key=lambda item: (item.window_count, item.mean_score_margin, item.duration_sec),
                reverse=True,
            )[: args.max_clips_per_speaker]
            speaker_segments = sorted(speaker_segments, key=lambda item: item.start_sec)
        segments.extend(speaker_segments)
        print(f"{speaker}: {len(speaker_segments)} clips", flush=True)

    if not args.no_export:
        for speaker in speakers:
            speaker_segments = [segment for segment in segments if segment.speaker == speaker]
            for index, segment in enumerate(speaker_segments, start=1):
                name = (
                    f"{index:03d}_{segment.start_sec:010.3f}_{segment.end_sec:010.3f}"
                    f"_score-{segment.mean_top_score:.3f}_margin-{segment.mean_score_margin:.3f}.m4a"
                )
                output_path = args.output_dir / slugify(speaker) / name
                run_ffmpeg_copy(args.input, output_path, segment.start_sec, segment.end_sec, args.ffmpeg)
                segment.clip_path = output_path

    write_window_scores(args.output_dir / "window-scores.csv", scores)
    write_segments(args.output_dir / "detections.csv", segments)
    summary = {
        "created_at": utc_now(),
        "input": str(args.input),
        "centroids": str(args.centroids),
        "centroid_metadata": centroid_metadata,
        "model": args.model,
        "device": device,
        "sample_rate": args.sample_rate,
        "duration_sec": round(duration_sec, 6),
        "window_sec": args.window_sec,
        "step_sec": args.step_sec,
        "min_rms_db": args.min_rms_db,
        "min_score": args.min_score,
        "min_margin": args.min_margin,
        "merge_gap_sec": args.merge_gap_sec,
        "pad_sec": args.pad_sec,
        "min_clip_sec": args.min_clip_sec,
        "max_clip_sec": args.max_clip_sec,
        "speakers": {
            speaker: {
                "accepted_windows": sum(1 for score in scores if score.accepted and score.predicted_speaker == speaker),
                "clips": sum(1 for segment in segments if segment.speaker == speaker),
                "clip_duration_sec": round(
                    sum(segment.duration_sec for segment in segments if segment.speaker == speaker),
                    6,
                ),
            }
            for speaker in speakers
        },
        "outputs": {
            "detections_csv": str(args.output_dir / "detections.csv"),
            "window_scores_csv": str(args.output_dir / "window-scores.csv"),
            "summary_json": str(args.output_dir / "summary.json"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote outputs to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
