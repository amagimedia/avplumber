#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_CONTENT_ROOT_ENV = "AVP_TELEVISA_CONTENT_ROOT"
DEFAULT_CONTENT_ROOT = (
    Path(os.environ[DEFAULT_CONTENT_ROOT_ENV])
    if os.environ.get(DEFAULT_CONTENT_ROOT_ENV)
    else None
)
DEFAULT_SPEAKERS = ["denise", "Genaro", "Leo", "Ray", "Sergio"]
DEFAULT_EXCLUDED = ["Rene"]


@dataclass(frozen=True)
class SpeechBlock:
    speaker: str
    block_id: str
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass(frozen=True)
class Window:
    speaker: str
    block_id: str
    start_sec: float
    end_sec: float
    own_dbfs: float
    loudest_speaker: str
    loudest_dbfs: float
    loudest_margin_db: float
    status: str
    reject_reason: str = ""

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class EmbeddedWindow:
    window: Window
    embedding: np.ndarray
    scores: dict[str, float] | None = None
    predicted_speaker: str = ""
    top_score: float = 0.0
    second_speaker: str = ""
    second_score: float = 0.0
    score_margin: float = 0.0


@dataclass(frozen=True)
class SpeakerSelection:
    windows: list[Window]
    selected: list[Window]
    mode: str


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def speaker_audio_path(content_root: Path, speaker: str) -> Path:
    return content_root / f"televisa-{speaker}_norm.m4a"


def speaker_blocks_path(content_root: Path, speaker: str) -> Path:
    return content_root / speaker / "blocks.csv"


def read_blocks(path: Path, speaker: str) -> list[SpeechBlock]:
    blocks: list[SpeechBlock] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                start_sec = float(row["start_sec"])
                end_sec = float(row["end_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            if end_sec <= start_sec:
                continue
            block_id = str(row.get("block") or len(blocks) + 1)
            blocks.append(SpeechBlock(speaker, block_id, start_sec, end_sec))
    return blocks


def decode_audio(path: Path, sample_rate: int, ffmpeg: str) -> np.ndarray:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode {path}: {stderr}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def rms_dbfs(audio: np.ndarray, sample_rate: int, start_sec: float, end_sec: float) -> float:
    start = max(0, int(round(start_sec * sample_rate)))
    end = min(audio.size, int(round(end_sec * sample_rate)))
    if end <= start:
        return -120.0
    segment = audio[start:end]
    if segment.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(segment.astype(np.float32, copy=False) ** 2)))
    if rms <= 0.0:
        return -120.0
    return max(-120.0, 20.0 * math.log10(rms + 1e-12))


def slice_audio(audio: np.ndarray, sample_rate: int, start_sec: float, end_sec: float) -> np.ndarray:
    start = max(0, int(round(start_sec * sample_rate)))
    end = min(audio.size, int(round(end_sec * sample_rate)))
    if end <= start:
        return np.empty(0, dtype=np.float32)
    return np.ascontiguousarray(audio[start:end], dtype=np.float32)


def generate_windows(
    blocks: list[SpeechBlock],
    window_sec: float,
    step_sec: float,
    min_window_sec: float,
) -> list[tuple[SpeechBlock, float, float]]:
    windows: list[tuple[SpeechBlock, float, float]] = []
    for block in blocks:
        if block.duration_sec < min_window_sec:
            continue
        if block.duration_sec <= window_sec:
            windows.append((block, block.start_sec, block.end_sec))
            continue

        start = block.start_sec
        last_start = block.end_sec - window_sec
        while start <= last_start + 1e-6:
            windows.append((block, start, start + window_sec))
            start += step_sec
        if windows and windows[-1][0] == block and block.end_sec - windows[-1][2] >= min_window_sec:
            windows.append((block, block.end_sec - window_sec, block.end_sec))
    return windows


def evaluate_window(
    speaker: str,
    block_id: str,
    start_sec: float,
    end_sec: float,
    tracks: dict[str, np.ndarray],
    sample_rate: int,
    min_rms_db: float,
    min_loudest_margin_db: float,
    disable_loudest_filter: bool,
) -> Window:
    levels = {
        name: rms_dbfs(track, sample_rate, start_sec, end_sec)
        for name, track in tracks.items()
    }
    own_dbfs = levels.get(speaker, -120.0)
    loudest_speaker, loudest_dbfs = max(levels.items(), key=lambda item: item[1])
    other_levels = [value for name, value in levels.items() if name != speaker]
    nearest_other = max(other_levels) if other_levels else -120.0
    margin = own_dbfs - nearest_other

    reject_reason = ""
    if own_dbfs < min_rms_db:
        reject_reason = "below_min_rms"
    elif not disable_loudest_filter and loudest_speaker != speaker:
        reject_reason = "not_loudest"
    elif not disable_loudest_filter and margin < min_loudest_margin_db:
        reject_reason = "low_loudest_margin"

    return Window(
        speaker=speaker,
        block_id=block_id,
        start_sec=start_sec,
        end_sec=end_sec,
        own_dbfs=own_dbfs,
        loudest_speaker=loudest_speaker,
        loudest_dbfs=loudest_dbfs,
        loudest_margin_db=margin,
        status="rejected" if reject_reason else "accepted",
        reject_reason=reject_reason,
    )


def evenly_limit(windows: list[Window], max_count: int) -> list[Window]:
    if max_count <= 0 or len(windows) <= max_count:
        return windows
    if max_count == 1:
        return [windows[len(windows) // 2]]
    indexes = np.linspace(0, len(windows) - 1, max_count)
    selected = []
    seen: set[int] = set()
    for value in indexes:
        index = int(round(float(value)))
        if index in seen:
            continue
        seen.add(index)
        selected.append(windows[index])
    return selected


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise RuntimeError("empty speaker embedding")
    return vector / norm


def load_embedding_model(model_name: str, token: str | None, device: str):
    warnings.filterwarnings(
        "ignore",
        message=r".*torchcodec is not installed correctly.*",
        category=UserWarning,
        module=r"pyannote\.audio\.core\.io",
    )
    from pyannote.audio import Inference, Model
    import torch

    try:
        model = Model.from_pretrained(model_name, token=token) if token else Model.from_pretrained(model_name)
    except TypeError:
        model = (
            Model.from_pretrained(model_name, use_auth_token=token)
            if token
            else Model.from_pretrained(model_name)
        )
    inference = Inference(model, window="whole")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    inference.to(torch.device(device))
    return inference, device


def extract_embedding(inference, audio: np.ndarray, sample_rate: int) -> np.ndarray:
    import torch

    waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
    embedding = inference({"waveform": waveform, "sample_rate": sample_rate})
    if hasattr(embedding, "data"):
        embedding = embedding.data
    return l2_normalize(np.asarray(embedding, dtype=np.float32))


def embed_windows(
    windows: list[Window],
    tracks: dict[str, np.ndarray],
    sample_rate: int,
    inference,
) -> list[EmbeddedWindow]:
    embedded: list[EmbeddedWindow] = []
    for window in windows:
        audio = slice_audio(tracks[window.speaker], sample_rate, window.start_sec, window.end_sec)
        if audio.size == 0:
            continue
        embedded.append(EmbeddedWindow(window=window, embedding=extract_embedding(inference, audio, sample_rate)))
    return embedded


def evaluate_speaker_windows(
    speaker: str,
    candidate_tuples: list[tuple[SpeechBlock, float, float]],
    tracks: dict[str, np.ndarray],
    sample_rate: int,
    min_rms_db: float,
    min_loudest_margin_db: float,
    disable_loudest_filter: bool,
) -> list[Window]:
    return [
        evaluate_window(
            speaker=block.speaker,
            block_id=block.block_id,
            start_sec=start_sec,
            end_sec=end_sec,
            tracks=tracks,
            sample_rate=sample_rate,
            min_rms_db=min_rms_db,
            min_loudest_margin_db=min_loudest_margin_db,
            disable_loudest_filter=disable_loudest_filter,
        )
        for block, start_sec, end_sec in candidate_tuples
    ]


def choose_speaker_windows(
    speaker: str,
    candidate_tuples: list[tuple[SpeechBlock, float, float]],
    tracks: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> SpeakerSelection:
    if args.selection_mode == "disable-loudest":
        windows = evaluate_speaker_windows(
            speaker=speaker,
            candidate_tuples=candidate_tuples,
            tracks=tracks,
            sample_rate=args.sample_rate,
            min_rms_db=args.min_rms_db,
            min_loudest_margin_db=args.min_loudest_margin_db,
            disable_loudest_filter=True,
        )
        return SpeakerSelection(
            windows=windows,
            selected=evenly_limit([window for window in windows if window.status == "accepted"], args.max_windows_per_speaker),
            mode="disable-loudest",
        )

    strict_windows = evaluate_speaker_windows(
        speaker=speaker,
        candidate_tuples=candidate_tuples,
        tracks=tracks,
        sample_rate=args.sample_rate,
        min_rms_db=args.min_rms_db,
        min_loudest_margin_db=args.min_loudest_margin_db,
        disable_loudest_filter=False,
    )
    strict_accepted = [window for window in strict_windows if window.status == "accepted"]
    if args.selection_mode == "strict-loudest" or len(strict_accepted) >= args.min_accepted_per_speaker:
        return SpeakerSelection(
            windows=strict_windows,
            selected=evenly_limit(strict_accepted, args.max_windows_per_speaker),
            mode="strict-loudest",
        )

    fallback_windows = evaluate_speaker_windows(
        speaker=speaker,
        candidate_tuples=candidate_tuples,
        tracks=tracks,
        sample_rate=args.sample_rate,
        min_rms_db=args.min_rms_db,
        min_loudest_margin_db=args.min_loudest_margin_db,
        disable_loudest_filter=True,
    )
    return SpeakerSelection(
        windows=fallback_windows,
        selected=evenly_limit([window for window in fallback_windows if window.status == "accepted"], args.max_windows_per_speaker),
        mode="fallback-disable-loudest",
    )


def build_centroids(embedded: list[EmbeddedWindow]) -> dict[str, np.ndarray]:
    by_speaker: dict[str, list[np.ndarray]] = {}
    for item in embedded:
        by_speaker.setdefault(item.window.speaker, []).append(item.embedding)
    return {
        speaker: l2_normalize(np.mean(np.stack(vectors), axis=0))
        for speaker, vectors in by_speaker.items()
        if vectors
    }


def score_embedded(embedded: list[EmbeddedWindow], centroids: dict[str, np.ndarray]) -> None:
    speaker_names = list(centroids)
    matrix = np.stack([centroids[name] for name in speaker_names])
    for item in embedded:
        values = matrix @ item.embedding
        scores = {speaker: float(score) for speaker, score in zip(speaker_names, values)}
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        top_name, top_score = ranked[0]
        second_name, second_score = ranked[1] if len(ranked) > 1 else ("", 0.0)
        item.scores = scores
        item.predicted_speaker = top_name
        item.top_score = top_score
        item.second_speaker = second_name
        item.second_score = second_score
        item.score_margin = top_score - second_score


def write_report(path: Path, evaluated: list[Window], embedded: list[EmbeddedWindow], speakers: list[str]) -> None:
    by_key = {
        (item.window.speaker, item.window.block_id, item.window.start_sec, item.window.end_sec): item
        for item in embedded
    }
    fieldnames = [
        "speaker",
        "block",
        "start_sec",
        "end_sec",
        "duration_sec",
        "status",
        "reject_reason",
        "own_dbfs",
        "loudest_speaker",
        "loudest_dbfs",
        "loudest_margin_db",
        "predicted_speaker",
        "top_score",
        "second_speaker",
        "second_score",
        "score_margin",
    ] + [f"score_{speaker}" for speaker in speakers]

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for window in evaluated:
            item = by_key.get((window.speaker, window.block_id, window.start_sec, window.end_sec))
            row = {
                "speaker": window.speaker,
                "block": window.block_id,
                "start_sec": f"{window.start_sec:.6f}",
                "end_sec": f"{window.end_sec:.6f}",
                "duration_sec": f"{window.duration_sec:.6f}",
                "status": window.status,
                "reject_reason": window.reject_reason,
                "own_dbfs": f"{window.own_dbfs:.3f}",
                "loudest_speaker": window.loudest_speaker,
                "loudest_dbfs": f"{window.loudest_dbfs:.3f}",
                "loudest_margin_db": f"{window.loudest_margin_db:.3f}",
                "predicted_speaker": "",
                "top_score": "",
                "second_speaker": "",
                "second_score": "",
                "score_margin": "",
            }
            if item is not None:
                row.update({
                    "predicted_speaker": item.predicted_speaker,
                    "top_score": f"{item.top_score:.6f}",
                    "second_speaker": item.second_speaker,
                    "second_score": f"{item.second_score:.6f}",
                    "score_margin": f"{item.score_margin:.6f}",
                })
                for speaker in speakers:
                    row[f"score_{speaker}"] = f"{(item.scores or {}).get(speaker, 0.0):.6f}"
            writer.writerow(row)


def write_outputs(
    out_dir: Path,
    args: argparse.Namespace,
    speakers: list[str],
    evaluated: list[Window],
    embedded: list[EmbeddedWindow],
    centroids: dict[str, np.ndarray],
    device: str,
    selection_modes: dict[str, str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    centroids_npz = out_dir / "speaker-centroids.npz"
    centroids_json = out_dir / "speaker-centroids.json"
    report_csv = out_dir / "enrollment-report.csv"
    summary_json = out_dir / "enrollment-summary.json"

    ordered_speakers = [speaker for speaker in speakers if speaker in centroids]
    centroid_matrix = np.stack([centroids[speaker] for speaker in ordered_speakers])
    np.savez(
        centroids_npz,
        speakers=np.asarray(ordered_speakers),
        centroids=centroid_matrix,
        model=np.asarray(args.model),
        sample_rate=np.asarray(args.sample_rate),
    )

    centroid_payload = {
        "created_at": utc_now(),
        "model": args.model,
        "device": device,
        "sample_rate": args.sample_rate,
        "content_root": str(args.content_root),
        "excluded_speakers": args.exclude,
        "speakers": {
            speaker: {
                "samples": sum(1 for item in embedded if item.window.speaker == speaker),
                "centroid": centroids[speaker].astype(float).tolist(),
            }
            for speaker in ordered_speakers
        },
    }
    centroids_json.write_text(json.dumps(centroid_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(report_csv, evaluated, embedded, ordered_speakers)

    per_speaker = {}
    for speaker in speakers:
        speaker_windows = [window for window in evaluated if window.speaker == speaker]
        speaker_embedded = [item for item in embedded if item.window.speaker == speaker]
        correct = sum(1 for item in speaker_embedded if item.predicted_speaker == speaker)
        margins = [item.score_margin for item in speaker_embedded]
        own_scores = [
            (item.scores or {}).get(speaker, 0.0)
            for item in speaker_embedded
        ]
        per_speaker[speaker] = {
            "candidate_windows": len(speaker_windows),
            "accepted_windows": sum(1 for window in speaker_windows if window.status == "accepted"),
            "embedded_windows": len(speaker_embedded),
            "selection_mode": selection_modes.get(speaker, ""),
            "self_check_correct": correct,
            "self_check_accuracy": round(correct / len(speaker_embedded), 6) if speaker_embedded else 0.0,
            "mean_own_score": round(float(np.mean(own_scores)), 6) if own_scores else 0.0,
            "mean_score_margin": round(float(np.mean(margins)), 6) if margins else 0.0,
        }

    summary = {
        "created_at": utc_now(),
        "model": args.model,
        "device": device,
        "sample_rate": args.sample_rate,
        "content_root": str(args.content_root),
        "audio_files": {speaker: str(speaker_audio_path(args.content_root, speaker)) for speaker in speakers},
        "block_files": {speaker: str(speaker_blocks_path(args.content_root, speaker)) for speaker in speakers},
        "excluded_speakers": args.exclude,
        "window_sec": args.window_sec,
        "step_sec": args.step_sec,
        "min_window_sec": args.min_window_sec,
        "min_rms_db": args.min_rms_db,
        "min_loudest_margin_db": args.min_loudest_margin_db,
        "disable_loudest_filter": args.disable_loudest_filter,
        "selection_mode": args.selection_mode,
        "max_windows_per_speaker": args.max_windows_per_speaker,
        "outputs": {
            "speaker_centroids_npz": str(centroids_npz),
            "speaker_centroids_json": str(centroids_json),
            "enrollment_report_csv": str(report_csv),
            "enrollment_summary_json": str(summary_json),
        },
        "speakers": per_speaker,
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enroll known Televisa speakers with pyannote speaker embeddings.")
    parser.add_argument(
        "--content-root",
        type=Path,
        default=DEFAULT_CONTENT_ROOT,
        required=DEFAULT_CONTENT_ROOT is None,
        help=f"Televisa content directory; may also be set with {DEFAULT_CONTENT_ROOT_ENV}.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--speakers", nargs="+", default=DEFAULT_SPEAKERS)
    parser.add_argument("--exclude", nargs="+", default=DEFAULT_EXCLUDED)
    parser.add_argument("--model", default="pyannote/embedding")
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"),
        help="Optional Hugging Face token. Defaults to HF_TOKEN/HUGGINGFACE_HUB_TOKEN or cached login.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--window-sec", type=float, default=3.0)
    parser.add_argument("--step-sec", type=float, default=1.5)
    parser.add_argument("--min-window-sec", type=float, default=1.5)
    parser.add_argument("--min-rms-db", type=float, default=-70.0)
    parser.add_argument("--min-loudest-margin-db", type=float, default=3.0)
    parser.add_argument("--disable-loudest-filter", action="store_true")
    parser.add_argument(
        "--selection-mode",
        choices=["strict-loudest", "disable-loudest", "prefer-loudest"],
        default="strict-loudest",
        help="Window selection rule. prefer-loudest falls back per speaker when strict selection has too few windows.",
    )
    parser.add_argument("--max-windows-per-speaker", type=int, default=120)
    parser.add_argument("--min-accepted-per-speaker", type=int, default=3)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.disable_loudest_filter:
        args.selection_mode = "disable-loudest"
    args.content_root = args.content_root.resolve()
    if args.output_dir is None:
        args.output_dir = args.content_root / "speaker-enrollment" / time.strftime("%Y%m%d-%H%M%S")
    args.output_dir = args.output_dir.resolve()

    excluded = set(args.exclude)
    speakers = [speaker for speaker in args.speakers if speaker not in excluded]
    if not speakers:
        print("No speakers to enroll after applying exclusions.", file=sys.stderr)
        return 2

    for speaker in speakers:
        audio_path = speaker_audio_path(args.content_root, speaker)
        blocks_path = speaker_blocks_path(args.content_root, speaker)
        if not audio_path.exists():
            print(f"Missing audio file: {audio_path}", file=sys.stderr)
            return 2
        if not blocks_path.exists():
            print(f"Missing blocks file: {blocks_path}", file=sys.stderr)
            return 2

    print(f"Loading pyannote model: {args.model}", flush=True)
    inference, device = load_embedding_model(args.model, args.hf_token, args.device)
    print(f"Embedding device: {device}", flush=True)

    print("Decoding normalized audio tracks", flush=True)
    tracks = {
        speaker: decode_audio(speaker_audio_path(args.content_root, speaker), args.sample_rate, args.ffmpeg)
        for speaker in speakers
    }

    evaluated: list[Window] = []
    selected: list[Window] = []
    selection_modes: dict[str, str] = {}
    for speaker in speakers:
        blocks = read_blocks(speaker_blocks_path(args.content_root, speaker), speaker)
        candidate_tuples = generate_windows(blocks, args.window_sec, args.step_sec, args.min_window_sec)
        selection = choose_speaker_windows(speaker, candidate_tuples, tracks, args)
        selection_modes[speaker] = selection.mode
        evaluated.extend(selection.windows)
        selected.extend(selection.selected)
        print(
            f"{speaker}: {len(selection.selected)} accepted of {len(selection.windows)} candidate windows"
            f" ({selection.mode})",
            flush=True,
        )

    missing = [
        speaker
        for speaker in speakers
        if sum(1 for window in selected if window.speaker == speaker) < args.min_accepted_per_speaker
    ]
    if missing and not args.allow_partial:
        print(
            "Not enough accepted windows for: "
            + ", ".join(missing)
            + ". Lower --min-loudest-margin-db or use --allow-partial.",
            file=sys.stderr,
        )
        return 3

    if not selected:
        print("No accepted windows to embed.", file=sys.stderr)
        return 3

    selected = sorted(selected, key=lambda window: (window.speaker, window.start_sec, window.end_sec))
    print(f"Embedding {len(selected)} accepted windows", flush=True)
    embedded = embed_windows(selected, tracks, args.sample_rate, inference)
    centroids = build_centroids(embedded)
    if not centroids:
        print("No centroids produced.", file=sys.stderr)
        return 3
    score_embedded(embedded, centroids)

    write_outputs(args.output_dir, args, speakers, evaluated, embedded, centroids, device, selection_modes)
    print(f"Wrote enrollment outputs to {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
