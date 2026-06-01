#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PTS_TIMEBASE = 90000.0


@dataclass
class Segment:
    start_sec: float
    end_sec: float
    start_pts: int
    end_pts: int
    target: str = "primary"
    event_count: int = 1

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class ClipBlock:
    index: int
    segment: Segment
    vad_overlap_sec: float
    vad_block_count: int
    clip_path: Path | None


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("_").lower() or "speaker"


def discover_sources(root: Path) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for path in sorted(root.glob("televisa-*.ts")):
        person = path.stem.removeprefix("televisa-")
        sources[person] = path
    return sources


def pts_to_sec(pts: int) -> float:
    return pts / PTS_TIMEBASE


def sec_to_pts(sec: float) -> int:
    return int(round(sec * PTS_TIMEBASE))


def load_vad_blocks(person_dir: Path) -> list[Segment]:
    blocks_csv = person_dir / "blocks.csv"
    if not blocks_csv.exists():
        return []

    blocks: list[Segment] = []
    with blocks_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                start_sec = float(row["start_sec"])
                end_sec = float(row["end_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            if end_sec <= start_sec:
                continue
            start_pts = int(row.get("start_pts_90k") or sec_to_pts(start_sec))
            end_pts = int(row.get("end_pts_90k") or sec_to_pts(end_sec))
            blocks.append(Segment(start_sec, end_sec, start_pts, end_pts, target="vad"))
    return blocks


def load_visual_segments(events_jsonl: Path) -> list[Segment]:
    starts: dict[str, int] = {}
    counts: dict[str, int] = {}
    segments: list[Segment] = []

    if not events_jsonl.exists():
        return segments

    with events_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_name = str(event.get("event", ""))
            target = str(event.get("target") or "primary")
            if event_name == "visual_speech_start":
                start_pts = event.get("speech_start_pts", event.get("pts"))
                if start_pts is None:
                    continue
                starts[target] = int(start_pts)
                counts[target] = 1
            elif event_name == "visual_speech_stop":
                start_pts = event.get("speech_start_pts", starts.get(target))
                end_pts = event.get("speech_end_pts", event.get("pts"))
                if start_pts is None or end_pts is None:
                    continue
                start_pts = int(start_pts)
                end_pts = int(end_pts)
                if end_pts <= start_pts:
                    continue
                segments.append(
                    Segment(
                        start_sec=pts_to_sec(start_pts),
                        end_sec=pts_to_sec(end_pts),
                        start_pts=start_pts,
                        end_pts=end_pts,
                        target=target,
                        event_count=counts.get(target, 1),
                    )
                )
                starts.pop(target, None)
                counts.pop(target, None)

    return segments


def merge_segments(segments: list[Segment], gap_sec: float, min_duration_sec: float) -> list[Segment]:
    merged: list[Segment] = []
    for segment in sorted(segments, key=lambda item: (item.start_sec, item.end_sec)):
        if not merged or segment.start_sec > merged[-1].end_sec + gap_sec:
            merged.append(segment)
            continue

        current = merged[-1]
        if segment.end_sec > current.end_sec:
            current.end_sec = segment.end_sec
            current.end_pts = segment.end_pts
        current.event_count += segment.event_count

    return [segment for segment in merged if segment.duration_sec >= min_duration_sec]


def overlap_sec(segment: Segment, blocks: list[Segment]) -> tuple[float, int]:
    total = 0.0
    count = 0
    for block in blocks:
        overlap = max(0.0, min(segment.end_sec, block.end_sec) - max(segment.start_sec, block.start_sec))
        if overlap > 0:
            total += overlap
            count += 1
    return total, count


def intersect_segments(visual_segments: list[Segment], vad_blocks: list[Segment]) -> list[Segment]:
    segments: list[Segment] = []
    for visual in visual_segments:
        for vad in vad_blocks:
            start_sec = max(visual.start_sec, vad.start_sec)
            end_sec = min(visual.end_sec, vad.end_sec)
            if end_sec <= start_sec:
                continue
            segments.append(
                Segment(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    start_pts=sec_to_pts(start_sec),
                    end_pts=sec_to_pts(end_sec),
                    target=visual.target,
                    event_count=visual.event_count,
                )
            )
    return sorted(segments, key=lambda item: (item.start_sec, item.end_sec))


def prepend_env_path(env: dict[str, str], name: str, paths: list[str]) -> None:
    existing = env.get(name)
    values = [path for path in paths if path]
    if existing:
        values.append(existing)
    env[name] = ":".join(values)


def person_output_dir(args: argparse.Namespace, person: str) -> Path:
    if args.output_layout == "per-host-run":
        return args.content_root / person / "visual-speech" / args.run_id
    return args.output_root / person


def copy_cached_analysis(args: argparse.Namespace, person: str, out_dir: Path) -> bool:
    if not args.event_cache_root:
        return False

    cache_dir = args.event_cache_root / person
    events_src = cache_dir / "visual-speech-events.jsonl"
    if not events_src.exists() or events_src.stat().st_size <= 0:
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "visual-speech-events.jsonl",
        "visual-speech-summary.json",
        "visual-speech-graph.log",
    ]:
        source = cache_dir / name
        if source.exists():
            shutil.copy2(source, out_dir / name)
    return True


def run_visual_graph(args: argparse.Namespace, person: str, source: Path, out_dir: Path) -> None:
    events_jsonl = out_dir / "visual-speech-events.jsonl"
    summary_json = out_dir / "visual-speech-summary.json"
    log_path = out_dir / "visual-speech-graph.log"

    if args.reuse_events and events_jsonl.exists() and events_jsonl.stat().st_size > 0:
        print(f"[{person}] reusing {events_jsonl}")
        return
    if args.reuse_events and copy_cached_analysis(args, person, out_dir):
        print(f"[{person}] reusing cached analysis from {args.event_cache_root / person}")
        return

    env = os.environ.copy()
    prepend_env_path(
        env,
        "LD_LIBRARY_PATH",
        [
            "/usr/local/lib",
            "/opt/tensorrt/lib",
            "/usr/local/cuda-13.0/targets/x86_64-linux/lib",
        ],
    )
    prepend_env_path(env, "PYTHONPATH", [str(args.avplumber_root)])
    env.update(
        {
            "AVP_INPUT": str(source),
            "AVP_EVENTS_JSONL": str(events_jsonl),
            "AVP_SUMMARY_JSON": str(summary_json),
            "AVP_MODEL_DIR": str(args.model_dir),
            "AVP_SOURCE_NAME": person,
            "AVP_TIMEOUT_S": str(args.graph_timeout_s),
            "AVP_VISUAL_START_THRESHOLD": str(args.visual_start_threshold),
            "AVP_VISUAL_STOP_THRESHOLD": str(args.visual_stop_threshold),
            "AVP_VISUAL_START_CONFIRM_MS": str(args.visual_start_confirm_ms),
            "AVP_VISUAL_STOP_CONFIRM_MS": str(args.visual_stop_confirm_ms),
            "AVP_VISUAL_LOG_EVERY_N": str(args.visual_log_every_n),
        }
    )

    command = [
        sys.executable,
        str(args.avplumber_root / "pyplumber" / "examples" / "visual-speech-events.py"),
    ]
    print(f"[{person}] running visual speech graph")
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_fh:
        log_fh.write("$ " + " ".join(command) + "\n\n")
        log_fh.flush()
        subprocess.run(
            command,
            cwd=args.avplumber_root,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=args.graph_timeout_s + 30,
        )
    elapsed = time.monotonic() - started
    print(f"[{person}] graph finished in {elapsed:.1f}s")


def cut_clip(args: argparse.Namespace, source: Path, clip_path: Path, start_sec: float, end_sec: float) -> None:
    if end_sec <= start_sec:
        return
    if clip_path.exists() and clip_path.stat().st_size > 0 and not args.force_clips:
        return

    command = [
        args.ffmpeg,
        "-hide_banner",
        "-loglevel",
        args.ffmpeg_log_level,
        "-y",
        "-copyts",
        "-ss",
        f"{max(0.0, start_sec):.6f}",
        "-i",
        str(source),
        "-to",
        f"{end_sec:.6f}",
        "-map",
        "0",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "disabled",
        "-muxpreload",
        "0",
        "-muxdelay",
        "0",
        "-mpegts_copyts",
        "1",
        str(clip_path),
    ]
    subprocess.run(command, check=True)
    if not clip_path.exists() or clip_path.stat().st_size <= 0:
        raise RuntimeError(f"ffmpeg produced an empty clip: {clip_path}")


def build_clip_blocks(
    args: argparse.Namespace,
    person: str,
    source: Path,
    out_dir: Path,
    visual_segments: list[Segment],
    vad_blocks: list[Segment],
) -> list[ClipBlock]:
    clips_dir = out_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_blocks: list[ClipBlock] = []
    slug = slugify(person)
    clip_candidates = visual_segments
    if args.clip_mode == "audio-visual":
        clip_candidates = [
            segment
            for segment in intersect_segments(visual_segments, vad_blocks)
            if segment.duration_sec >= args.min_duration_sec
        ]

    for segment in clip_candidates:
        vad_overlap, vad_count = overlap_sec(segment, vad_blocks)
        if args.clip_mode == "audio-visual" and vad_overlap * 1000.0 < args.min_vad_overlap_ms:
            continue

        if args.max_clips_per_person and len(clip_blocks) >= args.max_clips_per_person:
            break

        index = len(clip_blocks) + 1
        clip_name = (
            f"{slug}_visual_speech_{index:03d}_"
            f"{segment.start_sec:.3f}-{segment.end_sec:.3f}.ts"
        )
        clip_path = clips_dir / clip_name
        cut_clip(args, source, clip_path, segment.start_sec, segment.end_sec)
        clip_blocks.append(ClipBlock(index, segment, vad_overlap, vad_count, clip_path))

    return clip_blocks


def write_blocks_csv(path: Path, clip_blocks: list[ClipBlock]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "block",
                "target",
                "start_sec",
                "end_sec",
                "duration_ms",
                "start_pts_90k",
                "end_pts_90k",
                "visual_event_count",
                "vad_overlap_ms",
                "vad_overlap_pct",
                "vad_block_count",
                "clip",
            ]
        )
        for block in clip_blocks:
            segment = block.segment
            duration_ms = int(round(segment.duration_sec * 1000.0))
            overlap_ms = int(round(block.vad_overlap_sec * 1000.0))
            overlap_pct = 0.0 if segment.duration_sec <= 0 else block.vad_overlap_sec / segment.duration_sec
            writer.writerow(
                [
                    block.index,
                    segment.target,
                    f"{segment.start_sec:.6f}",
                    f"{segment.end_sec:.6f}",
                    duration_ms,
                    segment.start_pts,
                    segment.end_pts,
                    segment.event_count,
                    overlap_ms,
                    f"{overlap_pct:.3f}",
                    block.vad_block_count,
                    str(block.clip_path or ""),
                ]
            )


def write_person_report(
    args: argparse.Namespace,
    person: str,
    source: Path,
    vad_blocks: list[Segment],
    raw_visual_segments: list[Segment],
    merged_visual_segments: list[Segment],
    clip_blocks: list[ClipBlock],
    out_dir: Path,
) -> None:
    total_clip_sec = sum(block.segment.duration_sec for block in clip_blocks)
    total_vad_overlap_sec = sum(block.vad_overlap_sec for block in clip_blocks)
    vad_dir = args.content_root / person

    lines = [
        f"# {person} Visual Speech Activity",
        "",
        f"Source file: `{source}`",
        f"Output directory: `{out_dir}`",
        f"Audio reference: `{vad_dir / 'blocks.csv'}`",
        "",
        "The graph detects mouth activity from the camera feed and keeps clip candidates that overlap the existing audio VAD blocks.",
        "In audio-visual mode each clip interval is the intersection of visual mouth activity and the VAD block timeline.",
        "Clips are cut from the original TS with FFmpeg stream copy; audio and video are not transcoded.",
        "",
        "## Settings",
        "",
        f"- Clip mode: `{args.clip_mode}`",
        f"- Minimum clip duration: `{args.min_duration_sec:.3f} s`",
        f"- Merge gap: `{args.merge_gap_sec:.3f} s`",
        f"- Minimum VAD overlap: `{args.min_vad_overlap_ms:.0f} ms`",
        f"- Visual start threshold: `{args.visual_start_threshold}`",
        f"- Visual stop threshold: `{args.visual_stop_threshold}`",
        f"- Visual start confirm: `{args.visual_start_confirm_ms} ms`",
        f"- Visual stop confirm: `{args.visual_stop_confirm_ms} ms`",
        "",
        "FFmpeg clip command shape:",
        "",
        "```sh",
        "ffmpeg -copyts -ss START -i SOURCE -to END -map 0 -c copy -avoid_negative_ts disabled -mpegts_copyts 1 OUT.ts",
        "```",
        "",
        "## Summary",
        "",
        f"- Raw visual speech segments: `{len(raw_visual_segments)}`",
        f"- Merged visual speech segments: `{len(merged_visual_segments)}`",
        f"- VAD blocks loaded: `{len(vad_blocks)}`",
        f"- Clips written: `{len(clip_blocks)}`",
        f"- Total clip duration: `{total_clip_sec:.3f} s`",
        f"- Total VAD overlap inside clips: `{total_vad_overlap_sec:.3f} s`",
        "",
        "## Clips",
        "",
        "| # | start | end | duration | VAD overlap | clip |",
        "|---:|---:|---:|---:|---:|---|",
    ]

    for block in clip_blocks:
        segment = block.segment
        clip_name = block.clip_path.name if block.clip_path else ""
        lines.append(
            f"| {block.index} | {segment.start_sec:.3f} | {segment.end_sec:.3f} | "
            f"{segment.duration_sec:.3f} | {block.vad_overlap_sec:.3f} | `{clip_name}` |"
        )

    (out_dir / "speech-activity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_person(args: argparse.Namespace, person: str, source: Path) -> dict:
    out_dir = person_output_dir(args, person)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_visual_graph(args, person, source, out_dir)

    raw_visual_segments = load_visual_segments(out_dir / "visual-speech-events.jsonl")
    merged_visual_segments = merge_segments(
        raw_visual_segments,
        gap_sec=args.merge_gap_sec,
        min_duration_sec=args.min_duration_sec,
    )
    vad_blocks = load_vad_blocks(args.content_root / person)
    clip_blocks = build_clip_blocks(
        args,
        person,
        source,
        out_dir,
        merged_visual_segments,
        vad_blocks,
    )

    write_blocks_csv(out_dir / "blocks.csv", clip_blocks)
    write_person_report(
        args,
        person,
        source,
        vad_blocks,
        raw_visual_segments,
        merged_visual_segments,
        clip_blocks,
        out_dir,
    )

    return {
        "person": person,
        "source": str(source),
        "output_dir": str(out_dir),
        "raw_visual_segments": len(raw_visual_segments),
        "merged_visual_segments": len(merged_visual_segments),
        "vad_blocks": len(vad_blocks),
        "clips": len(clip_blocks),
        "clip_duration_sec": round(sum(block.segment.duration_sec for block in clip_blocks), 6),
        "vad_overlap_sec": round(sum(block.vad_overlap_sec for block in clip_blocks), 6),
    }


def write_batch_summary(output_root: Path, results: list[dict], failures: list[dict]) -> None:
    summary = {
        "results": results,
        "failures": failures,
        "total_clips": sum(item.get("clips", 0) for item in results),
        "total_clip_duration_sec": round(sum(item.get("clip_duration_sec", 0.0) for item in results), 6),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# Televisa Visual Speech Batch",
        "",
        f"Output directory: `{output_root}`",
        f"Total clips: `{summary['total_clips']}`",
        f"Total clip duration: `{summary['total_clip_duration_sec']:.3f} s`",
        "",
        "| person | visual segments | merged | VAD blocks | clips | clip duration |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['person']} | {item['raw_visual_segments']} | "
            f"{item['merged_visual_segments']} | {item['vad_blocks']} | "
            f"{item['clips']} | {item['clip_duration_sec']:.3f} |"
        )
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures:
            lines.append(f"- `{failure['person']}`: {failure['error']}")
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run visual-speech analysis and cut Televisa speaker clips.")
    parser.add_argument(
        "--content-root",
        type=Path,
        default=DEFAULT_CONTENT_ROOT,
        required=DEFAULT_CONTENT_ROOT is None,
        help=f"Televisa content directory; may also be set with {DEFAULT_CONTENT_ROOT_ENV}.",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-layout", choices=["output-root", "per-host-run"], default="output-root")
    parser.add_argument("--run-id", help="Run directory name used by --output-layout per-host-run.")
    parser.add_argument("--event-cache-root", type=Path, help="Existing output-root-style analysis cache to reuse.")
    parser.add_argument("--avplumber-root", type=Path, default=DEFAULT_AVPLUMBER_ROOT)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        required=DEFAULT_MODEL_DIR is None,
        help=f"Face model directory; may also be set with {DEFAULT_MODEL_DIR_ENV}.",
    )
    parser.add_argument("--people", nargs="*", help="Person names to process. Defaults to all televisa-*.ts files.")
    parser.add_argument("--reuse-events", action="store_true", help="Reuse existing visual-speech-events.jsonl files.")
    parser.add_argument("--keep-going", action="store_true", help="Continue processing other people after a failure.")
    parser.add_argument("--clip-mode", choices=["audio-visual", "visual"], default="audio-visual")
    parser.add_argument("--min-vad-overlap-ms", type=float, default=100.0)
    parser.add_argument("--min-duration-sec", type=float, default=3.0)
    parser.add_argument("--merge-gap-sec", type=float, default=1.0)
    parser.add_argument("--max-clips-per-person", type=int, default=0)
    parser.add_argument("--force-clips", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffmpeg-log-level", default="error")
    parser.add_argument("--graph-timeout-s", type=float, default=1200.0)
    parser.add_argument("--visual-start-threshold", type=float, default=0.16)
    parser.add_argument("--visual-stop-threshold", type=float, default=0.04)
    parser.add_argument("--visual-start-confirm-ms", type=float, default=200.0)
    parser.add_argument("--visual-stop-confirm-ms", type=float, default=1200.0)
    parser.add_argument("--visual-log-every-n", type=int, default=900)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.content_root = args.content_root.resolve()
    if not args.run_id:
        args.run_id = time.strftime("%Y%m%d-%H%M%S")
    if args.output_root is None:
        if args.output_layout == "per-host-run":
            args.output_root = args.content_root / "visual-speech-runs" / args.run_id
        else:
            args.output_root = args.content_root / "visual-speech"
    args.output_root = args.output_root.resolve()
    if args.event_cache_root is not None:
        args.event_cache_root = args.event_cache_root.resolve()
    args.avplumber_root = args.avplumber_root.resolve()
    args.model_dir = args.model_dir.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    sources = discover_sources(args.content_root)
    if args.people:
        missing = [person for person in args.people if person not in sources]
        if missing:
            print(f"Missing sources for: {', '.join(missing)}", file=sys.stderr)
            return 2
        selected = {person: sources[person] for person in args.people}
    else:
        selected = sources

    if not selected:
        print(f"No televisa-*.ts sources found under {args.content_root}", file=sys.stderr)
        return 2

    results: list[dict] = []
    failures: list[dict] = []
    for person, source in selected.items():
        try:
            result = process_person(args, person, source)
            results.append(result)
            print(f"[{person}] wrote {result['clips']} clips to {result['output_dir']}")
        except Exception as exc:
            failures.append({"person": person, "error": str(exc)})
            print(f"[{person}] failed: {exc}", file=sys.stderr)
            if not args.keep_going:
                break

    write_batch_summary(args.output_root, results, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
DEFAULT_CONTENT_ROOT_ENV = "AVP_TELEVISA_CONTENT_ROOT"
DEFAULT_MODEL_DIR_ENV = "AVP_FACE_MODEL_DIR"
DEFAULT_CONTENT_ROOT = (
    Path(os.environ[DEFAULT_CONTENT_ROOT_ENV])
    if os.environ.get(DEFAULT_CONTENT_ROOT_ENV)
    else None
)
DEFAULT_AVPLUMBER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = (
    Path(os.environ[DEFAULT_MODEL_DIR_ENV])
    if os.environ.get(DEFAULT_MODEL_DIR_ENV)
    else None
)
