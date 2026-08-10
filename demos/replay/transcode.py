#!/usr/bin/env python3
"""Convert one VOD into a video-only, seekable AVPlumber replay recording."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from replay import (HISTORY_STRUCT, HISTORY_SUFFIX, SEEK_TABLE_SUFFIX,
                    TEXT_SEEK_TABLE_SUFFIX, TranscodeConfig,
                    build_transcode_application, read_seek_table,
                    validate_recording)


SEEK_TABLE_SUFFIXES = (SEEK_TABLE_SUFFIX, TEXT_SEEK_TABLE_SUFFIX)
SEEK_TABLE_BACKING_FILES = 4


def parse_wallclock_start(value: str, *, now=lambda: datetime.now(timezone.utc)) -> datetime:
    if value == "now":
        return now().astimezone(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("--wallclock-start must include Z or a timezone offset")
    return parsed.astimezone(timezone.utc)


def output_family(output: Path) -> tuple[Path, ...]:
    canonical = [
        output,
        Path(f"{output}{SEEK_TABLE_SUFFIX}"),
        Path(f"{output}{TEXT_SEEK_TABLE_SUFFIX}"),
        Path(f"{output}{HISTORY_SUFFIX}"),
    ]
    backing = [
        Path(f"{output}{suffix}.{index}")
        for suffix in SEEK_TABLE_SUFFIXES
        for index in range(SEEK_TABLE_BACKING_FILES)
    ]
    return tuple(canonical + backing)


def output_collisions(output: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in output_family(output) if os.path.lexists(path)))


def write_history(
    path: Path,
    *,
    first_timestamp_ms: int,
    wallclock_start: datetime,
) -> None:
    wallclock_ms = round(wallclock_start.timestamp() * 1000)
    path.write_bytes(HISTORY_STRUCT.pack(0, 0, first_timestamp_ms - wallclock_ms, 0))


def parse_args(argv: list[str] | None = None) -> TranscodeConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, metavar="PATH")
    parser.add_argument("--output", required=True, type=Path, metavar="PATH")
    parser.add_argument("--fps", required=True, type=int)
    parser.add_argument("--wallclock-start", default="now", metavar="ISO8601|now")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        wallclock_start = parse_wallclock_start(args.wallclock_start)
        return TranscodeConfig(
            args.input.resolve(),
            args.output.resolve(),
            args.fps,
            wallclock_start,
            args.force,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


def _validate_text_seek(path: Path, expected_count: int) -> None:
    lines = path.read_text().splitlines()
    if len(lines) != expected_count:
        raise ValueError(
            f"text seek table has {len(lines)} entries; expected {expected_count}"
        )
    for line in lines:
        timestamp, byte_offset = line.split()
        int(timestamp), int(byte_offset)


def _publish(staged: Path, output: Path) -> None:
    stage_dir = staged.parent
    materialized = {}
    for suffix in SEEK_TABLE_SUFFIXES:
        source = Path(f"{staged}{suffix}")
        regular = stage_dir / f"publish{suffix}"
        shutil.copyfile(source.resolve(strict=True), regular)
        materialized[suffix] = regular

    for suffix in SEEK_TABLE_SUFFIXES:
        for index in range(SEEK_TABLE_BACKING_FILES):
            source = Path(f"{staged}{suffix}.{index}")
            if source.exists():
                os.replace(source, Path(f"{output}{suffix}.{index}"))
    os.replace(materialized[SEEK_TABLE_SUFFIX], Path(f"{output}{SEEK_TABLE_SUFFIX}"))
    os.replace(
        materialized[TEXT_SEEK_TABLE_SUFFIX],
        Path(f"{output}{TEXT_SEEK_TABLE_SUFFIX}"),
    )
    os.replace(Path(f"{staged}{HISTORY_SUFFIX}"), Path(f"{output}{HISTORY_SUFFIX}"))
    os.replace(staged, output)


def run(config: TranscodeConfig) -> None:
    if not config.output.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {config.output.parent}")
    collisions = output_collisions(config.output)
    if collisions and not config.force:
        listing = "\n".join(f"  {path}" for path in collisions)
        raise FileExistsError(f"output family already exists:\n{listing}")

    stage_dir = Path(tempfile.mkdtemp(prefix=f".{config.output.name}.", dir=config.output.parent))
    staged = stage_dir / config.output.name
    staged_config = replace(config, output=staged)
    try:
        application = build_transcode_application(staged_config)
        errors = []
        application.avp.on_exception = (
            lambda name, node_type, message: errors.append(f"{name} ({node_type}): {message}")
        )
        application.run()
        if errors:
            raise RuntimeError(errors[-1])

        seek_entries = read_seek_table(Path(f"{staged}{SEEK_TABLE_SUFFIX}"))
        write_history(
            Path(f"{staged}{HISTORY_SUFFIX}"),
            first_timestamp_ms=seek_entries[0].timestamp_ms,
            wallclock_start=config.wallclock_start,
        )
        artifact = validate_recording(staged)
        _validate_text_seek(
            Path(f"{staged}{TEXT_SEEK_TABLE_SUFFIX}"),
            artifact.frame_count,
        )
        _publish(staged, config.output)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    try:
        config = parse_args(argv)
        run(config)
    except Exception as exc:
        print(f"transcode failed: {exc}")
        return 1
    print(
        f"Replay recording: {config.output} ({config.fps} fps, "
        f"wallclock {config.wallclock_start.isoformat()})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
