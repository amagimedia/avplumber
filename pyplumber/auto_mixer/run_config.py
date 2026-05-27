from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import NoReturn

from .config import FPS_DEN, FPS_NUM
from .output_formats import infer_record_output_format
from .profiles.talkshow import (
    DEFAULT_PROGRAM_AUDIO_INPUT_INDEX,
    DEFAULT_SPECIAL_SPEAKER_INDEX,
    DEFAULT_SPECIAL_SPEAKER_MARGIN_DB,
    DEFAULT_STATIC_FACE_CROP_INPUTS,
    DEFAULT_VAD_ONLY_PRIORITY_SPEAKER_INDEX,
)


@dataclass(frozen=True)
class AutoMixerRunConfig:
    n_inputs: int
    janus_enabled: bool
    record_enabled: bool
    switch_pts_lead_ms: int | None
    talkshow_profile: bool
    program_audio_input_index: int
    special_speaker_index: int | None
    special_speaker_margin_db: float
    vad_only_priority_speaker_index: int | None
    static_face_crop_inputs: tuple[int, ...]
    record_output_format: str | None
    output_targets: tuple[str, ...]


def _config_error(parser: argparse.ArgumentParser | None, message: str) -> NoReturn:
    if parser is not None:
        parser.error(message)
    raise ValueError(message)


def _output_targets(args, *, record_enabled: bool, janus_enabled: bool) -> tuple[str, ...]:
    targets: list[str] = []
    if record_enabled:
        targets.append(args.output)
    if janus_enabled:
        target = f"Janus RTP video={args.janus_host}:{args.janus_video_port}"
        if not args.janus_video_only:
            target += f" audio={args.janus_host}:{args.janus_audio_port}"
        targets.append(target)
    return tuple(targets)


def _validate_input_index(
    parser: argparse.ArgumentParser | None,
    *,
    name: str,
    value: int,
    n_inputs: int,
) -> int:
    if not 0 <= value < n_inputs:
        _config_error(parser, f"{name} must be between 0 and {n_inputs - 1}.")
    return value


def _optional_input_index(
    args,
    parser: argparse.ArgumentParser | None,
    *,
    attr: str,
    name: str,
    n_inputs: int,
    default: int | None = None,
) -> int | None:
    value = getattr(args, attr, None)
    if value is None:
        value = default
    if value is None:
        return None
    return _validate_input_index(parser, name=name, value=value, n_inputs=n_inputs)


def _static_face_crop_inputs(
    args,
    parser: argparse.ArgumentParser | None,
    *,
    n_inputs: int,
    talkshow_profile: bool,
) -> tuple[int, ...]:
    values = []
    if talkshow_profile:
        values.extend(DEFAULT_STATIC_FACE_CROP_INPUTS)
    values.extend(getattr(args, "static_face_crop_input", ()) or ())

    result = []
    seen = set()
    for value in values:
        index = _validate_input_index(
            parser,
            name="--static-face-crop-input",
            value=value,
            n_inputs=n_inputs,
        )
        if index not in seen:
            seen.add(index)
            result.append(index)
    return tuple(result)


def derive_run_config(args, parser: argparse.ArgumentParser | None = None) -> AutoMixerRunConfig:
    n = len(args.inputs)
    if n < 2:
        _config_error(parser, "At least 2 inputs are required.")
    talkshow_profile = bool(getattr(args, "talkshow_profile", False))
    janus_enabled = args.janus_preview or args.janus_output
    record_enabled = args.output is not None
    if not record_enabled and not janus_enabled:
        _config_error(parser, "Either --output or --janus-output/--janus-preview is required.")
    if args.janus_preview and not record_enabled and not args.janus_output:
        _config_error(parser, "--janus-preview needs --output; use --janus-output for Janus-only.")
    if args.janus_video_bitrate_kbps <= 0:
        _config_error(parser, "--janus-video-bitrate-kbps must be greater than 0.")
    if 0 <= args.switch_pts_lead_ms < 100:
        _config_error(parser, "--switch-pts-lead-ms must be at least 100, or negative to disable PTS scheduling.")
    if args.fade_frames <= 0:
        _config_error(parser, "--fade-frames must be greater than 0.")
    if args.fade is None:
        args.fade = args.fade_frames * FPS_DEN / FPS_NUM
    else:
        args.fade_frames = max(1, int(round(args.fade * FPS_NUM / FPS_DEN)))
    if args.fade <= 0.0:
        _config_error(parser, "--fade must be greater than 0.")
    if args.auto_switch_transition == "wipe" and not args.wipe:
        _config_error(parser, "--auto-switch-transition wipe requires the wipe subgraph.")

    program_audio_default = DEFAULT_PROGRAM_AUDIO_INPUT_INDEX if talkshow_profile else 0
    program_audio_input_index = _optional_input_index(
        args,
        parser,
        attr="program_audio_input",
        name="--program-audio-input",
        n_inputs=n,
        default=program_audio_default,
    )
    if program_audio_input_index is None:
        program_audio_input_index = 0

    special_speaker_index = _optional_input_index(
        args,
        parser,
        attr="special_speaker_index",
        name="--special-speaker-index",
        n_inputs=n,
        default=DEFAULT_SPECIAL_SPEAKER_INDEX if talkshow_profile else None,
    )
    vad_only_priority_speaker_index = _optional_input_index(
        args,
        parser,
        attr="vad_only_priority_speaker_index",
        name="--vad-only-priority-speaker-index",
        n_inputs=n,
        default=DEFAULT_VAD_ONLY_PRIORITY_SPEAKER_INDEX if talkshow_profile else None,
    )
    static_face_crop_inputs = _static_face_crop_inputs(
        args,
        parser,
        n_inputs=n,
        talkshow_profile=talkshow_profile,
    )
    special_speaker_margin_db = float(
        getattr(args, "special_speaker_margin_db", DEFAULT_SPECIAL_SPEAKER_MARGIN_DB)
    )
    record_output_format = None
    if record_enabled:
        try:
            record_output_format = infer_record_output_format(
                args.output,
                getattr(args, "output_format", None),
            )
        except ValueError as exc:
            _config_error(parser, str(exc))

    return AutoMixerRunConfig(
        n_inputs=n,
        janus_enabled=janus_enabled,
        record_enabled=record_enabled,
        switch_pts_lead_ms=args.switch_pts_lead_ms if args.switch_pts_lead_ms >= 0 else None,
        talkshow_profile=talkshow_profile,
        program_audio_input_index=program_audio_input_index,
        special_speaker_index=special_speaker_index,
        special_speaker_margin_db=special_speaker_margin_db,
        vad_only_priority_speaker_index=vad_only_priority_speaker_index,
        static_face_crop_inputs=static_face_crop_inputs,
        record_output_format=record_output_format,
        output_targets=_output_targets(
            args,
            record_enabled=record_enabled,
            janus_enabled=janus_enabled,
        ),
    )
