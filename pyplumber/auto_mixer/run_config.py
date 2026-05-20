from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import NoReturn

from .config import FPS_DEN, FPS_NUM
from .inputs import find_named_input
from .profiles.talkshow import GENARO_INPUT_NAME, RENE_INPUT_NAME, SERGIO_INPUT_NAME


@dataclass(frozen=True)
class AutoMixerRunConfig:
    n_inputs: int
    janus_enabled: bool
    record_enabled: bool
    switch_pts_lead_ms: int | None
    sergio_input_index: int | None
    rene_input_index: int | None
    genaro_input_index: int | None
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


def derive_run_config(args, parser: argparse.ArgumentParser | None = None) -> AutoMixerRunConfig:
    n = len(args.inputs)
    if n < 2:
        _config_error(parser, "At least 2 inputs are required.")
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

    return AutoMixerRunConfig(
        n_inputs=n,
        janus_enabled=janus_enabled,
        record_enabled=record_enabled,
        switch_pts_lead_ms=args.switch_pts_lead_ms if args.switch_pts_lead_ms >= 0 else None,
        sergio_input_index=find_named_input(args.inputs, SERGIO_INPUT_NAME),
        rene_input_index=find_named_input(args.inputs, RENE_INPUT_NAME),
        genaro_input_index=find_named_input(args.inputs, GENARO_INPUT_NAME),
        output_targets=_output_targets(
            args,
            record_enabled=record_enabled,
            janus_enabled=janus_enabled,
        ),
    )
