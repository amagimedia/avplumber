#!/usr/bin/env python3

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pyplumber
from pyplumber.node import DecAudio, Demux, InputRec, ResampleAudio
from pyplumber.vad import MetadataJsonlSink, SileroVADNode


INPUT_URL = os.environ.get("AVP_INPUT", "input.ts")
RUN_SECONDS = float(os.environ.get("AVP_RUN_SECONDS", "0"))
SAMPLE_RATE = int(os.environ.get("AVP_VAD_SAMPLE_RATE", "16000"))


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def build_graph() -> pyplumber.AVPlumber:
    avp = pyplumber.AVPlumber()
    avp.edges.planCapacity("*", 15)

    nodes = [
        InputRec({
            "url": INPUT_URL,
            "dst": "in_mux",
            "group": "in",
            "name": "input",
            "timeout": -1,
            "preseek": _float_env("AVP_PRESEEK", 0),
            "loop": _bool_env("AVP_LOOP", False),
            "auto_restart": "off",
        }),
        Demux({
            "src": "in_mux",
            "routing": {"a:0": "a_pkt"},
            "group": "in",
            "auto_restart": "off",
        }),
        DecAudio({
            "src": "a_pkt",
            "dst": "a_dec",
            "group": "vad",
            "name": "Audio_Decode",
            "auto_restart": "off",
        }),
        ResampleAudio({
            "src": "a_dec",
            "dst": "a_vad_pcm",
            "group": "vad",
            "name": "Audio_Resample_For_VAD",
            "auto_restart": "off",
            "dst_sample_rate": SAMPLE_RATE,
            "dst_sample_format": "flt",
            "dst_channel_layout": "mono",
            "compensation": 0,
        }),
        SileroVADNode({
            "src": "a_vad_pcm",
            "dst": "vad_events",
            "group": "vad",
            "name": "silero-vad",
            "source": os.environ.get("AVP_VAD_SOURCE", "a:0"),
            "sample_rate": SAMPLE_RATE,
            "threshold": _float_env("AVP_VAD_THRESHOLD", 0.5),
            "neg_threshold": _float_env("AVP_VAD_NEG_THRESHOLD", 0.35),
            "min_emit_ms": _int_env("AVP_VAD_MIN_EMIT_MS", 1200),
            "min_silence_ms": _int_env("AVP_VAD_MIN_SILENCE_MS", 450),
            "speech_pad_ms": _int_env("AVP_VAD_SPEECH_PAD_MS", 120),
            "window_size_samples": _int_env("AVP_VAD_WINDOW_SIZE", 512),
            "model_path": os.environ.get("AVP_SILERO_MODEL"),
            "repo_or_dir": os.environ.get("AVP_SILERO_REPO", "snakers4/silero-vad"),
            "device": os.environ.get("AVP_VAD_DEVICE", "cpu"),
        }),
        MetadataJsonlSink({
            "src": "vad_events",
            "group": "vad",
            "name": "vad-event-jsonl",
            "path": os.environ.get("AVP_VAD_JSONL"),
        }),
    ]

    for node in nodes:
        avp.addNode(node)
    return avp


def main() -> None:
    avp = build_graph()
    avp.group("in").startNodes()
    avp.group("vad").startNodes()

    start = time.monotonic()
    try:
        while RUN_SECONDS <= 0 or time.monotonic() - start < RUN_SECONDS:
            time.sleep(1)
            avp.heartbeat()
    finally:
        avp.group("vad").stopNodes()
        avp.group("in").stopNodes()


if __name__ == "__main__":
    main()
