"""Wire the open-source ``audio_mixer`` node as the mixer's ``audio_router_node``.

Orchestrator / AUX followers emit ``transition_ended`` / ``transition_progress`` to
**one** node name. This helper builds that consumer from ``MixerApplication.audio_edges``.

To route different outputs to different consumers (e.g. ``audio_mixer`` for program,
another AFV consumer for one AUX, none for others) without changing the orchestrator,
put a thin fan-out ``IInputsObjects`` node as ``audio_router_node`` and forward by
``output``. Each consumer still needs its ``audio_edges`` (shared or separate); an AUX
cell with the ``program`` sentinel needs a tap of the program audio bus.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple


def wire_audio_mixer(
    avp,
    api,
    audio_edges: Sequence[Tuple[str, str]],
    *,
    name: str = "audio_mixer",
    outputs: Optional[Mapping[str, str]] = None,
    dst_channel_layout: str = "stereo",
    dst_sample_rate: int = 48_000,
    frame_samples: int = 960,
    group: str = "output",
) -> str:
    """Add an ``audio_mixer`` node; return its name for ``MixerOptions.audio_router_node``.

    ``audio_edges`` is ``((source_id, edge_name), ...)`` from ``MixerApplication``.
    ``outputs`` maps bus id (``program`` / AUX id) to destination edge names; default is
    a single ``program`` bus named ``{name}_program``.
    """
    if not audio_edges:
        raise ValueError("wire_audio_mixer: audio_edges is empty (set has_audio on video sources)")
    source_ids = [sid for sid, _ in audio_edges]
    src = [edge for _, edge in audio_edges]
    if outputs is None:
        outputs = {"program": f"{name}_program"}
    avp.addNode(api.AudioMixer({
        "name": name,
        "src": src,
        "source_ids": source_ids,
        "outputs": dict(outputs),
        "dst_channel_layout": dst_channel_layout,
        "dst_sample_rate": dst_sample_rate,
        "frame_samples": frame_samples,
        "group": group,
    }))
    return name
