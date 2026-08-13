# Fixed graph examples

This directory contains small, generic AVPlumber graphs. Paths and local
stream URLs are examples; adjust them before running a graph.

## Core media graphs

- `simple_audio_transcoder.avplumber` — decode, resample, and encode audio.
- `remux_with_statistics.avplumber` — remux audio/video while reporting queue statistics.
- `remux_analyze_audio.avplumber` — remux while decoding audio for analysis.
- `complicated_transcoder.avplumber` — sentinel-backed live transcoding.
- `multiaudio.avplumber` and `generate_multiaudio_graph.sh` — multi-track audio handling.
- `video_player.avplumber` and `multi_video_player.avplumber` — decoded video playback.
- `video_recorder.avplumber` and `video_recorder_waveform.avplumber` — recording and audio-envelope output.

## Hardware and metadata graphs

- `cuda_dec_scale_enc.avplumber` — CUDA decode, scale, and encode.
- `optical_flow.avplumber` — NVIDIA Optical Flow frame interpolation.
- `from_dmabuf_hwdownload.avplumber` — explicit DMA-BUF to CPU interoperability.
- `extract_cc_data.avplumber` and `extract_scte_35.avplumber` — ancillary metadata extraction.

Model-specific and sport-analysis graphs live in their application
repositories. Python neural examples remain under `pyplumber/examples/` when
they demonstrate reusable framework integration.
