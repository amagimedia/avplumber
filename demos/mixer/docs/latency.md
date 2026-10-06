# Mixer latency

[Quick start](https://github.com/amagimedia/avplumber/blob/ad371df0145fa9d9f1ed1e10397e1952eb359bfa/demos/mixer/README.md) · [HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/)

- **Cut probe:** command receipt to the first matching encoded frame; enable
  `--cut-latency-encoder janus_encoder`. [Setup and meaning](https://github.com/amagimedia/avplumber/blob/ad371df0145fa9d9f1ed1e10397e1952eb359bfa/doc/mixer_cut_latency.md).
- **Browser probe:** mouse click to `requestVideoFrameCallback().expectedDisplayTime`
  for a verified destination frame. It includes playback, but not physical display scan-out.
- Protocol acknowledgment and settled mixer state measure neither of those.

## Recorded measurements

Historical T4 runs used 16 generated 640×360 H.264 sources, portrait 60 fps output
and a same-host Chromium software decoder. These are not current capacity claims.

| Run | Samples | Median | p95 / max |
| --- | ---: | ---: | ---: |
| [Initial baseline](https://amagimedia.github.io/avplumber/demos/mixer/docs/latency-baseline.json) | 11 | 190.0 ms | 207.3 ms |
| [Later graph](https://amagimedia.github.io/avplumber/demos/mixer/docs/latency-current.json) | 11 | 188.8 ms | 503.9 ms |

The later run retains two unexplained slow trials. Encoder settings differed;
compare raw conditions before drawing conclusions. [Load samples](https://amagimedia.github.io/avplumber/demos/mixer/docs/runtime-load.json)
and [1080p load samples](https://amagimedia.github.io/avplumber/demos/mixer/docs/runtime-load-1080p.json) describe separate workloads.

## Reproduce

Generate numbered clips with [frame_codes.py](https://github.com/amagimedia/avplumber/blob/ad371df0145fa9d9f1ed1e10397e1952eb359bfa/tests/mixer/frame_codes.py). Start the
mixer, a ttyd-hosted TUI and Janus player, then enable Chromium remote debugging.
With Node.js 22+:

```sh
node tests/mixer/measure_click_latency.cjs \
  http://127.0.0.1:9222 http://127.0.0.1:8080/ \
  http://127.0.0.1:7681/ /tmp/click-latency.json
```

The probe expects the browser TUI, not the normal web controls. Keep Direct off;
inspect the calibration screenshot and adjust click coordinates for the font/layout.
It changes Program. Preserve errors, raw samples and network conditions.

## Cut spam gate

```sh
python3 tests/mixer/cut_spam.py --url http://127.0.0.1:7681 --json
```

Requires the cut probe. The test compares spaced cuts, bursts, mixed takes and
recovery against its own baseline, checks deadline misses and final scene, and
reports AUX alignment separately. See `--help` for limits and overrides.
It measures encoded cuts; fade/wipe onset is not measured. Run on the NVIDIA host.

## Native input cadence check, 2026-10-06

[Captured report](https://amagimedia.github.io/avplumber/demos/mixer/docs/mixed-fps-20261006.json)
from `tests/cuda/smoke_mixed_fps.py`: eight 320×180 CUarray sources at 25, 30,
30000/1001, 469/20, 50, 24000/1001, 60000/1001 and 120 fps feed 60 fps program
and full-rate AUX. Native code was `9f940e44`; Python pacing was `d577a12d`.
Later source-routing and colour-tag changes are outside this report's scope.

During 120.3667 seconds of steady output each stream produced 7,223 frames,
with all 7,222 PTS steps exactly 1/60 second. All 29 checks passed, including
four cuts, a one-second source stall, recovery, bounded queues and source phase.
The 60000/1001 source repeated seven frames without skipped source frames;
the 120 fps source advanced by two frames per canvas tick.

The p95 output-PTS-to-diagnostic-observation age was 52.27 ms for program and
52.12 ms for AUX. The diagnostic branch includes GPU download. This metric is
neither decoded-source-to-output latency nor command-to-picture latency and
must not be compared directly with the cut/browser probes above. The small
fixture is not a full-show capacity test or a sustained latency qualification.
