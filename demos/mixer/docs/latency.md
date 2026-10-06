# Mixer latency

[Quick start](https://github.com/amagimedia/avplumber/blob/mixer-improv/demos/mixer/README.md) · [HTML cookbook](https://amagimedia.github.io/avplumber/demos/mixer/docs/cookbook/)

- **Cut probe:** command receipt to the first matching encoded frame; enable
  `--cut-latency-encoder janus_encoder`. [Setup and meaning](https://github.com/amagimedia/avplumber/blob/mixer-improv/doc/mixer_cut_latency.md).
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

Generate numbered clips with [frame_codes.py](https://github.com/amagimedia/avplumber/blob/mixer-improv/tests/mixer/frame_codes.py). Start the
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
