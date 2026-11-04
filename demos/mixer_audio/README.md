# Live mixer demo with audio-follow-video

Copy of [`demos/mixer`](../mixer) with open-source **`audio_mixer`** wired as
`audio_router_node`: encoded video sources get `has_audio`, distinct sine tones
in synthetic MP4s, and the program mix is published to Janus Opus (port 5002)
alongside program video.

The original [demos/mixer](../mixer) stays video-only.

Protocol: [Audio-follow-video](../../doc/mixer_audio_follow_video.md). Node:
`src/nodes/audio_mixer.cpp`.

Everything else tracks [`demos/mixer`](../mixer): the setup page, instance
profiles, patterns and docs are kept in step with it, so only the files listed
below carry audio changes.

## Differences from `demos/mixer`

- `prepare_demo` marks `kind: video` MP4/file sources `has_audio: true` and
  regenerates synthetic MP4s (cache `synthetic_v6_*`, one ahead of the video
  demo's `v5`) with AAC sine tones.
- The mixer wires `audio_mixer` for those sources by itself (doc/mixer_audio_follow_video.md).
- `build_application` adds `build_janus_audio_output` with Janus output.
- `audio_port` on a rendition ([docs/config.md](docs/config.md#renditions)):
  on the program it duplicates the program mix (the HDR output in
  `demo.example.json`), on an AUX bus's rendition it gives that bus its own
  independently mixed `audio_mixer` bus. `demo.example.json` ships three AUX
  buses, two of them with audio and one carrying Preview.
- `webui.py` defaults `--audio-router-node audio_mixer`, and the Compose stack
  enables `JANUS_AUDIO_ENABLED` on Janus.

## Run

Same host requirements as the video mixer. Point Compose at this package, or
run the video-mixer Compose with:

```sh
docker compose -f docker-compose/mixer_audio/compose.yaml up --build
```

Ensure mixer args include `--janus-output` (default in Compose) so Opus RTP is
built. Hear AFV by cutting/fading between scenes whose sources have audio.

## Lip-sync test sources

`lipsync_test: true` in the recipe (set in `demo.example.json` and
`demo.equal.json`) adds two extra synthetic sources, `synctest_a_000` and
`synctest_b_000`: a dark field with a brief full-frame flash and a matching
audio beep landing together on every whole second — white flash / 1000 Hz for
`a`, cyan flash / 1600 Hz for `b`. Cut to either in the web UI (they rotate
into `fullscreen_*` scenes like any other source, or cut to them directly by
id) and watch/listen for the flash and beep landing together; a lag between
them is audio/video desync. Unlike the per-source AFV tones (continuous, just
there so a cut is audible), these mark *when* audio and video should
coincide, not just *which* source you're hearing. Longer `generation.seconds`
gives more flash/beep cycles to judge before the clip loops.

## Configure and extend

Same docs as the video demo under [docs/](docs/) and the shared
[mixer base API](../../doc/mixer.md). Fan-out notes:
[mixer_audio_follow_video.md](../../doc/mixer_audio_follow_video.md); the
`audio_port` field is the one addition to [docs/config.md](docs/config.md).

## Operating this demo

- [OPS.md](OPS.md) — cheat sheet for the demo host: bring-up, the fast
  Python-only iteration loop, and the traps around recreating containers.
- [docs/audio_load_cost.md](docs/audio_load_cost.md) — what the audio mixer
  costs next to `demos/mixer` at the same source count.
- [scripts/](scripts/) — `build.sh`, `start.sh`, `stop-audio-video-mixer.sh`,
  `status.sh`, `sync.sh` and `sample_load.py` for the measurements.
