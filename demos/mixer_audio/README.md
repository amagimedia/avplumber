# Live mixer demo with audio-follow-video

Copy of [`demos/mixer`](../mixer) with open-source **`audio_mixer`** wired as
`audio_router_node`: encoded video sources get `has_audio`, distinct sine tones
in synthetic MP4s, and the program mix is published to Janus Opus (port 5002)
alongside program video.

The original [demos/mixer](../mixer) stays video-only.

Protocol: [Audio-follow-video](../../doc/mixer_audio_follow_video.md). Node:
`src/nodes/audio_mixer.cpp`.

## Differences from `demos/mixer`

- `prepare_demo` marks `kind: video` MP4/file sources `has_audio: true` and
  regenerates synthetic MP4s (cache `synthetic_v6_*`) with AAC sine tones.
- The mixer wires `audio_mixer` for those sources by itself (doc/mixer_audio_follow_video.md).
- `build_application` adds `build_janus_audio_output` with Janus output.

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
[mixer_audio_follow_video.md](../../doc/mixer_audio_follow_video.md).
