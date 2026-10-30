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

## Configure and extend

Same docs as the video demo under [docs/](docs/) and the shared
[mixer base API](../../doc/mixer.md). Fan-out notes:
[mixer_audio_follow_video.md](../../doc/mixer_audio_follow_video.md).
