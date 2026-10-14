# Audio-follow-video (AFV)

The mixer's own data path still carries video frames only (see
[Mixer base API](mixer.md)); it has no audio routing, mixing or output of its
own. What it does have is a semantic signal: as the program and preview
change, the orchestrator tells one named external node which sources should
be audible and how a crossfade between them should progress. That node — any
audio mixer/router implementing `IInputsObjects::setObject` — owns the actual
routing and gain policy. The orchestrator never touches an audio matrix or
mixer state directly.

## Enabling it

`mixer.init` accepts an optional `audio_router_node`: the name of the node to
notify. Leaving it unset (the default) makes every AFV call below a no-op, so
existing video-only setups are unaffected.

```text
mixer.init {"mixer":"mixer", ..., "audio_router_node":"audio_mixer"}
```

From Python, `MixerOptions.audio_router_node` (default `""`) flows through
`build_application` to `MixerGraphBuilder`; passing it to `MixerGraphBuilder`
directly works the same way for applications that build the graph by hand.

## What the orchestrator sends

Both calls are `setObject(audio_router_node_name, <key>, <params>)`:

- **`afv_program`** — sent on every cut and on completion of every take (cut,
  crossfade or wipe alike), including interrupted/aborted ones that restore
  the previous program. Params: `{"sources": [<ids of the new PGM scene's
  sources>]}`.
- **`afv_crossfade`** — sent once per output video frame for the duration of
  an active crossfade, at the same progress fraction (post-curve) the video
  blend uses. Params: `{"from": [<outgoing scene's sources>], "to": [<incoming
  scene's sources>], "t": <0..1>}`. Not sent during a cut or a wipe — those
  have no gradual video blend for audio to track.

The node maps `sources`/`from`/`to` (logical source ids, the same ones used in
scene definitions) onto its own channels and decides how to ramp gain; the
orchestrator has no opinion on gain curves beyond the crossfade progress it
reports.

## Feeding the node its audio

The mixer application can additionally decode a resampled audio edge per
source, for the audio router node to consume — it does not route this audio
itself, only hands out the edges:

- `Source.has_audio: bool` (default `false`, `kind: "video"` sources only)
  demuxes/decodes/resamples that source's audio track to fltp/48kHz, keeping
  its own channel layout (mono, stereo, 5.1, ...), then paces it through
  `realtime(set_pts=True)` on the same `RealTimeTeam` as that source's video
  (`input_{index}_av_sync` when no caller `sync_team` is supplied). Video is
  `master=true`, audio `master=false`; both use the same `tick_period` /
  timebase so `RealTimeTeam::checkTimeBase` accepts them and lip-sync shares
  one wallclock offset.
- `MixerApplication.audio_edges: tuple[(source.id, edge_name), ...]` on the
  built application lists one entry per `has_audio` source, each a
  realtime-paced edge not yet connected to anything. The caller builds and
  wires its own audio router/mixer node from these, then passes that node's
  name as `audio_router_node`. The router must not re-pace to wallclock;
  upstream `realtime` is already the audio clock.
- `pyplumber.mixer.inputs.build_input(..., has_audio=True)` is the lower-level
  helper behind this, for applications building inputs directly. It returns
  `(video_edge, audio_edge_or_None)` instead of a single edge.
