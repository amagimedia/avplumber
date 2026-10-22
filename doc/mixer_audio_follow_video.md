# Output transition events (AFV)

The mixer's own data path still carries video frames only (see
[Mixer base API](mixer.md)); it has no audio routing, mixing or output of its
own. What it does have is a semantic signal: as program and AUX compositions
change, the mixer tells one named external node which sources apply to each
output and how a timed take is progressing. That node — any
`IInputsObjects::setObject` impl, typically an audio mixer/router — owns the
actual routing and gain policy. The mixer never touches an audio matrix
directly.

## Enabling it

`mixer.init` accepts an optional `audio_router_node`: the name of the node to
notify. Leaving it unset (the default) makes every event below a no-op, so
existing video-only setups are unaffected.

```text
mixer.init {"mixer":"mixer", ..., "audio_router_node":"audio_mixer"}
```

From Python, `MixerOptions.audio_router_node` (default `""`) flows through
`build_application` to `MixerGraphBuilder`; passing it to `MixerGraphBuilder`
directly works the same way for applications that build the graph by hand.

## Events

Both are `setObject(audio_router_node_name, <key>, <params>)`:

### `transition_ended`

Steady sources for one output after a change has committed.

Sent when:

- a program take completes (cut, fade, dip or wipe), including interrupted or
  aborted takes that restore the previous program;
- an AUX bus applies a new composition (layout, slot assignment, page, or a
  preview follow tied to a program take).

Params: `{"output": <id>, "kind": <kind>, "sources": [<ids>], "pts": <ns>}`.

### `transition_progress`

Sent once per program output video frame while a timed program take is
running (fade, dip or wipe). Not sent for cuts. Not sent for AUX.

Params: `{"output": "program", "kind": <kind>, "from": [<ids>], "to": [<ids>],
"t": <0..1>, "pts": <ns>}`.

- For fade/dip, `t` is post-curve progress (`fadeCurveAt` on elapsed/duration),
  matching the video blend's easing. The curve is only tempo; it does not
  distinguish dip from mix.
- For wipe, `t` is linear over the wipe duration (`take_duration_ns`).

## Fields

- **`output`**: `"program"` for the main program path; an AUX bus id
  (`aux_buses[].id`) for that bus.
- **`kind`**: `"cut"` | `"fade"` | `"dip"` | `"wipe"`.
  - `"fade"` — mix A↔B (`Crossfade` without a dip colour).
  - `"dip"` — fade through colour (`Crossfade` with `take_dip` /
    `fade_color`); audio typically outs then ins, not a crossfade.
  - Dip is not a separate `TransitionMode`; it is reported as its own `kind`.
  - The same `kind` is published on `PreviewChange` (and thus AUX follow /
    `pvw_latency`) so followers agree with these events.
- **`sources` / `from` / `to`**: ids for channel selection on the audio router:
  - **Program:** logical `Source.id` values from the scene (same strings as
    scene items and `MixerApplication.audio_edges`).
  - **AUX:** `Source.id` from `slot` / `source` / `pvw` cells; a **`pgm` cell
    contributes the single sentinel `"program"`** (not the expanded list of
    sources on air). The router treats `"program"` as the program mix/bus.
- **`pts`**: media time in nanoseconds on the same clock as `realtime(set_pts=True)`
  / compositor frame PTS (host monotonic after rebase). The video state described
  by the event applies at this time. `0` means “as soon as the router can”
  (e.g. interrupted restore with no future frame). Routers that can change gain
  mid-buffer should **schedule** the change at `pts`, not at `setObject` arrival.

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

## Open-source consumer: `audio_mixer`

In-tree node `audio_mixer` (`src/nodes/audio_mixer.cpp`) implements this
protocol: multi-input fltp sum, multi-bus keyed by `output`, hard-sync gain at
`pts`. With `MixerOptions.audio_router_node` set and `has_audio` sources,
`build_application` wires it via `pyplumber.mixer.audio_mixer.wire_audio_mixer`
and can send the program bus to Janus Opus (`build_janus_audio_output`).

## One `audio_router_node` vs several consumers

The mixer notifies **one** node name. That is enough when a single node keeps a
mix/bus per `output` (as `audio_mixer` does).

To use different consumers per output without changing the orchestrator — e.g.
`audio_mixer` for `program`, another AFV consumer for one AUX, and no audio for
others — register a thin fan-out node as `audio_router_node`. It implements
`IInputsObjects::setObject`, reads `params.output`, and forwards (or drops) to
the chosen node.

Conditions when mixing consumers:

- Each consumer must receive the show’s `audio_edges` it needs (shared wiring or
  separate splits).
- An AUX `sources` list may contain the sentinel `"program"`. That consumer must
  hear the **program audio bus** (a tap of `audio_mixer`’s program output, or an
  equivalent bus in the other consumer) — not an expanded list of camera ids.
  The orchestrator does not wire that tap.
