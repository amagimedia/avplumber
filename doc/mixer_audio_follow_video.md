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

A show whose sources have `has_audio` needs nothing more: `build_application`
then defaults `audio_router_node` to the in-tree `audio_mixer` (an unread audio
edge would fill and hold its source's demux, and the video with it), Setup's
Janus program mountpoint (`tools/janus_mountpoints.py`) gets the Opus stream
on the first dirty Janus program rendition, and that output's entry in
`settings.program_outputs` carries `"audio": true`, which shows the viewers'
mute toggle. Set `audio_router_node` only to route to a different node.

## Events

All four are `setObject(audio_router_node_name, <key>, <params>)`.

A cut sends `transition_ended` alone. A timed take (fade, dip or wipe) sends
`transition_start` once, then `transition_progress` per program frame, with a
`transition_paused` whenever it is held or let go again, and `transition_ended`
when it commits. An AUX bus sends `transition_ended` only.

A consumer that routes on committed sources needs `transition_ended` and
nothing else; `transition_progress` adds the shape of a take in flight, and
`transition_start` / `transition_paused` describe that take in closed form for
an engine that would rather arm its own ramp than follow a per-frame value.
Unhandled keys are dropped by the node, so implementing a subset is normal —
the in-tree `audio_mixer` takes `transition_ended` and `transition_progress`
and ignores `transition_start` and `transition_paused`.

### `transition_ended`

Steady sources for one output after a change has committed.

Sent when:

- a program take completes (cut, fade, dip or wipe), including interrupted or
  aborted takes that restore the previous program;
- an AUX bus applies a new composition (layout, slot assignment, page, or a
  preview follow tied to a program take).

Params: `{"output": <id>, "kind": <kind>, "sources": [<ids>], "pts": <ns>}`; an
AUX bus's adds its own timing, `"latency_ns"` and `"program_delay_ns"` (see Fields).

### `transition_progress`

Sent once per program output video frame while a timed program take is
running (fade, dip or wipe). Not sent for cuts. Not sent for AUX.

Params: `{"output": "program", "kind": <kind>, "from": [<ids>], "to": [<ids>],
"t": <0..1>, "pts": <ns>}`.

- For fade/dip, `t` is post-curve progress (`fadeCurveAt` on elapsed/duration),
  matching the video blend's easing. The curve is only tempo; it does not
  distinguish dip from mix.
- For wipe, `t` is linear over the wipe duration (`take_duration_ns`).

`t` is the take's progress, not a gain. A consumer turning it into one should
mind that two sources heard at once add their *powers*: gains of `1 - t` and
`t` sum to half the power at the midpoint, a 3 dB sag through every fade and
wipe, where `cos(t·π/2)` and `sin(t·π/2)` hold it flat. The picture has the
opposite requirement — alpha is linear in `t`, since a mix of images is a mix
of amplitudes — so the two laws differing is correct, not a mismatch. A source
on both sides of the take is not in transition at all and should simply stay
up. `audio_mixer` does all of this in `gainsForProgress`
(`src/nodes/audio_mixer_core.hpp`); `dip` is the exception, and stays linear
because it passes through real silence rather than overlapping.

### `transition_start`

One timed program take, described in full before it runs: a consumer that owns
a ramp primitive can arm a single closed-form ramp from this instead of
chasing the per-frame `transition_progress` staircase.

Sent once when a fade, dip or wipe is armed, before any `transition_progress`
for that take. Not sent for cuts (they have no duration) and not sent for AUX.
Like `transition_progress`, it is skipped when the outgoing or incoming scene
cannot be resolved yet.

Params: `{"output": "program", "kind": <kind>, "from": [<ids>], "to": [<ids>],
"duration_ms": <ms>, "curve": <curve>, "manual": <bool>, "pts": <ns>}`.

### `transition_paused`

The take on air is held, or let go again — AUTO pressed a second time for the
scene already being taken, while a fade/dip or an armed wipe runs.

Never for a take under the T-bar (`"manual": true`): AUTO there runs the take
on from the lever instead of holding it, which is why `mixer.status` also
reports `transition_paused` only when the take is not manual.

It refers to the take `transition_start` already described, so it repeats
neither `kind` nor `from`/`to`. Program only.

Params: `{"output": "program", "paused": <bool>, "t": <0..1>, "pts": <ns>}`.

A resume does not restart the take: the orchestrator moves the take's start
forward by the time it was held, so `duration_ms * (1 - t)` of it is left to
run from `pts`. A consumer holding a closed-form ramp should stop it at `t`
and re-arm over that remainder, keeping the original `curve`.

## Fields

- **`output`**: `"program"` for the main program path; an AUX bus id
  (`aux_buses[].id`) for that bus.
- **`kind`**: `"cut"` | `"fade"` | `"dip"` | `"wipe"`. `transition_start` and
  `transition_progress` never carry `"cut"`; a cut only ever commits.
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
- **`t`** (`transition_progress`, `transition_paused`): how far the take has
  got, `0`..`1`, on the scale described under `transition_progress`.
- **`duration_ms`** (`transition_start`): the take's nominal length
  (`take_duration_ns`) in milliseconds, as a double. What a ramp armed from
  this event should span, measured from `pts`.
- **`curve`** (`transition_start`): `"linear"` | `"ease-in"` | `"ease-out"` |
  `"ease-in-out"` — the easing the video blend is using, so an audio ramp given
  the same one tracks the picture without reading `t` per frame. Always
  `"linear"` for a wipe: a wipe has no curve knob of its own and its progress
  is not eased (whatever a previous crossfade left in `take_curve` does not
  apply to it).
- **`manual`** (`transition_start`): the take follows the T-bar, not a clock.
  `duration_ms` is then a span, not a schedule — the lever can stop, reverse and
  return — so follow `transition_progress` for these instead of arming a ramp.
- **`paused`** (`transition_paused`): `true` holds the take at `t`, `false` runs
  it on from `t`.
- **`pts`**: media time in nanoseconds on the same clock as `realtime(set_pts=True)`
  / compositor frame PTS (host monotonic after rebase). The video state described
  by the event applies at this time. `0` means “as soon as the router can”; the
  in-tree orchestrator always sends a real pts, including when a take is
  interrupted, so a router sees `0` only from another caller. Routers that can change gain
  mid-buffer should **schedule** the change at `pts`, not at `setObject` arrival.
  On `transition_start` it is the moment the take was received, which is what its
  progress is measured from — the first blended frame, and so the first
  `transition_progress`, lands at or after it. On `transition_paused` it is when
  the hold or the resume takes effect.
- **`latency_ns`** / **`program_delay_ns`** (AUX only): the bus compositor's
  playout latency, and how far back its `pgm` cell shows the program
  (`pgm_delay_frames` aux ticks; 0 without a `pgm` cell). Its follower sends them
  with every change, its first composition included. A router plays the bus's
  `"program"` audio that much later than the program bus, and its blocks
  `latency_ns` after their media time, so the AUX audio stays with its picture.

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
protocol: multi-input fltp sum, multi-bus keyed by `output`, gain changes
scheduled at `pts` and slewed over a few tens of milliseconds (so a
frame-rate `transition_progress` staircase and a cut do not click).

It handles `transition_ended` and `transition_progress` only. Following
progress per frame is all it needs, since the slew already smooths the
staircase, so it has no use for `transition_start`'s duration and curve and
nothing to hold on `transition_paused` — its `setObject` drops both keys
without error, and a held take simply stops moving its gains. A consumer with
its own ramp primitive, such as an external `dyremix_mixer`, is the case those
two events exist for. With
`MixerOptions.audio_router_node` set, `has_audio` sources, and
`MixerOptions.build_audio_mixer` left at its default `True`, `build_application`
wires it via `pyplumber.mixer.audio_mixer.wire_audio_mixer` and can send the
program bus to Janus Opus (`build_janus_audio_output`); it also wires the
per-source gain controller (`audio_gain.py`), which speaks `audio_mixer`'s own
`set_source_gain` control command.

### The show's `audio` section: processors on the mixer's channels

`audio_mixer`'s channels are made automatically. There is an input per
`has_audio` source, named by the source id, and an output per bus: `program`,
plus each AUX bus whose first rendition has an `audio_port`, named by the bus id.
The show document's optional `audio` section creates none of them. Each entry
names one of these channels and adds instructions to it. For now the only
instruction is `processors` on an output, and the only processor type is a
sidechain `ducker` (see `audio_mixer` in `doc/NODES.md` for what it does):

```json
"audio": {"channels": [
  {"id": "program", "type": "output", "processors": [
    {"id": "program-duck", "type": "ducker",
     "target": ["rene", "genaro"], "sidechain": ["denise", "leo", "ray", "sergio"],
     "sidechain_tap": "post", "enabled": true,
     "threshold_db": -40, "depth_db": -20, "attack_ms": 10, "hold_ms": 300, "release_ms": 500}]}]}
```

- `id` and `type` name an existing channel. An entry for a channel the mixer did
  not make, such as a typo or an AUX bus without audio, stops the build.
- `target` and `sidechain` name input channels. Both are required and must not
  overlap. Every other field has the default shown above.
- To duck on several outputs, give each output its own entry with its own
  parameters.
- The section needs the built-in `audio_mixer`: it is an error with
  `build_audio_mixer = False`.

`pyplumber.mixer.audio_processors` parses and resolves the section. It also
registers the `mixer.audio_processor` / `mixer.audio_processor_status` control
commands. The web UI's `/levels` page shows each ducker with live parameters, a
gain-reduction meter and "Copy JSON", which returns the section with the current
values. Changes made there last until the mixer restarts.

To put a different `IInputsObjects` impl in this slot instead (e.g.
`dyremix_mixer`), set `MixerOptions.build_audio_mixer = False`:
`build_application` then leaves `audio_router_node` untouched instead of
building `audio_mixer` under that name, and skips the `audio_gain.py` gain
controller (it only understands `audio_mixer`'s protocol). The caller is then
responsible for building and wiring that node itself — including Janus program
audio and the `{audio_router_node}_{bus.id}` aux-leg edges `aux.py` expects for
AUX buses with an `audio_port` — before `MixerApplication.start()` runs.

## One `audio_router_node` vs several consumers

The mixer notifies **one** node name. That is enough when a single node keeps a
mix/bus per `output` (as `audio_mixer` does).

To use different consumers per output without changing the orchestrator — e.g.
`audio_mixer` for `program`, another AFV consumer for one AUX, and no audio for
others — register a thin fan-out node as `audio_router_node` (with
`MixerOptions.build_audio_mixer = False`, same as any other custom
`IInputsObjects` impl — see above). It implements `IInputsObjects::setObject`,
reads `params.output`, and forwards (or drops) to the chosen node.

Conditions when mixing consumers:

- Each consumer must receive the show’s `audio_edges` it needs (shared wiring or
  separate splits).
- An AUX `sources` list may contain the sentinel `"program"`. That consumer must
  hear the **program audio bus** (a tap of `audio_mixer`’s program output, or an
  equivalent bus in the other consumer) — not an expanded list of camera ids.
  The orchestrator does not wire that tap.
