# Adapting `dyremix_mixer` (avplumber-nodes) to mixer transition events

This guide is for updating **avplumber-nodes** `dyremix_mixer` (and demos such as
`audio_video_mixer`) against avplumber’s bus-scoped AFV contract. Full field
reference: [mixer_audio_follow_video.md](mixer_audio_follow_video.md).

Breaking change: there is **no** backward compatibility with `afv_program` /
`afv_crossfade`.

## Replace old keys

| Old (`setObject` key) | New | When |
| --- | --- | --- |
| `afv_program` | `transition_ended` | Steady sources after a commit |
| `afv_crossfade` | `transition_progress` | Per program video frame during fade / dip / wipe |

Ignore unknown keys; do not keep dual handlers unless you still support an
older avplumber binary.

## Handle `output`

Events carry `"output"`:

- `"program"` — main PGM path (former implicit AFV target).
- AUX bus id (`aux_buses[].id`) — only `transition_ended` (no progress on AUX).

One `audio_router_node` is enough: keep a mix/bus per `output` inside
`dyremix_mixer`. Do not assume a second router node name from the mixer.

## Handle `kind`

| `kind` | Video meaning | Suggested audio policy |
| --- | --- | --- |
| `cut` | Hard cut | Snap (or very short ramp) to `sources` at `pts` |
| `fade` | A↔B mix | Crossfade `from`→`to` with progress `t` |
| `dip` | Fade through colour (`take_dip`) | Out `from` then in `to` (not a simultaneous crossfade); use `t` (post-curve) |
| `wipe` | Media wipe | Ramp over wipe duration with linear `t` (or out/in at midpoint if product prefers) |

`kind` on AUX `transition_ended` follows the program take when the bus’s PVW
cells track a take (`cut` / `fade` / `dip` / `wipe`); layout-only changes use
`"cut"`.

## Source ids

- Program: logical `Source.id` strings — same as scene items and
  `MixerApplication.audio_edges`.
- AUX: ids from visible slot/source/pvw content; a **pgm** cell contributes
  exactly one sentinel: `"program"`. Map `"program"` to the **program audio
  bus** (follow PGM mix), not to a camera channel list.

Wire inputs from the show’s `audio_edges` by `Source.id`. Do not re-pace those
edges to wallclock; upstream `realtime` is already the audio clock.

## Hard sync with `pts`

Every event includes `"pts"` (int64 nanoseconds) on the same clock as
`realtime(set_pts=True)` / compositor frame PTS.

1. Treat `setObject` as **arming** a change, not applying it immediately
   (unless `pts == 0`).
2. When processing an audio buffer, apply gain at the sample whose timestamp
   equals `pts` (dyremix mid-buffer gain).
3. `pts == 0`: apply as soon as possible (interrupted/aborted program restore).
4. For `transition_progress`, each tick’s `pts` is that video frame’s PTS;
   interpolate gain with `t` (and `kind`) up to that time.
5. For AUX `transition_ended`, `pts` is the program frame PTS when the follow
   is timed, otherwise the aux draw deadline / apply time — schedule the same
   way.

Without scheduling on `pts`, behaviour stays soft-sync (arrival time of
`setObject`).

## Minimal handler sketch

```text
on setObject(key, params):
  output = params.output          # "program" | aux id
  kind   = params.kind            # cut|fade|dip|wipe
  pts    = params.pts             # ns; 0 = ASAP

  if key == "transition_ended":
    schedule_bus(output, sources=params.sources, kind=kind, pts=pts)

  if key == "transition_progress":
    # only output == "program" today
    schedule_bus_progress(output, from=params.from, to=params.to,
                          t=params.t, kind=kind, pts=pts)
```

## Checklist

- [ ] Remove `afv_program` / `afv_crossfade` handlers.
- [ ] Implement `transition_ended` / `transition_progress`.
- [ ] Multi-bus state keyed by `output`.
- [ ] Distinct audio curves for `fade` vs `dip` vs `wipe` vs `cut`.
- [ ] Resolve `"program"` sentinel on AUX to the program bus.
- [ ] Schedule gain at `pts` (mid-buffer); treat `pts == 0` as ASAP.
- [ ] Consume `audio_edges` by `Source.id`; no second wallclock pacer.
- [ ] Point mixer `audio_router_node` at the dyremix node name.
