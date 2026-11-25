# Host sync (experimental)

Two avplumber instances on different hosts can release the same source frame in the same program
tick and stamp their outputs on one shared show timeline. A backup mixer then airs the picture the
main one airs, frame for frame, which is what redundancy across instances needs.

**Status.** Merged in [#70](https://github.com/amagimedia/avplumber/pull/70). Tested with two mixers in
two containers on one L4, the second with its monotonic clock 12345 s off. At 30, 30000/1001 and
60000/1001 fps their MPEG-TS outputs had the same picture at every PTS after sync, keyframes on the
same PTS, and timed cuts landed on the same tick 10 out of 10 times. **Not yet run on two real hosts
with real clock skew**, and browser sources were not part of that test.

## Requirements

- `AVP_CLOCK_ORIGIN=tai` in the environment of every instance. Show time is then the host's
  `CLOCK_TAI` from 2026-01-01, so two hosts whose system clocks agree give a frame the same show time.
- Synchronised system clocks (chrony or PTP). `clock.info` reports what the instance sees:
  `synchronized`, `est_error_us`, `tai_offset_s`, and `origin_ns` / `drift_ns` of the show clock.
- The same program rate, exactly: `30` or `30000/1001`, never `29.97` (that is 2997/100, another rate).
  The playout latency and delay must match too; the handshake below checks all three.
- The same source streams with their own timestamps. Under `AVP_CLOCK_ORIGIN=tai` the mixer's inputs
  keep them (`input_rec` `keep_source_ts`) instead of starting each stream at zero where this host
  joined it, which put the two hosts' pictures 7 to 11 frames apart.

## The handshake

Sync is per realtime team: a graph names the team with the realtime node's `team` parameter (the
mixer runs a source with audio or NDI in `input_<index>_av_sync`). Offsets travel in 1/120000 s,
a unit every show rate divides into whole ticks.

On the first host (the leader):

```text
realtime.team.sync <team> 30000/1001
→ 123456789 fps=30000/1001 latency_us=66733 delay=0
```

It fixes the offset the team found on arrival and prints it with the terms the other host must
match. On the second host, pass that whole line:

```text
realtime.team.sync <team> 30000/1001 123456789 fps=30000/1001 latency_us=66733 delay=0
```

A follower refuses before changing anything when a term differs. `realtime.team.set_delay` is refused
once the offset is fixed. A graph built from a script can carry the same line in every member's
realtime node: `"sync": {"fps": "30000/1001", "offset": 123456789, "latency_us": 66733, "delay": 0}`.

## Sources whose timestamps jump

A source that restarts or wraps its timestamps keeps its place: the jump is detected from the
source's own step (backwards, or far into the future; a lost-packet gap is not a jump), and its
video continues one frame step after the last frame before the jump, later by whole seconds of
show frames if the frames arrived later than that. The audio follows the video's correction.

```text
realtime.team.source_shift <team>                     → {"<source>": <shift>}  (1/120000 s)
realtime.team.source_shift <team> <source> <shift>    give a host that started after the jump the other host's
```

## Outputs and timed takes

- MPEG-TS outputs (SRT, UDP, `.ts`) are stamped on show time (`mux` `show_time`), with keyframes on
  the same show frames on both hosts (`enc_video` `show_keyframes`, NVENC `forced-idr`). MP4, FLV and
  the Janus RTP previews are unchanged.
- `mixer.cut` and `mixer.fade` take `"at_show_tick": N`: the scene reaches program no earlier than
  show tick N, the same frame on both hosts.

## Not covered yet

- Two real hosts with real clock skew, and cross-host drift over time.
- Browser sources in the two-mixer test.
- Whether NVENC's own keyframe interval realigns after a forced IDR.
- A restart correction is exact only when no real time passed; an encoder restart is best effort
  to whole seconds.
- Coordinated control: nothing sends the same takes and graphics to both mixers yet.
