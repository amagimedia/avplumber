# Host sync (experimental)

Two instances on different hosts release the same source frame in the same program tick and stamp
outputs on one show timeline, so a backup mixer airs the main one's picture frame for frame.

Merged in [#70](https://github.com/amagimedia/avplumber/pull/70). Tested only as two containers on one
L4 (one clock shifted 12345 s): identical pictures at every PTS at 30, 30000/1001 and 60000/1001 fps,
and timed cuts on the same tick 10/10. Not yet run on two real hosts.

## Requirements

- `AVP_CLOCK_ORIGIN=tai` on every instance, and clocks synchronised by chrony or PTP (`clock.info`
  reports `synchronized` and `est_error_us`).
- The same program rate, exactly (`30000/1001`, never `29.97`), playout latency and delay.
- The same source streams. Under `AVP_CLOCK_ORIGIN=tai` the mixer's inputs keep their own timestamps.

## Handshake

Per realtime team (a node's `team`; the mixer's sources with audio or NDI use `input_<index>_av_sync`).
Offsets are in 1/120000 s.

```text
leader:    realtime.team.sync <team> 30000/1001
           → 123456789 fps=30000/1001 latency_us=66733 delay=0
follower:  realtime.team.sync <team> 30000/1001 123456789 fps=30000/1001 latency_us=66733 delay=0
```

The follower refuses if a term differs. A script can set the same on each realtime node:
`"sync": {"fps": "30000/1001", "offset": 123456789}`.

## Also

- A source whose timestamps jump keeps its place; `realtime.team.source_shift <team> [<source> <shift>]`
  reads or sets the corrections.
- MPEG-TS outputs (SRT, UDP, `.ts`) use show time with keyframes on the same frames on both hosts.
- `mixer.cut` / `mixer.fade` with `"at_show_tick": N` reach program at the same show tick on both.

Open: real clock skew between hosts, browser sources, and sending the same takes to both mixers.
