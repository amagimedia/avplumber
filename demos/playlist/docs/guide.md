# Playlist reference

Clip playout on AVPlumber's native mixer engine. Every element is a mixer
source with its decoder kept resident; each transition is a `mixer.cut`,
`fade` or `wipe` armed ahead of time at a wallclock `start_pts_ms`, and the
switch between two clips is decided inside the engine, frame by frame. The
program goes out as one video-only H.264 RTP stream to a Janus Streaming
mountpoint. See [the README](../README.md) for the recording and the short
version.

## Features

- Playlist transport: Play, Pause, Stop, Prev, Next; four playlist modes
  (`PlayAll`, `PlayCurrent`, `LoopAll`, `LoopCurrent`).
- Per element: Take, Pause, Stop, Edit (name, path, cue-in, cue-out, length,
  speed), End mode (`Play` to cue-out, `Timed`, `Loop`), on/off, reorder, add,
  remove. Up to sixteen elements.
- One playlist-wide transition: Cut, Fade or media Wipe with a duration. A
  manual action during a transition interrupts it, as in the mixer demo.
- Scheduled transitions with a live countdown in the TUI. Elements that leave
  air are parked on their cue-in frame with the decoder resident, so every
  take is warm.
- Backend and TUI are separate processes. The TUI attaches over AVPlumber's
  control port and runs anywhere, including in a browser through `ttyd`.
- `--dry-run` TUI against an in-memory backend, no GPU needed.

### How an element ends

| End | The element completes when |
| --- | --- |
| `Play` | its cue-out (or the media end) is reached on the wallclock |
| `Timed` | its configured length has elapsed; the media loops underneath |
| `Loop` | never by itself; leave it with Next, Prev or Take |

Playlist and End modes are independent. `PlayCurrent` stops after the current
element; `LoopCurrent` repeats it. Manual navigation always escapes a repeating
element. Disabled elements are skipped.

### Keyboard controls

The TUI has two action bars, one for the playlist and one for the selected
element, and the same actions on keys:

| Key | Action |
| --- | --- |
| Space | Playlist play/pause |
| `s` | Playlist stop |
| `p` / `n` | Previous / next |
| ↑ ↓ | Select an element |
| Enter | Take the selected element |
| `u` / `x` | Pause / stop the selected element |
| `e` | Edit the selected element |
| `m` | Cycle its End mode |
| `o` | Enable / disable it |
| `a` / Delete | Add / remove |
| `q` | Quit the TUI |

## Requirements

The same as the mixer demo: a Linux NVIDIA host with hardware decode and
NVENC, `pyplumber` built with CUDA and NVCC, FFmpeg with CUDA decoding and
`h264_nvenc`, the `avpmixer` package on `PYTHONPATH`, and a video-only Janus
Streaming mountpoint that accepts H.264 RTP (the shared Janus preview from the
[demo setup guide](../../README.md) provides one on port 5004). Neural models
and TensorRT are not needed.

## Start the backend

Generate the five 1080p30 fixtures once:

```sh
demos/playlist/test-media/generate.sh
```

Each clip is 1920×1080, 30 fps, 300 frames, no B-frames, with the clip label,
frame number and clock burned in, plus a machine-readable frame code (see
[Verifying frame continuity](#verifying-frame-continuity)).

```sh
python3 demos/playlist/server.py --janus-host 127.0.0.1 --janus-video-port 5004 \
    --control-port 7778 --log-file playlist-demo.log
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--mode`, `--transition`, `--transition-ms` | `LoopAll`, `Cut`, 500 | initial playlist settings |
| `--wipe-file` | none | alpha media for Wipe transitions |
| `--playlist FILE` | five fixtures | JSON list of elements (`url`, `name`, `cue_in`, `cue_out`, `duration`, `speed`, `end`) |
| `--media-dir` | `test-media/` | where the five fixture names are looked up |
| `--resume-offset-ms` | 8 | how long after its scheduled cut the incoming chain is resumed |
| `--switch-margin-ms` | 100 | the mixer's minimum lead for a scheduled cut |
| `--record FILE.ts` | none | also write the program for `tests/verify_recording.py` |
| `--webui-url URL` | none | register with an AVPlumber web UI, e.g. `http://127.0.0.1:22222` |
| `--control-port` | 7778 | AVPlumber control server (the TUI and `regression.py` connect here) |

`--janus-*` options match the mixer demo. Janus receives RTCP on the port after
the video port; PLI/FIR requests force a keyframe. The default playlist is the
five fixtures with element 3 cued 2 s..8 s and element 4 Timed 4 s at 2× speed.

## Start the TUI

```sh
python3 -m pip install -r demos/playlist/requirements.txt
python3 demos/playlist/player.py --host 127.0.0.1 --port 7778
```

`python3 demos/playlist/player.py --dry-run` runs the same TUI against an
in-memory backend. For a browser terminal, as in the recording:

```sh
ttyd -p 7681 -t fontSize=15 python3 demos/playlist/player.py --port 7778
```

## Docker

```sh
docker build -f demos/mixer/Dockerfile -t avplumber-mixer:local .
docker build --build-arg AVP_BASE_IMAGE=avplumber-mixer:local \
    --tag avplumber-playlist:local demos/playlist
docker run --rm --gpus all --network host avplumber-playlist:local
docker run --rm -it --network host --entrypoint python3 avplumber-playlist:local player.py
```

Set `AVP_BASE_IMAGE=<cuda-python-avplumber-image>` to any image that provides
`pyplumber` and `avpmixer`. The image generates and validates the fixtures
while building.

## Verifying frame continuity

Each fixture carries a 32-bit code strip at (64,0): clip number, frame number
and a checksum, with an inverted second row that turns blended or scaled frames
unreadable. Record a run with `--record`, then:

```sh
python3 demos/playlist/tests/verify_recording.py program.ts \
    --playlist playlist.json --fps 30 [--transition-frames 15] [--json report.json]
```

The report lists every element boundary as `last frame of A -> first frame of
B`, counts repeated and skipped frames inside elements, records where each
anomaly sits, and checks that each boundary lands on the configured cue-in and
cue-out frames. `PASS` requires zero repeats, zero gaps and exact boundaries;
`--transition-frames` allows that many blended frames at a Fade or Wipe. The
verifier is itself checked against synthetic splices: a clean splice passes; an
injected repeated frame, a one-frame-late cue-in and a half-second fade are
each reported.

`regression.py --port 7778` drives a running backend through every verb and
mode over the control port, including one scheduled advance; `regression.py
--local` runs the same checks against the in-memory backend.

## Tests

```sh
python3 -m pytest -q demos/playlist/tests
```

Policy, controller scheduling, JSON protocol, the engine's native command
sequences against a fake AVPlumber, the rendered TUI (Textual 8.2.8), the
fixture contract and the container contract.

## Implementation notes

`playlist.py` is the policy: modes, navigation, and the schedule. When an
element goes on air the controller computes its end time from cue points and
speed, arms the next element, and re-arms whenever the playlist, mode or
transition changes.

`engine.py` binds elements to sixteen fixed mixer sources: group
`pl_item_<slot>`, scene `item_<slot>` (fullscreen), decode chain from
`avpmixer.inputs.build_input` with the replay demo's additions: a pause team
on the input, a `pause` node and a realtime sync team, `h264_cuvid` with
`flush_magic` and low-delay decoding so a seek lands on the exact frame. Every
chain loops between its cue points and never reaches EOF; the schedule alone
ends an element.

A scheduled cut is armed 600 ms before its start (`arm_lead_ms`): `mixer.cut`
with `start_pts_ms`, plus the switcher's `active` key written into the mixer
timeline at the same time, so the switch is decided by frame timestamp rather
than by the polling order of the mixer's ready-cut task. The incoming chain is
resumed natively (`resume <team> at`) a few milliseconds after the cut time,
so its cue-in frame is the first frame stamped past the cut; the mixer's playout
buffer absorbs the lag. The switch is confirmed from `mixer.status` over the
local control port before the outgoing element is parked. Fades and wipes are
armed the same way and end at the scheduled time.

Two native additions were made for this demo: `resume <team> at <ms>` on the
pause team (an explicit pause cancels a scheduled resume) and a public
`mixer.interrupt` command that drops an armed or running transition. A decoder
fix, needed for exact seeks, makes stale frames from the NVDEC queue unable to
clear the post-seek discard target.

`control.py` is the JSON protocol shared by `server.py` (which registers
`playlist.status` and `playlist.<verb>` on the control server) and
`player.py`. The playlist is video-only, has no software-decoding fallback and
no per-element transition override.

## Demo recording

The published recording combines the program written by `--record` with a
separate Electron capture of the real TUI served by `ttyd`, composed to
1600×900 at 30 fps (`tests/capture_tui.cjs`, `tests/record_demo.py --short`,
`tests/compose_recording.sh`). The program is never resampled in time, so every
program frame appears exactly once in the composite. The processing graph
captures come from the AVPlumber web UI through `tests/capture_graph.cjs`,
grouped overview and full native graph. Media and images are public release
assets; [media.sha256](media.sha256) records their checksums.
