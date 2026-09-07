# Playlist demo

<img src="docs/tui.svg" alt="The playlist TUI: element table, on-air panel with the countdown to the next scheduled cut, and two action bars" width="100%">

A clip playlist on AVPlumber's native mixer engine. Every element is a mixer
source with its decoder kept resident; transitions are `mixer.cut`, `fade` or
`wipe` commands armed ahead of time at a wallclock `start_pts_ms`, so the cut
between two clips is decided inside the engine on the frame, not by Python
after the fact. The program goes out as one video-only H.264 RTP stream to a
Janus Streaming mountpoint.

## Features

- Playlist transport: Play, Pause, Stop, Prev, Next; four playlist modes
  (`PlayAll`, `PlayCurrent`, `LoopAll`, `LoopCurrent`).
- Per element: Take, Pause, Stop, Edit (name, path, cue-in, cue-out, length,
  speed), End mode (`Play` to cue-out, `Timed`, `Loop`), on/off, reorder,
  add, remove. Up to sixteen elements.
- One playlist-wide transition: Cut, Fade or media Wipe with a duration. A
  manual action during a transition interrupts it, as in the mixer demo.
- Scheduled transitions with a live countdown; elements that leave air are
  parked on their cue-in frame so the next take is warm.
- Backend and TUI are separate processes. The TUI attaches over AVPlumber's
  control port and can run anywhere, including in a browser through `ttyd`.
- `--dry-run` TUI with an in-memory backend, no GPU needed.

### How an element ends

| End | The element completes when |
| --- | --- |
| `Play` | its cue-out (or the media end) is reached on the wallclock |
| `Timed` | its configured length has elapsed; the media loops underneath |
| `Loop` | never by itself; leave it with Next, Prev or Take |

When an element goes on air the controller computes its end time, loads the
next element into the mixer's preview slot, arms the transition to end exactly
then, and schedules a native `resume` of the incoming chain a few milliseconds
earlier (`--preroll-ms`). Every chain loops between its cue points so no
decoder ever reaches EOF; the schedule alone decides when a clip ends.

### Keyboard

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

## Preview without a GPU

```sh
python3 -m pip install -r demos/playlist/requirements.txt
python3 demos/playlist/player.py --dry-run
```

The controls work against an in-memory backend; scheduled transitions fire on
the wallclock, but no video is decoded or sent.

## Run with video output

Requirements are the mixer demo's: an NVIDIA host, `pyplumber` built with
CUDA and NVCC, FFmpeg with CUDA decoding and `h264_nvenc`, the `avpmixer`
package on `PYTHONPATH`, and a video-only Janus Streaming mountpoint accepting
H.264 RTP. Follow the [NVIDIA setup and Janus preview guide](../README.md).

Generate the five 1080p30 fixtures (each carries a machine-readable frame code,
see below):

```sh
demos/playlist/test-media/generate.sh
```

Start the backend, then attach the TUI:

```sh
python3 demos/playlist/server.py --janus-host 127.0.0.1 --janus-video-port 5004 \
    --control-port 7778 --log-file playlist-demo.log
python3 demos/playlist/player.py --port 7778
```

Useful backend options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--mode`, `--transition`, `--transition-ms` | `LoopAll`, `Cut`, 500 | initial playlist settings |
| `--wipe-file` | none | alpha media for Wipe transitions |
| `--playlist FILE` | five fixtures | JSON list of elements (`url`, `name`, `cue_in`, `cue_out`, `duration`, `speed`, `end`) |
| `--preroll-ms` | 50 | how early the incoming chain is resumed before its cut |
| `--switch-margin-ms` | 100 | mixer's minimum lead for a scheduled cut |
| `--record FILE.mp4` | none | also write the program for `tests/verify_recording.py` |

`--janus-*` options match the mixer demo. Janus receives RTCP on the port
after the video port; PLI/FIR requests force a keyframe.

### Docker

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

## Verifying frame-perfect transitions

Each fixture carries a 32-bit code strip in its top-left corner: clip number,
frame number and a checksum, with an inverted second row that turns blended or
scaled frames unreadable. Record a run with `--record`, then:

```sh
python3 demos/playlist/tests/verify_recording.py program.mp4 \
    --playlist playlist.json --fps 30 [--transition-frames 15]
```

The report lists every element boundary as `last frame of A -> first frame of
B`, counts repeated and skipped frames inside elements, and checks that each
boundary lands on the configured cue-in and cue-out frames. `PASS` requires
zero repeats, zero gaps and exact boundaries; `--transition-frames` allows that
many blended frames at a Fade or Wipe. The verifier itself is checked against
synthetic splices with an injected repeat, a late cue-in and a fade.

`regression.py --port 7778` drives a running backend through every verb and
mode over the control port, including one scheduled advance. `regression.py
--local` runs the same checks against the in-memory backend.

## Tests

```sh
python3 -m pytest -q demos/playlist/tests
```

Policy, controller scheduling, JSON protocol, the engine's native command
sequences against a fake AVPlumber, the rendered TUI, the fixture contract and
the container contract. Textual 8.2.8 is needed for the TUI tests.

## Implementation notes

`playlist.py` is the policy: modes, navigation, the schedule. `engine.py`
binds elements to sixteen fixed mixer sources (`pl_item_<slot>` groups,
`item_<slot>` fullscreen scenes) built with `avpmixer.inputs.build_input`, the
same decode chain the mixer demo uses, plus a pause team and a speed node.
`server.py` registers `playlist.*` commands on the control server and runs the
policy loop; `control.py` is the JSON protocol shared with `player.py`.

The native `resume <team> at <wallclock_ms>` command was added for this demo:
the pause team resumes itself when the host clock reaches the deadline, so the
incoming clip's first frame arrives for the armed cut without Python on the
path. An explicit pause cancels a scheduled resume.

The playlist is video-only and has no audio, no software-decoding fallback
and no per-element transition override.
