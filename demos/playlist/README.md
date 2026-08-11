# Playlist player demo

A single, runtime-editable **playlist** of video clips played through one
output to a Janus H.264 RTP mountpoint. It is a *simulator/tester*: its job is
to exercise AVPlumber's playlist-relevant nodes (`source_switcher`, `pause`,
`realtime`, `force_fps`, `speed_video`, `rescale_video`, `input_rec`) hard
enough to surface backend bugs. It is built **entirely on existing nodes** --
there are no C++ changes -- and reuses the `demos/replay` decode/encode chain.

## What it does

- One playlist, **hard cuts** between clips.
- Two-level playback model:
  - **Playlist mode** (whole list): `PlayAll`, `PlayCurrent`, `LoopAll`,
    `LoopCurrent`.
  - **Element mode** (per clip): `PlayToEnd`, `Timed` (fixed duration),
    `LoopSelf`.
- Runtime **add / remove / reorder / enable-disable / retime / respeed** of
  clips.
- Mixed fps and mixed resolution inputs, normalised to one output fps and
  resolution (`force_fps` + `rescale_video`).
- **Gapless and never-black** is a hard requirement: outside a deliberate fade
  the output never shows a black frame. Frame-accuracy is best-effort
  (≈±1 frame) and measured by tests, not asserted.

Fades are a Phase-2 nice-to-have (reuse `MixerGraphBuilder`); the PoC does hard
cuts only.

## How it stays gapless and never-black

Each playlist slot is served by a **worker** = one copy of the replay decode
chain:

```
input_rec -> demux -> dec_video -> speed_video -> rescale_video -> force_fps -> pause  -> worker{i}_out
```

Two (or more) workers feed a single `source_switcher`, whose output is reclocked
by **one** shared `realtime<av::VideoFrame>` node (the common output clock) and
sent to the replay Janus output group (`force_keyframe -> enc_video (h264_nvenc)
-> bsf -> mux -> output`).

- The **incoming** clip is decoded ahead of time on the idle worker and held
  frozen by its `pause` node (`pause <team> now` at build; the node is added
  `paused=true`). A frozen worker sits on a real decoded frame, so it is never
  black.
- A cut is **`resume <team>` first, then flip `source_switcher active`**.
  Releasing the incoming worker before selecting it guarantees the switcher
  always has a produced frame for the newly-active input -- no gap.
- The `source_switcher` never synthesises black. We never configure a `fallback`
  and never use the `sentinel` node (a regression test enforces this).

Rebuilding a worker onto a new clip **deletes its nodes first**
(`group.stop` + `node.delete`) before re-adding, because `node.add` rejects a
name that already exists ("Name busy"). `input_rec` opens its URL at creation
and cannot be re-pointed at runtime, so a fresh clip means a fresh chain.

## Layout

| file | role |
|------|------|
| `playlist.py` | Pure logic: modes, `resolve_next`, edits, the graph *plan* (`NodeSpec` + emitted command strings), and `PlaylistController`. No live backend, no GPU. |
| `playlist_app.py` | Live glue: `build_playlist_application` instantiates AVPlumber, builds the worker/switcher/output graph, and wires the controller's command sink. GPU host only. |
| `player.py` | Textual TUI (playlist table + transport/mode/edit rows + add modal). `--dry-run` prints the command stream; `--no-tui` runs a smoke exercise. |
| `tests/` | The regression harness (see below). |

## Run

Requires the same NVIDIA + custom-FFmpeg environment as `demos/replay`
(CUDA decode, `h264_nvenc`). Install the TUI dependency:

```sh
python3 -m pip install -r demos/playlist/requirements.txt
```

Configure a video-only Janus Streaming mountpoint for H.264 RTP (defaults:
host `127.0.0.1`, video port `5004`, RTCP `5005`, payload type `96`,
SSRC `0x41565001`, `4000 kbit/s` CBR), then:

```sh
python3 demos/playlist/player.py --janus-host 127.0.0.1 --janus-video-port 5004
```

Inspect the exact AVP command stream a session would emit, without a GPU:

```sh
python3 demos/playlist/player.py --dry-run --no-tui
```

TUI keys: `space` play/pause, `n`/`p` next/prev, `a` add, `Del` remove,
`m` cycle element mode, `q` quit. Buttons cover mode, reorder, enable/disable
and play-selected.

## Regression harness

The controller drives the backend only through an injected `command(str)` sink,
so every feature is asserted against the exact command stream -- no GPU needed.
This doubles as an AVP playlist regression suite.

```sh
cd demos/playlist && python3 -m pytest tests/ -q
```

- `test_modes.py` -- `resolve_next` truth table, loop-self hold, disabled skip,
  prev wrap, mode flags, `Timed` requires duration.
- `test_graph.py` -- worker chain shape/edges, normalisation, preseek, output
  switcher+realtime, never-black detectors, cut ordering.
- `test_controller.py` -- transport, preload, mode changes, add/insert/remove
  (incl. remove-current and last-clip guards), reorder, disable, retime.
- `test_rebuild.py` -- `FakeBackend` enforces the "Name busy" rule: every
  rebuild must delete before re-adding.
- `test_app_build.py` -- the live builder against a fake AVPlumber (wiring +
  start/stop lifecycle).
- `test_no_black_frame.py` / `test_gapless.py` / `test_frame_perfect.py` --
  the never-black mechanism, preload arming, and frame-boundary carry.

Pixel-level never-black / frame-accuracy is a bench check on the GPU/Janus host,
not part of the unit suite.
