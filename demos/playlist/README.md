# Playlist demo

[![Watch the playlist: program output beside the TUI](https://github.com/amagimedia/avplumber/releases/download/playlist-demo-media-2026-09/playlist-demo.jpg)](https://amagimedia.github.io/avplumber/demos/playlist/docs/)

[Watch the demo](https://amagimedia.github.io/avplumber/demos/playlist/docs/) · [MP4](https://github.com/amagimedia/avplumber/releases/download/playlist-demo-media-2026-09/playlist-demo.mp4) · [Full processing graph](https://amagimedia.github.io/avplumber/demos/graph.html?demo=playlist) · [Reference](docs/guide.md)

Clip playout on AVPlumber's native mixer engine. Every element is a mixer
source with its decoder kept resident; each transition is a `mixer.cut`,
`fade` or `wipe` armed ahead of time at a wallclock start, so the switch between
two clips is decided inside the engine, not by Python after the fact. Program
output is on the left and the real terminal controls on the right, in a
26-second, 2.3 MB, 1600×900, 30 fps MP4: a manual take, two scheduled cuts landing on the
configured cue frames, and a fade. Click the first-frame preview to play it
with chapter buttons and inspect the web UI graph. The page is hosted on GitHub
Pages; media comes from public release assets, outside Git history.

## Frame continuity, measured

Every generated clip carries a machine-readable frame code. A recorded program
is decoded frame by frame and each element boundary is checked against the
configured cue-in and cue-out frames (`tests/verify_recording.py`). The
measurements published on the [demo page](https://amagimedia.github.io/avplumber/demos/playlist/docs/)
come from a 50 s LoopAll pass on a Tesla T4 with the five fixtures:

- Cue points inside a file land exactly: element 3 (cued 2.000 s to 8.000 s)
  goes on air on frame 60 and leaves on frame 239.
- Elements cued at 0 land on frame 0 or frame 1. The mixer's ready cut fires
  on the first fresh incoming frame and switches on the next one; the frame it
  consumes has to come from the loop wrap for a cue at 0, and the realtime
  resync at the wrap decides the race. This ±1 frame is the remaining known
  offset.
- No black or unreadable frames at any boundary in 1,509 recorded frames. One
  repeated and one skipped frame occur about 0.8 s after each transition.

Every anomaly and its position is listed in the published report; nothing is
claimed beyond what the verifier prints.

## Run

Use a Linux NVIDIA host with hardware decode and NVENC and follow the
[shared Docker/NVIDIA setup](../README.md). From the repository root:

```sh
docker build -f demos/mixer/Dockerfile -t avplumber-mixer:local .
docker build --build-arg AVP_BASE_IMAGE=avplumber-mixer:local \
    --tag avplumber-playlist:local demos/playlist
docker run --rm --gpus all --network host avplumber-playlist:local \
    --janus-host 127.0.0.1 --janus-video-port 5004 --control-port 7778
```

Open the output at <http://127.0.0.1:8080>. In another terminal:

```sh
python3 -m venv .venv-tui
.venv-tui/bin/python -m pip install -r demos/playlist/requirements.txt
.venv-tui/bin/python demos/playlist/player.py --host 127.0.0.1 --port 7778
```

`player.py --dry-run` runs the same TUI against an in-memory backend without a
GPU. Everything else, including options, keyboard controls, Docker details,
tests, the verification procedure and implementation notes, is in the
[reference](docs/guide.md).

<img src="docs/tui.svg" alt="The playlist TUI: element table, on-air panel with the countdown to the next scheduled cut, and two action bars" width="100%">
