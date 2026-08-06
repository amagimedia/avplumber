# Generic manual mixer

This demo builds one video-only 1080x1920 program from any positive number of
inputs. It exposes fullscreen scenes and paged 2/4/8/16-box views through the
generic mixer control protocol. There is no audio path, speaker detection,
automatic switching, face analysis, or source-specific policy.

Every supported geometry is created at startup. A `preheat_video_router` feeds
fixed CUDA scale/pad graphs for both mixer slots, so preview and program takes
do not rebuild filters. The two slot compositors use `cuda_rect_overlay` for
runtime rectangle placement. Media wipes additionally exercise FFmpeg's
`overlay_many_cuda` filter.

## Run

Build AVPlumber with CUDA, NVCC, and the Python module using the same FFmpeg
installation, then run:

```sh
LD_LIBRARY_PATH=/usr/local/lib python3 demos/mixer/mixer.py \
  --inputs <input-1> <input-2> [<input-N> ...] \
  --output <output-url-or-path> \
  --remote-control-port 7777
```

Use `--loop-inputs` for finite test files. Output format is inferred for RTMP,
SRT, `.flv`, `.ts`, `.mp4`, `.mkv`, and `.webm`; otherwise pass
`--output-format`.

Install Textual and start the separate control UI:

```sh
python3 -m pip install -r demos/mixer/requirements.txt
python3 demos/mixer/tui.py --host 127.0.0.1 --port 7777 \
  --wipe-file <optional-alpha-wipe.mov>
```

Selecting any scene first loads it into preview and waits for readiness. CUT,
FADE, and WIPE are not sent until that preheated preview slot reports ready.
Disconnected or incomplete inputs retain their assigned positions; they never
cause the remaining sources to reorder.

## Layouts

| View | Columns | Rows | Cell |
| --- | ---: | ---: | ---: |
| 2-box | 1 | 2 | 1080x960 |
| 4-box | 1 | 4 | 1080x480 |
| 8-box | 2 | 4 | 540x480 |
| 16-box | 2 | 8 | 540x240 |

Inputs keep their aspect ratio and are centered on black within each cell.
Additional inputs use consecutive pages. Empty cells and unavailable sources
remain black.

## Tests

Pure layout, graph-construction, and control tests do not require a GPU:

```sh
python3 -m pytest -q demos/mixer/tests
```

The runtime acceptance check requires the configured NVIDIA environment. It
must cover every layout, a disconnected input, preview plus cut/fade/wipe, and
the deterministic `overlay_many_cuda` matrix in `demos/cuda-overlay`.

