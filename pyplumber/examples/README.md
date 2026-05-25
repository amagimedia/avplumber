# pyplumber Examples

This directory contains Python programs that build AVPlumber graphs through the
`pyplumber` bindings. They range from minimal `PythonNode` examples to full
CUDA/TensorRT mixer applications.

The examples cover four broad topics:

- graph construction from Python;
- inserting Python code into packet, frame, audio, or metadata pipelines;
- CUDA, TensorRT, PyTorch, torchvision, and BlazeFace integration;
- mixer orchestration, automatic scene selection, audio VAD, and visual speech
  detection.

Most scripts are examples rather than turnkey products. Check the required build
flags, model files, and hardware path before using one as a starting point.

## Running

Run the scripts from the repository root unless a script says otherwise. That
keeps the local `pyplumber` package on Python's import path and makes the input
and output paths explicit:

```bash
AVP_INPUT=/path/to/input.mp4 \
AVP_OUTPUT=/tmp/output.ts \
python3 pyplumber/examples/simple-node.py
```

Common environment variables:

| Variable | Used by | Meaning |
| --- | --- | --- |
| `AVP_INPUT` | most single-input examples | Input URL or local media path. |
| `AVP_OUTPUT` | video output examples | Output URL or local media path. |
| `AVP_OUTPUT_FORMAT` | muxing examples | Muxer format, often `mpegts` or `flv`. |
| `AVP_USE_REALTIME` | CUDA examples | Set to `0` to skip realtime pacing. |
| `AVP_MODEL_DIR` | TensorRT face examples | Directory containing TensorRT engine and `data.yaml`. |
| `AVP_FACE_ENGINE` | face, visual speech, auto mixer | Explicit TensorRT face detection engine path. |
| `AVP_CLASS_NAMES` | face examples | Comma-separated model class labels when no `data.yaml` is available. |
| `AVP_SILERO_MODEL` | VAD, auto mixer | Optional local Silero VAD `.jit` model path. |
| `AVP_SILERO_REPO` | VAD, auto mixer | `torch.hub` repo or local checkout for Silero VAD. |
| `AVP_MODELS_DIR` | `tracker-live.py` | TensorRT model directory for the basketball tracker. |
| `AVP_BLAZEFACE_DIR` | `blazeface-node.py` | Directory for BlazeFace weights and anchors. |

Dependency groups:

- Basic graph and `PythonNode` examples need the `pyplumber` Python module and
  an AVPlumber build with the nodes they instantiate.
- PyTorch examples need `torch`; detection examples using torchvision also need
  `torchvision`.
- CUDA examples need a CUDA-capable AVPlumber build, compatible NVIDIA drivers,
  FFmpeg CUDA filters/codecs, and the matching Python CUDA stack.
- TensorRT examples need `cuda_infer_yolo` support, which means the native build
  must include CUDA, TensorRT, and NVCC support.
- VAD examples need the Python dependencies for Silero VAD. If no local
  `AVP_SILERO_MODEL` is supplied, the code can load through `torch.hub`.

## Which Example To Start From

| Topic | Script |
| --- | --- |
| Minimal `PythonNode` frame processing | `simple-node.py` |
| CPU PyTorch inference on decoded frames | `pytorch-node.py` |
| CUDA frame memory access from PyTorch | `pytorch-cuda-node.py` |
| torchvision object detection on CUDA frames | `torchvision-node.py` |
| BlazeFace detector implemented in Python | `blazeface-node.py` |
| TensorRT face-part detection and reframing | `face-detection.py` |
| Mouth ROI derivation and visual debug output | `mouth-roi-visualization.py` |
| Visual speech event extraction | `visual-speech-events.py` |
| Audio VAD event extraction | `vad-events.py` |
| Manual mixer control from Python | `manual_mixer.py` |
| Automatic multi-camera mixer | `auto_mixer.py` |
| Larger application graph with tracking | `tracker-live.py` |

## Basic `PythonNode` Pattern

Custom Python nodes subclass `PythonNode` and implement `process()`. A typical
node should:

1. Call `self._src.get()`. It blocks until a frame is available or the graph is
   closing; return if it gives back no frame.
2. Read or update `p.metadata` for downstream filters and drawing nodes.
3. Mutate frame planes only when the frame format and memory location match what
   the code expects.
4. Enqueue the frame to one or more destinations with `self._dst.enqueue(p)`.
5. Keep long-running setup, such as model loading, in `__init__`.

For GPU nodes, check `p.format.name`, expose CUDA memory to PyTorch with
`__cuda_array_interface__` only when the frame is actually CUDA-backed, and
synchronize the current CUDA stream before handing the frame back to downstream
AVPlumber nodes.

## Example Scripts

### `simple-node.py`

Minimal Python binding example. It decodes and rescales a frame, passes it
through `SimpleNode`, writes a few values into `p.metadata`, and forwards the
same frame to the rest of the graph. A downstream `FilterVideo` renders that
metadata with `drawtext`.

Use this script when learning the basic node contract:

```python
p = self._src.get()
if p:
    p.metadata["msg"] = f"Hello from python: {p}"
    self._dst.enqueue(p)
```

### `pytorch-node.py`

Runs a torchvision Faster R-CNN MobileNet detector on CPU-accessible video
frames. The node builds Torch tensors directly from AVFrame plane pointers with
`ctypes`, converts YUV planes to RGB, runs detection every few frames, draws
boxes into the frame planes, and writes metadata such as `person_count`,
`detection_time_ms`, and `pytorch_status`.

This is useful when Python owns the model code and AVPlumber owns decode,
timing, filtering, encoding, and I/O.

### `pytorch-cuda-node.py`

Shows CUDA frame access from PyTorch. The graph initializes a CUDA hardware
context, decodes into CUDA frames, uses `scale_cuda`, and wraps the luma plane
pointer with an object exposing `__cuda_array_interface__`. PyTorch applies a
small blur and Sobel edge pipeline on the GPU, then copies the result back into
the frame luma plane before NVENC encoding.

### `torchvision-node.py`

Runs a torchvision Faster R-CNN ResNet-50 FPN v2 detector against CUDA video
frames. It demonstrates a fuller object detection workflow than
`pytorch-cuda-node.py`: CUDA decode and scale, PyTorch tensor views over frame
planes, model inference, detection metadata, and hardware encoding.

Use this as the starting point when integrating a standard torchvision model
with the AVPlumber CUDA pipeline.

### `blazeface-node.py`

Implements a BlazeFace face detector in the example itself and runs it on CUDA
frames. The graph splits the stream:

- one branch stays at 1280x720 for encoding;
- one branch is scaled and padded to 128x128 for BlazeFace inference.

`BlazeFaceNode` downloads missing model assets into `models/blazeface` or
`AVP_BLAZEFACE_DIR`, runs face detection on the CUDA luma plane, serializes face
boxes and keypoints into `p.metadata["blazeface_faces"]`, and sends that
metadata through `JoinMetadata`. `DrawBBox` then renders face boxes and
keypoints on the main 720p stream before NVENC output.

Useful knobs:

- `AVP_BLAZEFACE_MIN_SCORE`: model score threshold. Defaults to `0.75`.
- `AVP_BLAZEFACE_NMS`: non-maximum suppression threshold. Defaults to `0.3`.

### `face-detection.py`

Builds a CUDA/TensorRT face-part graph around `CudaInferYolo`. The graph decodes
video on the GPU, scales a YOLO branch, joins detection metadata back onto the
full-resolution branch, tracks selected face-part labels, draws boxes and
labels, computes a smooth portrait viewport, crops with `CropMetadataCuda`, and
encodes the result with NVENC.

Configure the TensorRT model with `AVP_FACE_ENGINE` or `AVP_MODEL_DIR`. If the
model classes are not available in `data.yaml`, set `AVP_CLASS_NAMES`.

### `mouth-roi-visualization.py`

Runs the face-part detector and `FaceAnchoredMouthTrackerNode`, then draws the
mouth ROI metadata onto the output. It is a diagnostic graph for validating
mouth box extraction before feeding visual speech detection.

Useful variables include `AVP_VISUAL_TARGETS` for explicit target regions,
`AVP_MOUTH_TRACKER_ENABLED`, `AVP_MOUTH_ESTIMATE_ENABLED`, and
`AVP_BBOX_THICKNESS`.

### `visual-speech-events.py`

Runs CUDA face-part detection, derives mouth ROIs, and passes them through
`VisualSpeechGateNode`. It writes visual speech events to JSONL and a summary
JSON file rather than encoding a program output.

Use this when tuning lip-motion thresholds independently from the full auto
mixer. The main outputs are controlled by:

- `AVP_EVENTS_JSONL`: visual speech event log path.
- `AVP_SUMMARY_JSON`: final summary JSON path.
- `AVP_VISUAL_START_THRESHOLD` and `AVP_VISUAL_STOP_THRESHOLD`: event gates.
- `AVP_TIMEOUT_S`: max run time while waiting for the sink to finish.

### `vad-events.py`

Audio-only Silero VAD example. It demuxes and decodes the first audio stream,
resamples to 16 kHz mono float samples, runs `SileroVADNode`, and writes metadata
events through `MetadataJsonlSink`.

Useful variables:

- `AVP_INPUT`: input URL or path.
- `AVP_RUN_SECONDS`: run duration; `0` means run until interrupted.
- `AVP_VAD_JSONL`: optional JSONL output path.
- `AVP_VAD_THRESHOLD`, `AVP_VAD_NEG_THRESHOLD`, `AVP_VAD_MIN_EMIT_MS`,
  `AVP_VAD_MIN_SILENCE_MS`, `AVP_VAD_SPEECH_PAD_MS`: VAD tuning.

### `manual_mixer.py`

Minimal manual video mixer using `MixerGraphBuilder`. It decodes two looped MP4
inputs, registers fullscreen, picture-in-picture, and multiviewer scenes on a
1920x1080 canvas, then streams the encoded program to a file or RTMP URL.

Run it with:

```bash
python3 pyplumber/examples/manual_mixer.py \
  --input1 /path/to/a.mp4 \
  --input2 /path/to/b.mp4 \
  --output /tmp/manual-mix.ts
```

The script starts a small stdin REPL:

- `cut <scene>`: hard cut to a scene.
- `fade <scene> [secs]`: crossfade to a scene.
- `wipe <scene> <file>`: transition through a wipe media file.
- `scenes`: list registered scenes.
- `status`: print the current program scene.
- `quit`: stop the example.

### `auto_mixer.py`

Compatibility wrapper for `pyplumber.auto_mixer.cli`. This is the largest
example and is closer to an application than a minimal binding demo.

It builds a multi-input 1080x1920 program mixer for vertical output. Each input
is decoded once and fanned out into:

- an original 16:9 leg for picture-in-picture, multiviewer, and vertical stack
  layouts;
- a face-tracked 9:16 portrait leg for fullscreen and videoconference layouts;
- an audio leg for program output and Silero VAD;
- a visual speech leg that uses face-part detections, mouth ROIs, and
  lip-motion gates.

Automatic switching uses both speech signals. Silero VAD updates audio speaking
state, `VisualSpeechGateNode` updates visual speaking state, and the
`AutoSwitcher` changes scenes only when the configured speaker evidence is
active. Every portrait crop comes from the YOLO face-part model and tracks the
`Face` class. The mixer can use fixed scene families or a history-aware shot
profile that considers recent speaker changes and manual geometry hints.

Basic MPEG-TS file or RTMP/FLV output:

```bash
python3 pyplumber/examples/auto_mixer.py \
  --inputs /path/to/cam1.ts /path/to/cam2.ts \
  --output /tmp/auto-mix.ts \
  --face-engine /path/to/yolo_face.plan
```

With a control port and Web UI registration:

```bash
python3 pyplumber/examples/auto_mixer.py \
  --inputs /path/to/cam1.ts /path/to/cam2.ts /path/to/cam3.ts \
  --output rtmp://example.invalid/live/program \
  --face-engine /path/to/yolo_face.plan \
  --remote-control-port 22422 \
  --webui-api http://localhost:22222 \
  --instance-name auto-mixer \
  --logfile /tmp/auto-mixer.log
```

For WebRTC delivery through Janus RTP, use `--janus-output` instead of, or in
addition to, `--output`. `--janus-preview` adds a Janus copy alongside a regular
FLV or MPEG-TS output. Record outputs are intentionally limited to RTMP/FLV and
MPEG-TS, including SRT transport.

Important options:

| Option | Meaning |
| --- | --- |
| `--inputs` | Two or more input URLs or paths. |
| `--output` | RTMP/FLV, SRT, or MPEG-TS output. Optional when Janus output is enabled. |
| `--output-format` | Explicit recording muxer: `mpegts` or `flv`. |
| `--face-engine` | TensorRT face detector engine. Defaults can come from `AVP_FACE_ENGINE` or `AVP_MODEL_DIR`. |
| `--input-start-ts` | Seek all inputs to a common start timestamp. |
| `--silero-model`, `--silero-device`, `--silero-threshold` | Audio VAD model and tuning. |
| `--program-audio-input` | Input index used for program audio. Defaults to `0`. |
| `--auto-switch-layout` | Fixed auto switch family: `videoconf` or `full_face`. |
| `--auto-switch-shot-profile` | `fixed`, `conference`, `balanced`, or `variety`. |
| `--auto-switch-transition` | `cut`, `fade`, or `wipe` for automatic switches. |
| `--input-policy` | JSON per-input auto-switch policy. May be repeated. |
| `--talkshow-profile` | Enable the index-based default talk-show policy. |
| `--min-dwell` | Minimum program dwell time before another automatic switch. |
| `--switch-pts-lead-ms` | Schedule cuts ahead of VAD event PTS; negative disables PTS scheduling. |
| `--remote-control-port` | Enable the AVPlumber text control API. |
| `--webui-api`, `--instance-name`, `--logfile` | Register this instance with the Web UI. |
| `--janus-output`, `--janus-preview` | Enable Janus RTP output paths. |
| `--disable-preheated-scenes` | Use dynamic mixer scene loading instead of preheated geometry routes. |

Scene families include:

- `full_face_<i>`: speaker `i` as a face-tracked portrait filling the canvas.
- `videoconf_<i>`: speaker `i` in a top square crop with other cameras below.
- `pip_<i>_<j>`: speaker portrait with a landscape thumbnail.
- `vstack_<a>_<b>` and `vstack3_<a>_<b>_<c>`: two or three landscape sources
  stacked vertically.
- `multiviewer`: 2x2 grid of face portraits when at least three inputs exist.

By default the auto mixer uses preheated scene geometries, where common filter
outputs are kept hot and routed into mixer slots. That reduces transition-time
filter setup work at the cost of a larger standing graph. Use
`--disable-preheated-scenes` when debugging dynamic scene loading or when the
extra hot paths are not desirable.

Auto-switch policies are defined per input index. A policy contains one or more
rules; rules are alternatives, and `operator` controls whether audio VAD and
visual confirmation are combined as `and` or `or` inside a rule. `lead_db` adds a
measured-level lead requirement for that rule, `priority` lets a qualified rule
win before comparing loudness, and `require_level=false` allows visual-only or
VAD-only rules to ignore the minimum active audio level gate.

For example, this input can switch either when audio and visual confirmation are
both active, or when audio VAD is active with a 4.5 dB lead:

```bash
--input-policy '{
  "index": 2,
  "rules": [
    {"require_audio": true, "require_visual": true, "operator": "and"},
    {"require_audio": true, "require_visual": false, "lead_db": 4.5}
  ]
}'
```

The optional `--talkshow-profile` keeps the current talk-show defaults behind an
explicit flag:

- input `0`: program audio by default; auto-switching requires audio VAD and
  visual confirmation, plus a 3 dB lead over competing candidates;
- input `1`: auto-switching may use audio VAD alone with high priority and no
  minimum-level gate;
- all other inputs: default audio VAD and visual confirmation.

Override those conventions with `--program-audio-input` or `--input-policy`.
They are mixer application policy choices, not requirements of
`MixerGraphBuilder` itself.

### `tracker-live.py`

Larger end-to-end basketball tracking graph. It combines CUDA decode, TensorRT
YOLO inference nodes, metadata joins, ball/player tracking nodes, drawing nodes,
shot classification, smooth crop viewport selection, and NVENC output. It is
less of a minimal Python binding example and more of a complete application
pipeline showing how Python orchestration can assemble AVPlumber's built-in
tracking and inference nodes.

Set `AVP_MODELS_DIR` to the directory containing the expected TensorRT `.plan`
files before running it.

## Notes On Extending These Examples

- Keep graph construction explicit until a pattern repeats across examples.
- Keep CPU and GPU memory paths separate. Do not insert `hwdownload` or
  `hwupload_cuda` between CUDA decode, CUDA filters, TensorRT, drawing, and
  NVENC unless the example is intentionally about CPU/GPU interop.
- Use metadata keys consistently when joining inference branches back onto the
  main video branch.
- For live sources, consider queue capacity, realtime pacing, restart groups,
  and output timestamp behavior as part of the example, not as afterthoughts.
