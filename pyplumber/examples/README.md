# pyplumber Python Binding Examples

This directory contains sample Python programs that build AVPlumber graphs through
the `pyplumber` bindings and insert Python code into a live media pipeline. The
examples show how to receive decoded video frames in a `PythonNode`, inspect or
modify frame metadata, call external libraries such as PyTorch and torchvision,
and then send the frame back to the AVPlumber graph for filtering, encoding, and
output.

Most examples use this shape:

1. `InputRec` reads `AVP_INPUT` or `input.mp4`.
2. `Demux` and `DecVideo` extract and decode video packets.
3. A scaler or filter prepares frames for Python.
4. A custom `PythonNode` pulls frames with `self._src.get()`.
5. The node writes metadata, mutates pixels, or emits detection metadata.
6. The graph encodes and writes `AVP_OUTPUT` or `output.mp4`.

## Running

Run the scripts from this directory or make sure the relative `sys.path` entries
still point at the local `pyplumber` package.

```bash
cd avplumber/pyplumber/examples
AVP_INPUT=/path/to/input.mp4 AVP_OUTPUT=/tmp/output.ts python3 simple-node.py
```

Common environment variables:

- `AVP_INPUT`: input URL or local media path. Defaults to `input.mp4`.
- `AVP_OUTPUT`: output URL or local media path. Defaults to `output.mp4`.
- `AVP_OUTPUT_FORMAT`: muxer format. Most examples default to `mpegts`.
- `AVP_USE_REALTIME`: set to `0` in CUDA examples to skip realtime pacing.
- `AVP_BLAZEFACE_DIR`: directory for BlazeFace weights and anchors.
- `AVP_MODELS_DIR`: TensorRT model directory for `tracker-live.py`.

The PyTorch examples require `torch`; detection examples also require
`torchvision`, and `blazeface-node.py` requires `numpy`. CUDA examples require a
CUDA-enabled PyTorch build, compatible NVIDIA drivers, FFmpeg CUDA filters/codecs,
and AVPlumber CUDA hardware acceleration support.

## Example Scripts

### `simple-node.py`

Minimal Python binding example. It decodes and rescales a frame, passes it
through `SimpleNode`, writes a few values into `p.metadata`, and forwards the
same frame to the rest of the graph. The downstream `FilterVideo` uses
`drawtext` to render metadata on the video.

Use this script when learning the basic `PythonNode` contract:

```python
p = self._src.get()
if p:
    p.metadata["msg"] = f"Hello from python: {p}"
    self._dst.enqueue(p)
```

### `pytorch-node.py`

Runs a torchvision Faster R-CNN MobileNet person detector on CPU-accessible video
frames. The node builds Torch tensors directly from AVFrame plane pointers with
`ctypes`, converts YUV planes to RGB, runs detection every few frames, draws
boxes into the frame planes, and writes metadata such as `person_count`,
`detection_time_ms`, and `pytorch_status`.

This is useful for integrations where Python owns the model code and AVPlumber
owns decode, timing, filtering, encoding, and I/O.

### `pytorch-cuda-node.py`

Shows zero-copy-style access to CUDA frame memory from PyTorch. The graph
initializes a CUDA hardware context with:

```python
avp.executeCommandsFromString('hwaccel.init { "name": "@gpu", "type": "cuda" }')
```

The decoder requests CUDA frames, `scale_cuda` prepares a 1280x720 CUDA frame,
and `PytorchCUDANode` wraps the luma plane pointer with an object exposing
`__cuda_array_interface__`. PyTorch then applies a small blur and Sobel edge
pipeline on the GPU and copies the result back into the frame luma plane before
NVENC encoding.

### `torchvision-node.py`

Runs a torchvision Faster R-CNN ResNet-50 FPN v2 detector against CUDA video
frames. It demonstrates a fuller object detection workflow than
`pytorch-cuda-node.py`: CUDA decode and scale, PyTorch tensor views over frame
planes, model inference, detection metadata, and forwarding frames for hardware
encoding.

Use this as the starting point when integrating a standard torchvision model
with the AVPlumber CUDA pipeline.

### `blazeface-node.py`

Implements a BlazeFace face detector in the example itself and runs it on CUDA
frames. The graph splits the stream:

- one branch stays at 1280x720 for encoding;
- one branch is scaled and padded to 128x128 for BlazeFace inference.

`BlazeFaceNode` downloads missing model assets into `models/blazeface` or
`AVP_BLAZEFACE_DIR`, runs face detection on the CUDA luma plane, serializes face
boxes and keypoints into `p.metadata["blazeface_faces"]`, and sends that metadata
through `JoinMetadata`. `DrawBBox` then renders face boxes and keypoints on the
main 720p stream before NVENC output.

Useful knobs:

- `AVP_BLAZEFACE_MIN_SCORE`: model score threshold. Defaults to `0.75`.
- `AVP_BLAZEFACE_NMS`: non-maximum suppression threshold. Defaults to `0.3`.

### `tracker-live.py`

Larger end-to-end basketball tracking graph. It combines CUDA decode, TensorRT
YOLO inference nodes, metadata joins, ball/player tracking nodes, drawing nodes,
shot classification, smooth crop viewport selection, and NVENC output. It is
less of a minimal Python binding example and more of a complete application
pipeline showing how Python orchestration can assemble AVPlumber's built-in
tracking and inference nodes.

Set `AVP_MODELS_DIR` to the directory containing the expected TensorRT `.plan`
files before running it.

### `mediapipe-face-landmarker.py`

End-to-end GPU face mesh pipeline using the `mediapipe_face_mesh_gpu` C++ node and a Python
`FaceMeshOverlayNode` that draws landmark dots zero-copy on the NV12 luma plane via CuPy.

Requires avplumber built with `HAVE_MEDIAPIPE=1 HAVE_GL=1 HAVE_CUDA=1` (use `Dockerfile.mediapipe`).
At runtime `libavp_mediapipe_face_mesh.so` and `libnvrtc.so` must be on `LD_LIBRARY_PATH`.

Pipeline:
```
InputRec → Demux → DecVideo(cuvid) → Realtime → FilterVideo(scale_cuda)
  → Split → CudaToEglImage → MediaPipeFaceMeshGpu(face_md)
           ↘ FaceMeshOverlayNode(reads face_md, draws dots) → AssumeVideoFormat
  → EncVideo(h264_nvenc) → Mux → Output
```

Environment variables (in addition to the common ones):

| Variable | Default | Description |
|---|---|---|
| `AVP_FACE_DOT_RADIUS` | `2` | Landmark dot radius in pixels |
| `AVP_FACE_DOT_LUMA` | `235` | Luma brightness of dots (0–255) |
| `AVP_FACE_MAX_FACES` | `4` | Maximum faces to track |
| `AVP_FACE_WITH_ATTENTION` | `true` | Use attention landmark model (478 points vs 468) |
| `AVP_FACE_INFER_EVERY_N` | `1` | Run inference every N frames |
| `AVP_FACE_RESOURCE_ROOT` | auto | MediaPipe model resource directory |
| `AVP_FACE_METADATA_KEY` | `face_landmarks_v1` | Metadata key for landmark data |
| `AVP_FACE_DEBUG_LOG_EVERY_N` | `1` | C++ node log frequency (0 = off) |

Run via Docker (recommended):
```bash
docker run --rm --gpus all --network host \
  -v /path/to/test-content:/content:ro \
  -v /path/to/nvrtc/lib:/opt/nvrtc:ro \
  -e AVP_INPUT=/content/video.ts \
  -e AVP_OUTPUT=/output/out.ts \
  -e AVP_OUTPUT_FORMAT=mpegts \
  -e LD_LIBRARY_PATH=/usr/local/lib:/opt/tensorrt/lib:/usr/local/cuda-13.0/targets/x86_64-linux/lib:/opt/avplumber/deps/mediapipe-face-mesh/lib:/opt/nvrtc \
  avp-mediapipe:latest
```

## Writing Your Own `PythonNode`

Custom nodes subclass `PythonNode` and implement `process()`. A typical node
should:

1. Call `self._src.get()`. It blocks until a frame is available or the graph is
   closing; return if it gives back no frame.
2. Read or update `p.metadata` for downstream filters and drawing nodes.
3. Mutate frame planes only when the frame format and memory location match what
   your code expects.
4. Enqueue the frame to one or more destinations with `self._dst.enqueue(p)`.
5. Keep long-running setup, such as model loading, in `__init__`.

For GPU nodes, check `p.format.name`, expose CUDA memory to PyTorch with
`__cuda_array_interface__` only when the frame is actually CUDA-backed, and
synchronize the current CUDA stream before handing the frame back to downstream
AVPlumber nodes.
