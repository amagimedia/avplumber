# pyplumber Examples

These programs build AVPlumber graphs through the Python bindings. Run them
from the repository root so the local `pyplumber` package is importable.

Most examples read their input and output from environment variables:

```bash
AVP_INPUT=/path/to/input.mp4 \
AVP_OUTPUT=/tmp/output.ts \
python3 pyplumber/examples/simple-node.py
```

Common variables are `AVP_INPUT`, `AVP_OUTPUT`, `AVP_OUTPUT_FORMAT`, and
`AVP_USE_REALTIME`. CUDA examples require an AVPlumber build with CUDA and a
compatible NVIDIA driver. TensorRT examples additionally need the appropriate
native build flags and model assets.

## Examples

- `simple-node.py`: minimal `PythonNode` frame-processing graph.
- `pytorch-node.py`: CPU PyTorch inference on decoded frames.
- `pytorch-cuda-node.py`: direct access to CUDA frame memory from PyTorch.
- `torchvision-node.py`: torchvision object detection on CUDA frames.
- `blazeface-node.py`: BlazeFace inference implemented in Python.
- `ultralytics-bytetrack.py`: Ultralytics inference with ByteTrack metadata.
- `cuda-scene-detect.py`: GPU scene-change detection with optional Janus
  video output.
- `tracker-live.py`: larger basketball tracking and analytics graph.
- `molmo-vllm-node.py`: Molmo video-window inference integration.

The generic manual N-input mixer is maintained separately under
`demos/mixer/`. It provides fullscreen and 2/4/8/16-box portrait layouts,
preheated CUDA transitions, a Textual TUI, and video-only Janus output.

## Python Node Contract

Custom nodes subclass `PythonNode` and implement `process()`. A node normally
reads one item from its source, updates the payload or metadata, then enqueues
it downstream. Long-running setup belongs in `__init__`.

For CUDA frames, verify the frame format before exposing memory through
`__cuda_array_interface__`, and synchronize the CUDA stream before returning
the frame to AVPlumber.
