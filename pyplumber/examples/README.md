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
`AVP_USE_REALTIME`. Reusable Python nodes and their examples live beside the
related native sources.

## Examples

- `simple-node.py`: minimal `PythonNode` frame-processing graph.

Feature-specific examples:

- `src/nodes/scene_cut/example.py`: CUDA scene-change detection;
- `src/nodes/neural_net/examples`: generic PyTorch, torchvision, and
  BlazeFace examples;
- `src/nodes/neural_net/tracking/example.py`: Ultralytics with ByteTrack;
- `src/nodes/neural_net/vlm/molmo/example.py`: Molmo video-window inference.

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
