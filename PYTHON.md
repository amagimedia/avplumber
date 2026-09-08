# AVPlumber Python Bindings Manual

This is a practical guide for using AVPlumber from Python and creating custom Python-only nodes.

## Build `python_module` (short)

From repository root:

```bash
cd avplumber
make -j8 \
  NEURAL_NET=1 \
  HAVE_CUDA=1 \
  HAVE_NVOF_FRUC=1 \
  HAVE_NVCC=1 \
  NVCC=/usr/local/cuda-13.0/bin/nvcc \
  TENSORRT_ROOT=/opt/tensorrt \
  PKG_CONFIG_PATH=/usr/local/lib/pkgconfig \
  CXXFLAGS+=' -I/usr/local/include -I/usr/local/cuda-13.0/include -I/usr/local/cuda-13.0/targets/x86_64-linux/include' \
  LFLAGS+=' -L/usr/local/lib -Wl,-rpath,/usr/local/lib -L/usr/local/cuda-13.0/targets/x86_64-linux/lib -Wl,-rpath,/usr/local/cuda-13.0/targets/x86_64-linux/lib' \
  python_module
```

This script runs `make ... python_module` with CUDA/TensorRT-related flags used in this repo.

If your environment differs, run `make` manually and keep `python_module` as the target.

## 1) What the Python binding gives you

From Python you can:

- Build a full AVPlumber graph (`InputRec`, `Demux`, `DecVideo`, `FilterVideo`, `EncVideo`, `Output`, etc.).
- Add custom `PythonNode` classes into that graph.
- Read packets/frames from input edges and push them to output edges.
- Attach or modify frame metadata (`p.metadata[...]`) that downstream nodes can render/use.

---

## 2) Project setup pattern

The minimal framework sample lives in `pyplumber/examples/`. Reusable Python
neural nodes and their examples are grouped by purpose under
`src/nodes/neural_net/`. See
[`pyplumber/examples/README.md`](pyplumber/examples/README.md) for the index.

Those scripts add the local package to `sys.path` before importing `pyplumber`:

```python
import sys
sys.path.append("../..")

import pyplumber
from pyplumber.node import PythonNode, InputRec, Demux, DecVideo, ...
```

Then create a pipeline:

```python
avp = pyplumber.AVPlumber()
```

Add regular built-in nodes first, then add your Python node.

---

## 3) Minimal custom Python node

Subclass `PythonNode` and implement `process()`:

```python
class PythonSimpleNode(PythonNode):
    def process(self):
        p = self._src.get()
        if not p:
            return
        p.metadata["msg"] = "Hello from python"
        self._dst.enqueue(p)
```

Key points:

- `self._src.get()` blocks until the next item is available or the graph is
  closing. Return if it gives back no frame.
- `self._dst.enqueue(p)` forwards the item.
- If you do not enqueue (or intentionally drop), downstream stops receiving that frame.

---

## 4) `src`/`dst` shapes control node mode (SISO/SIMO/MISO/MIMO)

`PythonNode` mode is inferred from `src` and `dst` in constructor args:

- `src: "edge", dst: "edge"` -> SISO
- `src: "edge", dst: ["edge1", "edge2"]` -> SIMO
- `src: ["a", "b"], dst: "out"` -> MISO
- `src: ["a", "b"], dst: ["x", "y"]` -> MIMO

In SIMO/MIMO, `self._dst` is a dictionary of edges; otherwise it is a single edge.

```python
node = PythonSimpleNode({
    "src": "source_edge",
    "dst": "dest_edgs",
    "group": "g1",
    "name": "python-test-node",
})
```

---

## 5) Register and run your Python node

Create node instance and add it:

```python
avp.addNode(node)
```

Start groups:

```python
avp.group("in").startNodes()
avp.group("g1").startNodes()
```

Keep heartbeat alive:

```python
while True:
    time.sleep(1)
    avp.heartbeat()
```

---

## 6) Metadata workflow (very useful)

Metadata is the easiest way to pass custom values through the graph.

```python
p.metadata["person_count"] = person_count
p.metadata["msg"] = f"Persons detected: {person_count}"
```

Metadata may be drawn on image with FFmpeg `drawtext` inside a `FilterVideo` node:

```text:\\%{metadata:msg}
```

So your Python node can compute values and downstream video nodes can visualize them.

---

## 7) PyTorch / TorchVision custom processing

You can use Pytorch inside avplumber graph

- optional `torch` and `torchvision` imports,
- model initialization once in `__init__`,
- per-frame inference in `process()`,
- writing detection results into metadata and drawing boxes into the frame planes.

Recommended structure:

1. Initialize model once in `__init__`.
2. Handle missing dependencies gracefully (set status metadata, keep forwarding frames).
3. Run inference every N frames (`detect_every_n`) to keep throughput reasonable.
4. Always forward frame unless intentional drop.

Example parameters:

```python
node = PyTorchNode({
    "src": "src_node",
    "dst": "dst_node",
    "group": "g1",
    "name": "python-test-node",
    "detect_every_n": 5,
    "person_score_threshold": 0.60,
    "max_person_boxes": 8,
    "box_thickness": 4,
})
```

---

## 8) Putting Python node into a larger production graph

`pyplumber/examples/tracker-live.py` shows how to assemble a larger CUDA/TensorRT
tracking graph from Python:

- upstream writes metadata,
- Python node enriches metadata or performs custom logic,
- downstream nodes continue standard AVPlumber processing.

This is a good pattern when you need rapid iteration for custom logic without adding a new C++ node.

For the smaller runnable examples, start with `pyplumber/examples/simple-node.py`
and the usage guide in `pyplumber/examples/README.md`.

---

## 9) Common pitfalls

- **No output forwarding:** if you forget `enqueue`, output can stall.
- **Wrong edge names:** `src`/`dst` must match actual edges in graph.
- **Frame format mismatch:** Python pixel manipulation usually expects CPU-readable planes.
- **Performance:** heavy Python inference each frame can bottleneck; use sampling (`every_n`) and/or lower inference resolution.
- **Dependency availability:** guard imports and degrade gracefully.

---
