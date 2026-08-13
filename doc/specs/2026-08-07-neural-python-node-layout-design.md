# Neural Python Node Layout

## Goal

Keep `pyplumber` focused on the Python binding and graph runtime. Reusable
neural processing implementations belong beside the C++ and CUDA code for the
same feature under `src/nodes/neural_net`.

## Layout

The neural source tree is organized by purpose, not implementation language:

```text
src/nodes/neural_net/
|-- scene_cut/
|   |-- cuda_scene_detect.py
|   |-- luma_diff.cpp
|   `-- luma_diff.cu
|-- tracking/
|   |-- ultralytics_bytetrack.py
|   `-- player_tracker.cpp
`-- vlm/
    |-- molmo/
    |   |-- node.py
    |   |-- transformers_runner.py
    |   |-- transformers_sidecar.py
    |   `-- vllm_runner.py
    |-- qwen/
    |   |-- node.py
    |   `-- transformers_runner.py
    |-- molmo2_preprocess.cu
    `-- qwen3_vl_preprocess.cu
```

Python package markers make these modules importable from a source checkout.
The C++ build continues selecting `.cpp` and `.cu` inputs explicitly, so the
colocated Python files do not enter native compilation.

## `pyplumber` Boundary

`pyplumber` retains:

- the native extension entry point and `AVPlumber` binding;
- graph and Python-node base classes;
- wrappers for native node types;
- graph-runtime helpers such as the mixer builder and RTCP feedback handling;
- low-level metadata decoding helpers used as part of the binding surface.

It no longer owns reusable neural node implementations. In particular:

- `CudaSceneDetectNode` moves to `src/nodes/neural_net/scene_cut`;
- `UltralyticsByteTrackNode` moves to `src/nodes/neural_net/tracking`;
- Molmo and Qwen nodes and runners move to `src/nodes/neural_net/vlm`;
- the unused Python `MetadataStoreNode` is removed rather than relocated.

The native `store_metadata` node is removed as well. Current EKA Recorder AI
publishes Kafka metadata through its application-owned metadata gate and
explicitly does not create `StoreMetadata`. The old scene-cut example is the
only remaining first-party caller, so its optional Kafka branch is removed.
Sports analysis will not use an external metadata sink.

## Examples and Imports

Neural examples move out of `pyplumber/examples` and live with the relevant
neural feature. Generic examples of authoring a `PythonNode` remain available
as framework examples.

All first-party imports, tests, Docker files, and documentation are updated to
the new module paths. The migration intentionally does not retain compatibility
modules under `pyplumber`: leaving forwarding modules would preserve the
misleading public layout this cleanup is intended to remove.

## Verification

- Import each relocated module from a clean Python process rooted at the
  repository checkout.
- Run the scene detector, VLM parser/runner, Qwen, Molmo, and RTCP unit tests.
- Compile all Python sources to catch stale imports and syntax errors.
- Inspect the generated native source list to confirm colocated `.py` files are
  not compiled.
- Do not run CUDA or TensorRT execution locally when no NVIDIA device is
  available; those checks require the configured NVIDIA environment.
