# Scene-Cut Module Layout

## Goal

Make scene cutting the source-layout owner for all current scene-cut and
camera-motion implementations. Most of these algorithms are classical CUDA
processing, so their location and build selection must not imply that they are
neural networks.

This is a source-organization and build-gating change. It does not change node
types, parameters, metadata, runtime behavior, or graph-management behavior.

## Source Layout

Move the complete directory:

```text
src/nodes/neural_net/scene_cut/
```

to:

```text
src/nodes/scene_cut/
```

The destination remains organized by the scene-cut domain rather than by
implementation language or acceleration mechanism. It contains:

- the CUDA luma-difference metric;
- the CUDA HOG-difference metric;
- the TensorRT pairwise ONNX detector;
- the CUDA/NVIDIA Optical Flow camera-motion estimator;
- their CUDA kernels and shared CUDA-frame helpers;
- the CuPy scene detector and its example.

The existing node type names remain unchanged. Python imports move from
`src.nodes.neural_net.scene_cut` to `src.nodes.scene_cut`; no compatibility
package remains at the old path.

The CPU PySceneDetect program remains under `pyplumber/examples`. It is a
standalone example rather than a reusable scene-cut node and is outside this
move. This change does not introduce a native CPU scene-cut implementation.

## Build Selection

Every native implementation in `src/nodes/scene_cut` requires CUDA. The
Makefile must select each source only when its complete dependency set is
enabled:

| Implementation | Required build settings |
|---|---|
| `luma_diff` | `HAVE_CUDA=1 HAVE_NVCC=1` |
| `hog_diff` | `HAVE_CUDA=1 HAVE_NVCC=1` |
| `cuda_infer_scene_cut_onnx` | `HAVE_CUDA=1 HAVE_NVCC=1 HAVE_TENSORRT=1 NEURAL_NET=1` |
| `cuda_camera_motion` | `HAVE_CUDA=1 HAVE_NVOF=1` and the dense NVOF headers |

`HAVE_NVCC=1` continues to enable the optional GPU IRLS backend for
`cuda_camera_motion`; without NVCC, the existing non-PTX estimator path remains
available. `HAVE_OPENCV=1` remains an optional backend setting for that node.

The learned ONNX detector remains behind `NEURAL_NET=1` because that particular
algorithm is neural. The classical luma, HOG, and camera-motion algorithms no
longer depend on `NEURAL_NET`.

No new `SCENE_CUT` build flag is introduced. CUDA is a real implementation
dependency for every current native scene-cut node, while another feature flag
would only duplicate selection state. If a non-CUDA native implementation is
added later, its source can be selected by its own dependencies without moving
the scene-cut module again.

The CuPy implementation is not compiled by the C++ Makefile. It retains its
runtime CuPy/CUDA availability checks.

## Internal Sharing

The CUDA-frame helper stays private to the scene-cut module and moves with its
callers. The move does not introduce a new public interface or base class:
current algorithms expose distinct node types and there is no second adapter
requiring a new seam. Common implementation details should continue to be
shared through the existing helper rather than copied between nodes.

## Documentation and References

Update Makefile source paths, embedded CUDA-header paths, Python imports,
examples, and documentation references to the new directory. Move
`doc/specs/neural_net/scene_cut.md` to `doc/specs/scene_cut.md` so the
documented ownership matches the source layout.

Documentation of `NEURAL_NET` must describe only the learned scene-cut detector
as neural; it must not imply that the whole scene-cut family requires that
flag.

## Verification

Verify Makefile source selection for these configurations without relying on
stale generated objects:

- CUDA and NVCC enabled, neural disabled: `luma_diff` and `hog_diff` are
  selected; the ONNX detector is not.
- Neural enabled without CUDA: no source under `src/nodes/scene_cut` is
  selected.
- CUDA, NVCC, TensorRT, and neural enabled: the ONNX detector and its CUDA
  kernel are selected.
- CUDA and NVOF enabled with dense headers: `cuda_camera_motion` is selected;
  NVCC independently controls its GPU IRLS kernel.
- CUDA disabled: no native source under `src/nodes/scene_cut` is selected.

Import the relocated Python scene-cut package from a clean repository-root
Python process and check for stale old-path imports. Regenerate the node factory
through the normal build workflow after the move.

Run the CUDA/TensorRT build and node smoke checks on the configured NVIDIA host,
not on a local machine without the required hardware. Confirm that node type
names and emitted metadata remain unchanged.

## Out of Scope

- Adding a CPU scene-cut node.
- Changing scene-cut metrics, thresholds, debounce policy, or metadata schemas.
- Removing the existing `LumaSceneCut` Python wrapper.
- Refactoring framework source, graph management, or node registration.
- Introducing a general inference interface solely for this move.
