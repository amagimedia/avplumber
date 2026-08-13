# Basketball Demo Merge Cleanup Design

## Goal

Make the accumulated `basketball-demo` integration branch clean enough to
merge into `develop` without changing its intended feature scope. The removal
of the older basketball, sport-specific, neural-demo, and auto-mixer code is
intentional because the newer demos replace it.

## Scope

The cleanup will:

- replace committed workstation paths with repository-relative paths;
- follow the documented node-file and `DECLNODE` conventions;
- replace duplicated scene-cut CUDA/frame setup with shared helpers;
- update stale NVOF work-in-progress documentation to describe the implemented
  build gate and node;
- remove first-party whitespace errors reported by `git diff --check`;
- preserve imported patch files and vendored NVIDIA headers byte-for-byte while
  configuring Git whitespace checks not to report their upstream formatting;
- record the resolved scope and verification results on PR #19.

The cleanup will not restore superseded demos, alter graph behavior, or perform
unrelated refactoring.

## Non-Regression Invariants

The cleanup must not change node parameters, metadata keys or schemas, CUDA
context/stream ownership, build feature gates, graph topology, or runtime
behavior. Shared helpers will preserve the existing branches, log messages,
return values, and ownership flags of both callers. Formatting and documentation
changes must not alter imported patch semantics or generated media assets.

## Shared CUDA Helpers

Add a small header under `src/nodes/neural_net/scene_cut/` for behavior shared
by `luma_diff` and `hog_diff`:

- CUDA error reporting with a caller-provided node name;
- supported CUDA luma-format detection;
- JSON string escaping used by diagnostic metadata;
- extraction of the software pixel format from a hardware frame;
- initialization of the CUDA context and stream from an input frame.

The context initializer will operate on caller-owned context, stream, and
ownership fields. This keeps resource lifetime in each node and avoids a new
class hierarchy or framework-wide interface.

## Vendored Whitespace

Imported Chromium/FFmpeg patch files and NVIDIA Optical Flow SDK files retain
their upstream contents. Path-specific `.gitattributes` rules disable only
trailing-whitespace diagnostics for those files. First-party sources remain
subject to the repository's normal whitespace checks and will be corrected.

## Verification

The branch is clean when:

- `git diff --check origin/develop...HEAD` passes;
- Python compilation succeeds;
- root, mixer, playlist, replay, CUDA-overlay reference, and DMA-BUF Python
  suites pass independently;
- DMA-BUF browser unit tests pass;
- the same local suites that passed before cleanup still pass afterward with no
  new skips;
- CodeQL completes successfully;
- an NVIDIA integration host builds the binary and Python module with matching
  CUDA, neural, NVCC/PTX, TensorRT, DRM, and GL features, with FRUC disabled;
- relevant zero-copy demo smoke checks run without CPU upload/download
  workarounds.
