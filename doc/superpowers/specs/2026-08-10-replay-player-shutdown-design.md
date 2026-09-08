# Replay player shutdown design

## Scope

Fix shutdown of the single-source replay demo without changing AVPlumber's
Python bindings, graph management, or shared node implementations.

## Cause

The player calls `AVPlumber.shutdown()` while its `PositionProbe` Python node
can still be running. Shutdown holds the Python GIL while it joins nodes, but
the probe needs the GIL to finish. The probe also retains its wrapper, graph,
and CUDA-backed edges after its worker exits.

## Design

`PlayerApplication.stop()` will stop the position probe first and wait, with a
bounded poll, until its worker is no longer running. It will then detach the
probe's wrapper, AVPlumber, and edge references. Only after that will it stop
the native output and player groups and call the unchanged
`AVPlumber.shutdown()` binding.

The cleanup is idempotent. A timeout stopping the probe is reported instead of
entering native shutdown with an active Python worker.

## Rejected alternatives

- A new C++ observation node broadens shared code for a demo-only need.
- An edge wiretap still invokes Python from a media worker and retains Python
  callback ownership during graph teardown.
- Releasing the GIL in the global `shutdown` binding changes framework-wide
  behavior and does not solve late ownership of CUDA-backed edges.

## Verification

- A focused fake-graph test must prove stop, wait, detach, group stop, and
  shutdown ordering, including timeout and idempotency.
- All local replay tests must pass.
- The NVIDIA/Janus `--exercise-v2` run must pass every operation and exit zero
  without a shutdown timeout, crash, or CUDA teardown error.
