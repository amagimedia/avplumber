# Integration Build Compatibility Design

## Goal

Keep the consolidated avplumber branch compatible with the EKA recorder and the
Streamstudio OBS plugin without restoring removed auto-mixer or ball-tracker
framework code.

Success means:

- A plain `HAVE_CUDA=1` build neither compiles nor links nvJPEG unless
  `HAVE_NVJPEG=1` is also supplied.
- TensorRT sources and libraries are selected only when TensorRT is explicitly
  enabled.
- Develop-era neural build flags continue to select the retained neural nodes as
  compatibility aliases for the consolidated `NEURAL_NET` flag.
- The OBS plugin states that it does not need nvJPEG, TensorRT, or neural nodes.
- EKA Compose builds Janus and the web UI from their current avplumber locations.
- EKA documentation and tests describe raw TrackNet ball detections, with no
  interpolation, coasting, tracker prediction, or predicted fallback.

## Repositories and isolation

Changes will be prepared in separate worktrees and commits for avplumber, EKA,
and OBS. The active worktrees, including concurrent basketball-demo work and
uncommitted downstream changes, will not be edited. No submodule revision will
be advanced as part of this compatibility fix.

## avplumber build contract

`HAVE_NVJPEG` becomes an independent, opt-in feature with a default of `0`.
The existing source filtering remains the enforcement point: `nvjpeg_enc.cpp`
and its `nvjpeg`/`cudart` link dependencies are present only when both CUDA and
nvJPEG are enabled.

`NEURAL_NET` remains the preferred switch for all retained neural nodes.
`NEURAL_NET_COMMON=1` or `NEURAL_NET_SPECIFIC=1` will enable `NEURAL_NET` when
the preferred switch was not set explicitly. These aliases preserve old build
commands; they do not recreate the removed common/sport-specific source layout.

TensorRT inference sources, preprocessing kernels, the `HAVE_TENSORRT`
preprocessor definition, include/library paths, and TensorRT libraries will all
be gated by `HAVE_TENSORRT=1`. Non-TensorRT neural nodes remain available with
`NEURAL_NET=1 HAVE_TENSORRT=0`.

The README and development basics will document these independent gates and the
legacy neural aliases.

## OBS integration

The embedded avplumber make invocation will explicitly pass:

```text
HAVE_NVJPEG=0 HAVE_TENSORRT=0 NEURAL_NET=0
```

This records the plugin's intended dependency surface and protects it if an
avplumber default changes later. The plugin's avplumber submodule will remain at
its existing revision.

## EKA integration

The Janus and web-ui services will use the avplumber submodule as their build
context and select these Dockerfiles:

```text
docker-compose/images/janus/Dockerfile
docker-compose/images/web-ui/Dockerfile
```

Ball metadata will continue to come directly from TrackNet's raw detection
payload. Stale EKA documentation and assertions that promise a `BallTracker`,
coasted/interpolated detections, prediction flags, or predicted fallback will be
rewritten or removed. References to interpolation in unrelated media, player,
or camera-motion behavior are outside this change.

## Validation

On the local non-NVIDIA host, Make's resolved database and dry-run output will
verify these configurations without compiling CUDA code:

- CUDA enabled with default nvJPEG: no nvJPEG source or libraries.
- CUDA and nvJPEG enabled explicitly: nvJPEG source and libraries included.
- Neural enabled without TensorRT: retained non-TensorRT nodes included and no
  TensorRT sources, definition, paths, or libraries included.
- Neural and TensorRT enabled: TensorRT sources and libraries included.
- Each develop-era neural alias enables the consolidated neural node set.

OBS validation will inspect its resolved make command and run available CMake
configuration checks that do not require a GPU. EKA validation will resolve the
Compose configuration, verify both Dockerfile paths, scan for stale ball
tracking promises, and run targeted non-GPU tests covering changed contracts.

CUDA, TensorRT, NVENC, and OBS runtime execution require the configured NVIDIA
environment and will be reported as unverified if that environment is not
available during this task.

## Non-goals

- Restoring the deleted native ball tracker or auto-mixer Docker hierarchy.
- Changing graph management, the control protocol, framework lifecycle, or OBS
  runtime APIs.
- Updating downstream avplumber submodule revisions.
- Refactoring unrelated Makefile, recorder, or plugin code.
