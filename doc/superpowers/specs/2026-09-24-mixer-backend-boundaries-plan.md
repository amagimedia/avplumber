# Mixer backend boundaries

Status: implementation plan, not implemented. Based on `mixer-improv` at
`3e4b251`. Scope agreed 2026-09-24: prepare mixer composition for a future
Vulkan implementation and eventual AMD/Intel deployment while preserving the
working CUDA pipeline.

## Scope and invariants

The first phase isolates backend selection and commands. CUDA remains the only
implemented backend. It does not advertise Vulkan support or change the demo UI.
One compositor backend is selected per mixer instance and shared by PGM, aux
and wipes. Mixed CUDA/Vulkan composition buses are out of scope; the adapter
must not expose per-bus backend overrides.
Decoder and encoder implementations remain independent of compositor selection;
NVDEC/NVENC, their options, and current browser/v210 ingest remain unchanged.

Preserve these properties:

- Existing public configurations work without new fields. CUDA is the default.
- Node and edge names, topology, capacities, groups, startup order, filter strings,
  device references and failure policies remain equivalent.
- Preserve both warm scene slots, source sharing and reference-only aux delivery.
- Preserve fade expressions, timestamp units, direction, warmup and cleanup.
- Preserve wipe decode/upload, overlay readiness, scene switch point and tail drain.
- Preserve color tags, alpha association, chroma handling, format promotion and
  identity-conversion fast paths.
- Do not change kernels, frame allocation, CUDA streams/events, synchronization,
  DMA-BUF lifetime/cache bounds, playout history or encoder buffering.
- No new per-frame dispatch in Python, extra graph nodes, queues or pixel copies.

### Downstream compatibility

The current downstream live reframer uses `pyplumber.node.CudaRectOverlay`
directly and imports `pyplumber.mixer.dmabuf_inputs.dmabuf_cuda_input_nodes`.
Its application source does not currently use `MixerGraphBuilder` or mixer
fade commands. Therefore builder-only compatibility checks are insufficient.

Preserve the concrete CUDA node name, Python import, node parameters and DMA-BUF
helper signature/return structure. In particular preserve:

- Unclocked composition when no `fps` is supplied: upstream owns pacing.
- Per-frame `metadata_key` layers, including moving crops, contain and two-box
  layouts; existing source-index and z-order semantics.
- Program metadata propagation with program placed last in the input list but
  drawn first, beneath the browser overlay.
- Premultiplied browser alpha, texture-backed input handling and shared browser
  surfaces across output aspects, without added copies or retained frames.
- Acceptance of existing legacy parameters such as `scale=True`; do not turn
  ignored parameters into errors as incidental cleanup.

The downstream repository pins its own AVPlumber revision. Do not update that
pin or require application changes as part of this preparation. Validate its
graph/layout contracts against the candidate AVPlumber code in an isolated
remote environment, and add a native unclocked metadata/alpha smoke case; stub
graph tests alone cannot establish rendering compatibility.

Zero-copy means avoiding unnecessary intermediate pixel copies and CPU round
trips. Composition necessarily writes an output surface. Required conversions
must be explicit; never hide an incompatible backend behind CPU download/upload.
Future AMD/Intel interop must be demonstrated on hardware, not inferred from
Vulkan support. Unsupported combinations fail before starting the graph where
capabilities are known; runtime format changes must also be validated.

## Boundaries

Shared mixer code owns scenes, geometry, color intent, clocks, routing, warmup,
subscriptions, queue budgets and transition scheduling. Backend code chooses
concrete composition, conversion, scaling and transition implementations.

Use a small Python backend object with one concrete `CudaMixerBackend`. Inject
it into `MixerGraphBuilder`; aux uses that same backend. Keep existing node API
injection usable by the demo and tests. No plugin registry or speculative backend
inheritance tree is needed. Methods describe existing operations and produce
node specifications or filter graphs; callers continue to own graph topology.

The initial operations are compositor construction, transition construction,
color conversion, scaling and wipe upload preparation. Move CUDA spellings into
the adapter without changing the strings it emits. Device/context creation and
external ingest are integration boundaries, not portable implementations in
this phase. Do not claim all demo code becomes GPU-independent.

Keep FFmpeg pixel formats and `av::VideoFrame` ownership. NV12/P010/P210 describe
storage and are not CUDA APIs. Custom source/scene filter graphs remain an
explicit backend-specific escape hatch; do not attempt to translate them.

## Directory ownership

Use these locations for the preparation work:

```text
pyplumber/mixer/
  backend.py                 # backend contract and explicit selection
  backends/
    __init__.py              # no eager node/native imports
    cuda.py                  # concrete CUDA node/filter construction
  color.py                   # color intent, validation, compatibility wrapper
  graph.py                   # shared topology and routing
  aux.py                     # shared aux layout and lifecycle

src/mixer/
  transition_control.hpp     # backend-independent fade request/control contract
  backends/
    cuda/
      transition_control.cpp # existing CUDA expression/command translation
  orchestrator/              # cut/fade/wipe scheduling and routing
  primitives/                # shared timing, geometry and frame subscriptions

src/nodes/hwaccel/            # existing registered hardware nodes and CUDA helpers
```

`backend.py` must not accumulate CUDA filter strings or concrete node creation.
`backends/cuda.py` is the single implementation location for the Python adapter;
it consumes shared color contracts rather than redefining them. Keep it as one
module until its size or responsibilities justify a CUDA package. Do not create
empty Vulkan directories or an unused renderer interface.

Backend selection may import the concrete implementation lazily. Importing
configuration/color helpers must continue to work without `_avplumber`. The
legacy `color.conversion_graph` wrapper delegates lazily to the CUDA function;
the CUDA module must not import that wrapper back. Existing public imports stay
stable while implementation code moves.

In C++, shared contracts stay outside `backends/cuda`; only the implementation
includes backend-specific dependencies. `orchestrator` owns when operations
happen; the CUDA directory owns how they map to CUDA processing. The fade
adapter is mixer-specific control code and has no `DECLNODE`, so it belongs
under `src/mixer`, not `src/nodes`.

Keep existing registered nodes, kernels and their support files in place during
this phase. Later renderer extraction can move mixer-specific draw, allocation
and synchronization implementation together into `src/mixer/backends/cuda`,
leaving registered node wrappers under `src/nodes`. General DMA-BUF import and
other reusable hardware nodes are not mixer internals and stay outside it.
Do not split one implementation across both locations or duplicate helpers.
Check source discovery and feature guards when adding implementation files;
directory organization must not introduce CUDA dependencies into shared builds.

## Commit 1: route mixer construction through the CUDA adapter

Primary files:

- `pyplumber/mixer/backend.py` (new)
- `pyplumber/mixer/backends/__init__.py` (new)
- `pyplumber/mixer/backends/cuda.py` (new)
- `pyplumber/mixer/graph.py`
- `pyplumber/mixer/aux.py`
- `demos/mixer/mixer.py`

1. Capture representative pre-refactor graph construction results on the remote
   test environment: no aux, aux enabled, SDR, HDR 420/422 and wipes. Include
   named/routed sources where supported and both initial PGM slots.
2. Add the default CUDA adapter and pass it through the builder to aux.
3. Route PGM A/B, wipe overlay and aux compositor creation through the same
   operation. Preserve all scheduling and subscription parameters at callers.
4. Move existing transition graph and built-in scale/upload construction into
   adapter operations. Avoid a general filter-expression language.
5. Reject unknown backend names; do not silently select CUDA for a Vulkan request.

Check complete node parameters, edge capacities and emitted mixer commands
against the baseline, not only node counts. A recording adapter should also
verify that all three compositor uses go through the boundary; it need not fake
a working Vulkan implementation. Exclude only genuinely nondeterministic IDs
from comparisons, never timing, color, capacity or device parameters.

## Commit 2: backend-independent fade control

Primary files:

- `src/mixer/orchestrator/fade.cpp`
- `src/mixer/orchestrator/MixerOrchestrator.hpp`
- `src/mixer/primitives/MixerState.hpp`
- `src/mixer/transition_control.hpp` (new)
- `src/mixer/backends/cuda/transition_control.cpp` (new)
- `src/avplumber.cpp` (existing `mixer.init` handler only)
- `pyplumber/mixer/graph.py`

Introduce a narrow mixer-local transition control interface accepting the
existing start timestamp in milliseconds, duration in seconds, and source/destination
slot direction. The CUDA implementation alone builds the existing alpha
expression and sends the existing `filter_command` to `transition_cuda`.

The orchestrator continues to own preparation, routing, timeline publication,
cancellation generations and cleanup. Configure the adapter once per mixer,
not once per frame. The adapter uses the existing node-object command mechanism
and the existing permanent transition filter; it adds no processing node.

Forward the mixer instance's single optional backend setting through
`mixer.init`; the transition adapter is selected from that same setting. Do not
introduce an independently selectable transition backend. An optional transition
node name may identify the existing processing node. Defaults reproduce the
current derived transition node name and CUDA behavior. The graph builder must
emit the same backend choice it used for PGM, aux and wipe construction.
Keep new configuration parsing in the mixer module as much as possible. The
small forwarding change in `src/avplumber.cpp` is the only central-file change:
no command syntax, graph management, sentinel or generic filter-node changes.
Retain the public `mixer.fade` API and legacy direct filter commands.

Tests must compare CUDA command target, expression and update ordering with the
old path for A-to-B and B-to-A fades, fractional durations and existing boundary
cases. Verify invalid backend settings fail at initialization. Exercise interrupted
fades and warm transitions remotely; do not improve rounding or timing here.

## Commit 3: shared validation and color intent

Primary files:

- `src/mixer/primitives/compositor_layers.hpp`
- `src/nodes/hwaccel/cuda_rect_overlay.cpp`
- `pyplumber/mixer/color.py`
- `pyplumber/mixer/backends/cuda.py`
- `demos/mixer/mixer.py`

Move the `HAVE_CUDA_RECT_SCALE`/NVCC sizing restriction out of shared layer
parsing into CUDA capability validation. Preserve early failure when CUDA
cannot execute the requested operation, including runtime composition updates.
Shared geometry tests must compile and work without CUDA-specific defines.

Keep color metadata, transfer contracts and parameter validation shared. Move
CUDA filter generation behind the adapter and update mixer callers. Preserve
the existing `conversion_graph` import through a thin compatibility wrapper
for external users; avoid circular module imports. Do not simultaneously loosen
format restrictions or change tone mapping. Renderer-specific format support
belongs to capability validation, independently of whether a color description
is valid.

Use existing color tests to verify identical generated filter graphs and errors.
Keep rendition encoder policy separate: extracting it into a new generic codec
layer is outside this patch series.

## Deferred native rendering interface

Do not extract allocation/synchronization from `CudaRectOverlay` in this phase.
Before implementing a second renderer, define its responsibilities together:

- Validate actual input storage and resolve backend-specific imports.
- Allocate output surfaces and consume shared resolved draw operations.
- Wait for producer readiness and retain inputs until GPU reads complete.
- Publish output with the backend's correct completion/ownership information.
- Drain and release resources safely on stop and failure.

Keep CUDA streams, texture handles and Vulkan image layouts/semaphores private
to implementations. Do not expose a generic device pointer plus stream, or
assume that separate GPU queues always execute concurrently. The invariant is
that aux does not add a blocking dependency to PGM. Reuse FFmpeg's hardware-frame
contracts rather than inventing a second image ownership system.

A later Vulkan experiment must prove video decode -> compositor -> encoder and
DMA-BUF browser -> compositor, including synchronization, alpha and bounded
retention. Prove supported SDR/HDR and 420/422 cases individually. Its results
will determine the native renderer interface and any required interop adapter.

## Validation and rollout

All builds and tests run in the configured remote environment; no local tests.
For each commit run mixer/demo Python coverage, affected native tests, and a
remote build with the existing CUDA/DRM/GL/NVCC and neural/TensorRT feature set.
Rebuild Python bindings with matching flags. Keep web UI registration enabled.

Before deployment, run the isolated CUDA smoke tests with distinct ports and
resources. Do not run additional GPU tests during performance measurements.
Preserve the previous binary/configuration for rollback and replace native
modules atomically, never overwrite a library mapped by a running process.

Compare the same fixed source set, scenes, aux assignments, encoder settings and
cut/fade/wipe sequence before and after. Include PGM alone, PGM plus aux, rapid
transitions and aux consumer stall/recovery. Record output cadence and gaps,
cut latency, GPU use, steady/peak VRAM and browser retained/quarantined buffers.
Compare repeated steady-state windows to distinguish variation from regression;
do not declare success from average GPU utilization alone.

Acceptance: equivalent graphs and fade commands, passing pixel/color and control
tests, unchanged warm-cut behavior, no new aux-to-PGM blocking, and no reproducible
increase in frame gaps, latency or memory retention. Stop and investigate any
regression rather than compensating with larger queues or latency allowances.

Efficiency is a release gate. A reproducible 5% relative increase in CUDA/SM
utilization is unacceptable even with visually smooth output (for example,
40% to 42%, not just 40% to 45%). Target no measurable regression; smaller
changes are not an automatic allowance. Compare repeated matched workloads
after warmup, controlling other GPU processes and clocks. Include process GPU
metrics and per-frame kernel/copy timings where aggregate utilization is too
coarse to resolve the difference. Ambiguous measurements require further
measurement, not a passing verdict. VRAM, CPU cost and latency must not hide
regressions behind unchanged GPU utilization.

Expected preparation cost: roughly 200–400 net new lines including tests, with
existing construction code moved rather than duplicated. This is an estimate;
reassess if the fade adapter or compatibility scaffolding materially exceeds it.
Each commit must remain independently reviewable and revertible.
