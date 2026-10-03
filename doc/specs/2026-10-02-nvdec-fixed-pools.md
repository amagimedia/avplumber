# NVDEC optimisation plan

## Implemented direction: upstream FFmpeg

Custom NVDEC implementation work is stopped. The only experimental native code
was an unbuilt driver-loading helper; it has been removed. No native decoder was
integrated or deployed. The native design below is retained as background, not an
active implementation plan.

The implemented path uses the pinned upstream FFmpeg revision and codec headers
recorded in `deps/ffmpeg/9/bases.env`, with the existing CUDA-filter patches ported
to it. AVP and avcpp build against this revision and FFmpeg 8.1. Ordinary native
`hevc`, `h264` and `av1` decoders use NVDEC with optional CUarray output; this is
not the separate `*_cuvid` decoder path. A stable FFmpeg 9 release must not be
assumed to contain the same features as the pinned development revision.

The mixer owns resizing and compositing and now consumes YUV arrays directly.
Same-size, same-format `scale_cuda` stages preserve CUarray storage and frame
handles. Array scaling or format conversion in that filter fails explicitly.
The extended `pad_cuda`, `crop_cuda` and `transition_cuda` filters consume arrays
and produce linear CUDA output; other filters retain their linear-input contract.
GPU reads are ordered against the decoder's producer stream, and input frame
references remain alive until those reads complete.

Preserve the current per-source decoder selection, linear buffers, CPU paths,
decoded queue of three frames and aux delay of 80 ms. Compare the unchanged
192-input / 20-output workload before and after each change, including VRAM,
CPU threads and load, decoder/encoder/SM utilisation, pixel and metadata
correctness, deadlines, repeats and cut latency. Record startup and stress peaks.
Keep the current build as rollback. Acceptance requires true decoder zero-copy,
lower VRAM and demonstrated capacity for additional aux outputs or NVDEC inputs,
with preserved quality, buffering, CPU load and mixer deadlines. A build or
version upgrade without useful capacity gain is not a successful outcome.

Opt-in sources use existing decoder options with `pixel_format=cuarray`,
`hwaccel_flags=unsafe_output` and `threads=1`. The source configuration defaults
to `extra_hw_frames=3`, supplementing codec and FFmpeg working surfaces; this is
not a total pool size or a validated budget for every graph. The patched upstream
NVDEC path honors the explicit budget instead of adding another sixteen arrays.
Three extras passed finite decoder tests retaining up to eight output references,
but the full mixer exhausted its 12-surface HEVC pools at startup. The tested
20-output configuration uses eight extras (17 surfaces); the 28-output version
needs twelve extras (21 surfaces). Seventeen surfaces exhausted with 28 outputs.
Pool demand depends on codec requirements and distinct frames retained across the
whole graph, not just its three-frame decoded edge. AVP decoder and graph APIs
remain unchanged. Framework changes require a demonstrated necessity.

FFmpeg 8.0 and 8.1 share the unchanged `deps/ffmpeg/8` series. All AVP changes must
still build and run with 8.1; guard new array functionality by availability.
Replay's immediate intra-frame seeking is a separate regression milestone.
Review `flush_magic` against actual decoder flushing and packet ownership, and
test target pixels, stale output, single-packet latency and retained frames on
both library versions before changing the workaround. Newer FFmpeg behavior
alone does not justify removing support needed by 8.1 or CPU decoding.

## Completed regression coverage

All GPU checks below ran on the NVIDIA validation host with explicit runtime
library selection. Reports remain outside the repository; test scripts accept
fixture paths at runtime. These checks establish behavior for the tested inputs,
not a universal capacity or latency guarantee.

| Coverage | Result and boundary |
|---|---|
| AVP/avcpp compatibility | Full Python modules built against the pinned upstream revision and FFmpeg 8.1, including the final compositor changes. |
| Native C++ regressions | Playout, preview follow, preview swap and source-mask tests passed against both library versions. The standalone compiler flags now include the same constant-macro definition used by avcpp. |
| CPU decoding | H264 and HEVC Main10 fixtures decoded all 50 frames through AVP, reached EOF and shut down without initializing a CUDA device. This is not a CPU-only build validation. |
| Native NVDEC pixels and retention | `tests/cuda/nvdec/probe.py`: 28 cases passed. Upstream linear CUDA and CUarray pixels, dimensions and PTS match FFmpeg 8.1 CUDA for H264, HEVC Main10 and AV1, including reordered pictures and non-aligned 638×358 dimensions. Each codec/storage path also completed with 3 or 8 retained GPU frame references, without accessing their pixel data. |
| AV1 software reference | FFmpeg 8.1 CUDA also exactly matches an explicitly labelled external libdav1d CPU CLI reference. The validation AVP builds have no AV1 software decoder; this is not an AVP CPU-AV1 result. |
| Real decoder/compositor chain | `tests/cuda/nvdec/compositor_decode.py`: 50 frames across two outputs match linear CUDA, including identity normalization and explicit pad/crop into linear storage. EOF, handle passthrough, surface reuse and shutdown are checked. |
| Mixed codecs in one graph | `tests/cuda/nvdec/mixed_decode.py`: simultaneous HEVC CUarray plus H264/AV1 linear CUDA share one device and compose three scaled panels. All 50 canvas pixel hashes and normalized PTS match the all-linear baseline; input geometries differ and every input/output reaches EOF. |
| Filter pixel matrix | `tests/cuda/nvdec/filters.py`: 80 cases passed, covering NV12/P010/P210 array consumers, pad/crop and transition modes, including mixed array/linear inputs. Synthetic P210 coverage does not establish HEVC 4:2:2 hardware decoding. |
| Final FFmpeg 8.1 module | Linear CUDA compositor produced 50 frames on both outputs; HEVC CUDA pixels matched the saved CPU reference; holding eight GPU frames completed with EOF and shutdown. |

H264 and HEVC Main10 CPU hashes differed from NVDEC already on FFmpeg 8.1 for
the non-aligned fixtures. Their cross-version GPU results match exactly; the
CPU/GPU difference remains to be characterized and is not attributed to the new
CUarray path. CPU CLI `showinfo` checked BT.709 and HLG/BT.2020 fixture colour tags
on both FFmpeg builds; this does not prove complete metadata propagation through
the AVP graph. The probes do not verify held pixel contents across seek/flush.

Still pending: immediate intra-frame replay seeking and stale-frame rejection,
held-frame lifetime across flush, corrupt input and sequence-change recovery,
compressed HEVC 4:2:2 on capable hardware, and a CPU-only build without NVIDIA
dependencies. Full-show memory, performance, stress and output-capacity results
are documented separately in the FFmpeg and mixer configuration documentation.
The [capacity report](../research/2026-10-03-cuarray-capacity.md) distinguishes
the matched 16-extra-aux comparison (20 total outputs) from the 28-total-output
capacity setup (24 extra aux). Both totals include two program outputs and two
monitors. Different output counts must not be treated as a matched performance
comparison. `tests/cuda/nvdec/aux_cycles.py` provides a separate finite live test:
it staggers pages only on extra aux buses advertising `source_pages`, checks
accepted commands and applied compositions, then restores original layouts/pages
and verifies unchanged scene assignments. Its report names the selected buses;
it does not imply every aux supports page cycling or prove source pixel freshness.

## Earlier native design (inactive)

Everything below is historical design material for a separate native decoder.
Its proposed node, APIs, pool accounting and milestones are deferred, not part of
the implemented FFmpeg approach. Requirements and hypotheses here must not be
read as completed features or acceptance results.

Implement an opt-in NVDEC node that reduces decoder memory while preserving the
mixer's buffering and playback behaviour. Use fixed pools allocated during source
admission. Keep FFmpeg demuxing, packet edges and CUDA video-frame interoperability;
do not patch FFmpeg or change graph management to accommodate the decoder.

This is a design and validation plan, not a claim of measured native-decoder savings.
The comparison baseline must include the newly generated low-DPB demo assets.
The current practical ceiling reported after that optimisation is 192 sources
with 20 total outputs: 16 extra aux, two monitors and two program outputs. Use
this configuration to measure native-decoder savings and headroom, then test
additional output capacity only when measured headroom permits. This observation
does not establish a hardware encoder-session limit.
Current whole-GPU usage is approximately 20.7 GiB with the optimised inputs and
FFmpeg decoding. Measure again after replacing the HEVC decoders with native
NVDEC at the same 192-input / 20-output load. Keep assets, layouts, buffering,
encoder settings and background GPU workload identical; compare warmed steady
state and stress peaks as well as startup. Report whole-GPU usage, decoder-owned
allocations and the measured difference separately. Do not count the earlier
input-reencoding savings as native-decoder savings.

### Agreed constraints

- Implement HEVC/H.265 first, with a C++ codec abstraction from the start so
  H.264, AV1 and later codecs can reuse the decoder engine. Use the existing
  FFmpeg HEVC path as the behavioural reference. Implement other codecs later.
- The first implementation must support HEVC 10-bit 4:2:2 as well as the current
  4:2:0 inputs, preserving the mixer's existing 10-bit/4:2:2 feature. Validate
  native 4:2:2 decoding on hardware that advertises support for that format.
- Target the current Ada Lovelace/L4 host and its deployed driver/runtime first.
  Blackwell hardware validation is a later milestone. Keep 10-bit 4:2:2 support
  in the first implementation, with that hardware path explicitly unvalidated
  until the future Blackwell run. Older NVIDIA deployments are not the initial
  native-backend validation target; CPU and existing decoder paths stay unchanged.
- Improve VRAM use, throughput and efficiency through different internal resource
  management where validated. FFmpeg's allocation, copy and scheduling choices
  are not compatibility requirements.
- Fixed decode, mapped-output and application-frame pool capacities. No automatic
  pool growth during playback, including an unbounded FFmpeg buffer-pool allocator.
- Size each source for its initial sequence requirements plus measured processing
  headroom. Do not reserve for hypothetical larger future formats. A later change
  exceeding the admitted budget is an explicit source error; a deliberate restart
  is required to admit a larger budget.
- Preserve the decoded edge capacity of 3 and the aux playout delay of 80 ms at
  25 fps. A three-frame edge does not imply a three-surface decoder.
- Share the existing CUDA context. Keep all production processing on the GPU.
- Preserve existing AVP APIs and low-level nodes. Implement this work in the new
  native decoder, with graph selection through existing construction mechanisms.
- Preserve CPU decoding and every existing decoder selection/default. Native
  NVDEC is opt-in per source; it is not a global replacement for video decoding.
  CPU-only builds and runs must remain usable without CUDA/NVDEC headers, drivers
  or runtime libraries. Mixed CPU/FFmpeg/native-NVDEC graphs remain supported.
- Keep FFmpeg-specific workarounds in the existing decoder implementation. The
  native NVDEC engine has its own lifecycle and does not inherit those workarounds.
- Support hardware decode statistics only through `debug_decode_stats`, default
  `false`. Enabling it is a creation-time choice with explicit memory cost.
- Investigate removing the decoded-output copy as a separate, measured option.
- Test opaque CUDA arrays firsthand on the current GPU as a primary optimisation
  experiment, including actual decoding, pixel correctness and measured memory/
  copy costs. Build the compatible linear path alongside it. Decoder and storage
  selection remain explicit per source; do not force arrays on other decoders.
- Future CUarray support is optional and additive. Keep classical pitch-linear
  CUDA buffers as the default, with unchanged pointer/pitch semantics and existing
  consumer behaviour. Other devices and decoder backends must remain usable
  without opaque-array support.
- Keep the existing decoder selectable for explicit rollback. Never silently
  create an additional decoder or switch to a higher-memory fallback on exhaustion.

### What the node can improve

FFmpeg 8.1's NVDEC hwaccel chooses decode-surface counts from its frame-context
requirements. It does not expose independent decode and mapped-output counts.
Its normal output path maps a decoder output and copies it into a CUDA frame pool.
That pool can retain its peak allocation after downstream consumers release frames.

A native node can select fixed counts independently, expose their actual use, and
avoid FFmpeg's software decoder front end by using NVIDIA's parser. Savings in
driver allocations, output buffers and host parsing memory must be measured
separately. The previously discussed 1–2 GiB GPU saving is a hypothesis, not an
acceptance result, and must not be added to low-DPB asset savings without testing.

SDK 13 supports dynamic decode-surface allocation, but this design does not use it:
delaying allocations until playback can turn a successful startup into a later
OOM. `bMemoryOptimize` must not be enabled as an untested substitute for correct
surface counts. Any later experiment requires separate throughput and memory tests.

### HEVC compatibility contract

Required first-release formats include HEVC Main/Main10 4:2:0 and Main 4:2:2 10.
Publish compatible CUDA frames with NV12, P010 or P210 storage as appropriate,
preserving bit depth, chroma and colour/HDR metadata. Keep 4:2:2 through the source
path into the existing P210 compositor; do not silently reduce it to 4:2:0 to make
the decoder integrate. Validate NVIDIA's 16-bit 4:2:2 output layout against P210
bit placement, pitch and plane geometry before wrapping or copying it into frames.
Account for full-height 4:2:2 chroma and 16-bit sample storage in fixed-pool sizing.

NVIDIA's capability table lists HEVC Main 4:2:2 10/12 decoding on Blackwell, not
Ada. The current L4 can validate 4:2:0 decoding and the approximately 20.7 GiB
baseline comparison, but cannot validate native HEVC 4:2:2 decode. Query actual
codec/chroma/bit-depth and output-format capabilities at admission rather than
inferring them from the SDK version. A supported Blackwell-class GPU is a required
validation dependency for the 4:2:2 acceptance test. The mixer's existing v210 to
P210 path proves compositor format support, not compressed HEVC 4:2:2 capability.
Unsupported native selections must fail clearly while existing CPU/FFmpeg and raw
input paths remain available unchanged. Future Blackwell testing is explicitly
deferred and does not block implementing or validating the Ada milestone. Report
4:2:2 hardware validation as pending until it has run on capable hardware.

Match the observable behaviour of the existing FFmpeg HEVC node for streams
supported by the target GPU and admitted within the fixed limits. The first
experiment uses the current 1080p NV12 clips, but compatibility must also exercise
HEVC reference pictures and display reordering; the new no-B-frame demo clips
alone are insufficient. Inventory the existing path's supported HEVC profiles,
bit depths, chroma formats and metadata before implementation, and make any
unsupported case explicit rather than silently narrowing support to the demo.

Preserve decoded pixels and crop geometry, pixel format, colour/HDR metadata,
aspect ratio, timestamps, time base, frame duration and display order wherever
the current node exposes them. Cover Annex-B and length-prefixed input with
extradata, delayed-frame draining at EOF, looping, seek/flush, `discardUntil`,
EOF/hold behaviour and sequence changes within the admitted limits. For malformed
or missing packets, compare error reporting and recovery with the existing path;
do not require identical concealed pixels from different decoder front ends, but
record differences and verify recovery at a valid random-access point.

Use paired runs against the existing FFmpeg CUDA path with identical packets and
graph settings. Check output frames, metadata and timing independently of memory
and performance. Fixed-capacity admission and explicit rejection of oversized
sequence changes are intentional differences. Do not reproduce FFmpeg's internal
surface counts, buffer-pool growth, copies or packet-refeeding workarounds merely
for parity. Accept a different strategy when measurements demonstrate lower VRAM
use or better performance/efficiency without playback regressions, additional
stutter, unsafe lifetimes or allocations beyond the fixed budget.

### Memory ownership and admission

Account for three distinct resources:

| Resource | Owner and lifetime | Fixed sizing rule |
|---|---|---|
| Decode surfaces | NVDEC reference pictures and work in flight | Parser minimum plus measured decode headroom |
| Mapped outputs | NVDEC outputs mapped for copying or downstream reads | Copy concurrency, or the full mapped-frame lifetime budget |
| Application frames | CUDA output buffers shared by the graph | All distinct frames simultaneously retained downstream, plus processing headroom |

Determine codec, dimensions, bit depth, chroma and the parser's required surface
count before publishing the first frame. Resolve any automatic count once, at this
admission boundary; reject a configured count below the requirement. Preallocate
and initialise all application buffers and request the full fixed driver pools.

Warm the decoder before declaring it ready and record the observed GPU memory
increase. Driver internals may allocate lazily, so a configured surface count alone
does not prove a hard byte limit. Keep GPU-wide headroom for driver overhead, browser
surfaces and other mixer pools; this node cannot enforce a total-process VRAM cap.

Maintain an inventory of each owned slot with free, decoding, copying, published
and awaiting-GPU-completion states. Reuse a slot only when all frame references and
GPU uses have ended. Compute geometry and byte estimates with libav utilities and
actual CUDA pitch; report application allocation bytes separately from estimated
or externally measured driver memory.

Measure retention across the complete source path: the decoded edge, any frame
held by the producer, pacing and force-fps nodes, colour filters, main prewarm and
aux playout. Count unique buffers as well as references. Multiple aux buses may
reference the same picture, but asynchronous execution can retain different
pictures. Do not size the output pool from the edge capacity alone.

### Node and graph integration

Add `nvdec_video` under `src/nodes/nvdec/`, gated by CUDA and the required NVDEC headers
and runtime symbols. Use `NodeSISO<av::Packet, av::VideoFrame>` and the existing
decoder, time-base, flush and stream-metadata interfaces as appropriate. Use RAII
for parser, decoder, context references, mappings, frames and CUDA events.

Keep responsibilities separate without copying the existing decoder template:

Do not derive the native engine from the FFmpeg-specific `Decoder<>` template in
`src/nodes/decoders.cpp` or make its state machine a shared backend abstraction.
Implement the existing graph interfaces on the new node and reuse backend-neutral
utilities where appropriate. The shared C++ engine described below serves native
NVDEC codecs; it does not merge the legacy FFmpeg and native decoder lifecycles.

- A packet/parser adapter uses existing stream metadata and libav bitstream
  filters for Annex-B conversion where needed. Bound parser and display queues.
- A decoder owner creates the fixed pools, submits pictures and handles sequence
  changes against the admitted stream limits.
- Frame ownership integrates with `AVBufferRef` and a valid `AVHWFramesContext`.
  Frame references may outlive the node's processing thread; their owner must keep
  the decoder/context alive until release and GPU completion are safe.
- Node status publishes snapshots of counters without exposing mutable decoder
  state to the control thread.

Select the new node through existing graph-construction APIs and demo source
configuration, without changing existing public API contracts or low-level nodes.
Retain the existing default until validation passes. Preserve feature flags when
building both the binary and Python module; rebuild to regenerate node registration.
No control-protocol, sentinel or graph-manager changes are planned. If integration
requires changing an existing API or low-level node, identify that as a design
blocker instead of adding an incidental compatibility patch.

Keep native NVDEC build and runtime dependencies optional. Gate its source and
registration appropriately; missing NVDEC libraries or unsupported hardware must
not prevent unrelated CPU nodes or the application from starting. Resolve native
backend availability when that backend is selected, and report an explicit error
for that selection. Do not route existing `dec_video` nodes to NVDEC globally or
silently replace CPU decoding. Backend selection and compressed codec selection
are separate: HEVC can still use the existing CPU or FFmpeg hardware path.

### C++ codec abstraction

Keep the graph node named `nvdec_video` and its packet/frame contract independent
of the compressed codec. Select the codec implementation from the upstream
`InputStreamMetadata` / `AVCodecParameters.codec_id`, consistent with the existing
decoder's metadata discovery. Initially register only HEVC; other codecs return
an explicit unsupported-codec error until their implementations are available.

Separate these responsibilities within the native decoder implementation:

- **Node adapter:** existing graph interfaces, packet delivery, playback control,
  frame publication and status. No HEVC-specific packet interpretation here.
- **Shared NVDEC engine:** CUDA context and stream handling, parser callback
  plumbing, decoder lifetime, fixed pools, admission, backpressure, GPU completion,
  frame ownership and debug statistics. Implement each once for all codecs.
- **Codec adapter:** FFmpeg/NVDEC codec identifiers, bitstream-filter and extradata
  setup, codec-specific metadata and interpretation of sequence requirements.
  Reuse NVIDIA's parser and libav facilities rather than writing bitstream parsers.
  Expose the requirements to the shared engine, which enforces the fixed budgets.

Use a small C++ descriptor/policy and add virtual hooks only for behaviour that
needs them. A different codec identifier alone does not justify a subclass or a
copy of the decoder node. Keep this abstraction local to the new implementation;
reuse existing graph interfaces rather than introducing a framework-wide decoder
hierarchy or rewriting the FFmpeg decoder.

Implement the HEVC adapter first. H.264 and AV1 should subsequently add codec
setup, any genuinely distinct behaviour and compatibility tests without duplicating
pool management or changing graph configuration and output ownership. Do not
assume Annex-B framing, HEVC metadata, or a one-to-one packet/picture/output
relationship in the shared engine. Keep codec selection separate from copy,
mapped-output and future opaque-array storage modes. No empty future codec
implementations are needed in the first change.

### Code quality and static analysis

Keep the implementation together in `src/nodes/nvdec/`, with its build inclusion
gated as an optional native backend. Use concise, descriptive file and type names.
A starting layout is `node.cpp` for graph integration, `decoder.hpp/.cpp` for the
shared engine, `frame_pool.hpp/.cpp` for buffer ownership, `codec.hpp` for the codec
contract and `hevc.cpp` for its first implementation. Add files only when there is
a cohesive responsibility to separate. Keep implementation types in an `nvdec`
namespace, using names such as `Decoder`, `FramePool` and `Codec` rather than long
prefixes, vague manager/helper names or abbreviations that obscure their purpose.

Keep each new `.cpp` file to roughly 600 lines or less. Split by responsibility:
node integration, shared decode engine, fixed-pool/frame ownership and codec
adaptation, as the implementation warrants. Keep headers focused; do not move
large implementations into headers merely to satisfy the file-size target.

Use RAII, explicit ownership and small cohesive functions. Share codec-independent
logic rather than copying it for HEVC, H.264 or AV1. Prefer existing libav/avcpp
utilities and repository conventions. Avoid unrelated refactoring or speculative
class hierarchies. Apply DRY within files as well as across codecs: extract shared
allocation, validation, cleanup and metadata handling when they represent the
same operation, while keeping genuinely codec-specific behaviour explicit.

Run Clang static analysis on the new C++ translation units using the real build
configuration and feature flags, through the project's existing analysis setup
or `clang-tidy` with Clang Static Analyzer checks. Fix findings introduced by this
implementation; do not blanket-suppress them to obtain a clean report. Record
which files/checks ran and any tooling limitations. Static analysis complements
the CPU regression checks and GPU ownership, playback and stress tests.

### Configuration contract

These are proposed node parameters; settle names against repository conventions
before implementation. They are creation-time parameters unless stated otherwise.

| Parameter | Meaning |
|---|---|
| `hwaccel` | Existing shared CUDA device/context |
| `decode_surfaces` | Explicit fixed count, or automatic resolution at initial sequence admission |
| `output_frames` | Fixed application-frame capacity in copy mode; selected from retention measurements |
| `mapped_output_surfaces` | Fixed number of simultaneously mapped decoder outputs |
| `output_mode` | `copy` initially; experimental `mapped` after ownership tests |
| `max_width`, `max_height` | Admitted coded-size envelope; default to the initial sequence |
| `debug_decode_stats` | Enable supported hardware statistics and their additional buffers; default `false` |

Do not invent a small universal default for `output_frames`. Establish a supported
configuration for the actual graph before promoting this node to normal use.
Validate every configured limit before admission and report the resolved counts.

### Exhaustion and stream changes

Reserve an output slot before submitting work that will require one. Keep packet
and display queues bounded, and ensure that waiting for a free output cannot hold
a lock needed by frame release, parser callbacks, flush or shutdown.

Exhaustion applies bounded backpressure and reports the exhausted pool. It must
not overwrite a referenced surface, allocate beyond the limit, drop a reference
picture, or invoke a hidden copy fallback. A sustained stall becomes an explicit
source error under the existing error policy. Backpressure can still cause visible
repeats, so it is a validation failure under the supported workload, not a cure for
an undersized pool.

Accept stream changes only when codec capabilities and every admitted memory limit
remain satisfied. Investigate `cuvidReconfigureDecoder()` for compatible changes.
Reject changes requiring more surfaces or larger storage, with the old and requested
requirements in status. A deliberate restart can admit a larger budget; it must
release the old generation before allocating its replacement to avoid double usage.

Use sequence/generation identities so frames outstanding through flush, seek or
restart cannot return a slot to a different pool generation. Preserve timestamp
ordering, loop continuity, EOF/hold behaviour and `discardUntil`. Establish the
observable requirements of `flush_magic` in the existing path, but keep its packet
refeeding algorithm and related FFmpeg-specific state in `src/nodes/decoders.cpp`.
Implement native parser/decoder drain, reset and seek handling directly. Verify
the requested target frame, delayed frames, absence of unintended duplicates and
loop continuity. Use the existing path to identify required playback results,
not as a state machine to inherit. The native node need not expose FFmpeg-specific
workaround options. Leave existing FFmpeg nodes and their behaviour unchanged.

### Copy mode and mapped mode

Start with a bounded GPU-to-GPU copy into preallocated application frames. This
decouples decoder reference recycling from consumers that hold a displayed picture
and provides the simplest compatibility baseline. Record a producer-ready event
and ensure consumers wait correctly across CUDA streams.

Then test mapped output: wrap the mapped pointer in a frame whose ownership keeps
the mapping and the required decoder picture alive. Unmap only after the last
reference and all GPU reads finish. Audit the existing compositor, filters and
encoder for this lifetime contract; CPU reference release alone is insufficient
for asynchronous reads. Never globally synchronise the whole CUDA context per frame.

Mapped output removes one application copy but can require additional decoder and
mapped-output surfaces. Compare total VRAM, memory bandwidth, throughput and latency
against copy mode at identical graph buffering. Keep copy mode if it performs better
or the lifetime contract cannot be proven. Aux composition still creates its own
output pictures in either mode.

### Debug decode statistics

With `debug_decode_stats=false`, leave the NVIDIA statistics feature disabled and
allocate no statistics buffers. Lightweight allocation and stall counters remain
available independently of hardware decode statistics.

When enabled, check `bIsDecodeStatsSupported`, enable
`cuvidDecodeFeature_DecStats` at decoder creation, and include its buffer cost in
admission. Expose per-block QP, coding type and motion vectors through a bounded
debug retrieval path. Define ownership and timestamp association so statistics
cannot outlive their buffers. Do not copy every frame's statistics to the CPU unless
debug retrieval requests it. Unsupported hardware returns a clear configuration
error. Changing this option requires deliberate decoder recreation.

### Implementation and validation stages

1. **Establish the baseline.** Use the newly generated low-DPB clips. Record the same
   source mix, aux layouts, prewarm configuration and buffer sizes for every run.
   Sample startup, steady state and cuts separately. Capture decoder errors,
   source repeats, compositor deadlines, encoder output and browser playback.
2. **Measure retention.** Use existing diagnostics and instrumentation in the new
   decoder or test harness for queued/held frames and stable buffer identities;
   do not modify existing low-level nodes for telemetry. Distinguish a consistent snapshot from
   samples taken at different times. Observe pool allocations, including free
   buffers retained at their peak count; `queues.json` alone is insufficient.
3. **Build the shared engine and HEVC fixed-pool copy prototype.** Implement the
   codec boundary with the HEVC adapter first. Use representative HEVC clips for
   the first controlled experiment. Verify parser requirements, metadata, timestamps,
   output ownership and bounded allocation before integrating many sources.
4. **Integrate and test HEVC behaviour against FFmpeg.** Cover the supported
   HEVC profile/format matrix, reference and reordered pictures, metadata, looping,
   EOF, seeking, flush, corrupt/missing packets, shutdown with outstanding frames,
   and permitted/rejected sequence changes. Keep the old decoder explicitly selectable.
   Verify existing CPU decoding in a CPU-only build/run without NVIDIA libraries,
   unchanged existing decoder configurations, the existing FFmpeg NVDEC path's
   pixel/metadata/timestamp correctness and performance, and a mixed CPU/native-NVDEC graph
   on the GPU host. Native backend failure must remain isolated under existing
   node/group error policies; it must not change global decoder selection.
5. **Compare mapped output.** Deliberately slow an aux consumer, retain old frames,
   and exercise different CUDA streams. Check for corruption, deadlocks, additional
   repeats and late allocations before considering it a normal mode.
6. **Add opt-in statistics.** Verify disabled/enabled memory differences, capability
   errors and bounded retrieval. Run performance comparisons with statistics disabled.
7. **Scale and soak.** Progress from one stream to representative multi-stream
   loads and the full mixed show. Validate the 20-total-output baseline, then add
   outputs incrementally only if measured headroom permits. Report total outputs
   and extra aux separately. Include at least 30 minutes of cuts/layout changes, repeated
   loop boundaries and multiple cold starts. Report maxima and percentiles.

Tests must demonstrate stable allocated pool counts after admission, correct
pixels/colour/timestamps, no reference corruption, and no additional missed
deadlines or repeats attributable to the decoder. Compare streams under the same
workload; browser-source repeats and network jitter are separate measurements.
Measure savings/headroom against the approximately 20.7 GiB optimised FFmpeg
baseline before deciding to replace the existing path. A smaller kernel/copy
count alone is not sufficient.

Run GPU validation on the configured NVIDIA host. Preserve CUDA, neural, NVCC/PTX,
TensorRT and DRM/GL build settings, leave FRUC disabled, and verify the intended
FFmpeg libraries are loaded. Run the relevant Python mixer tests after builder
changes. Keep the web UI backend and registration available during remote testing.

### Opaque CUDA array experiment and integration

SDK 13.1 can decode into application-allocated opaque CUDA arrays with
`cuvidDecodePictureAsync()` and stream-ordered execution. Keep ownership and pool
accounting separable from pointer/pitch representation. Exercise that API in a
real L4 decode test early, rather than accepting documentation or a capability
query as proof. Compare decoded pixels, timestamps, fixed-pool memory, copies
and throughput with linear native output and FFmpeg NVDEC on the same assets.

That phase requires capability checks, CUDA-array sampling in the compositor and
an explicit path for filters that currently expect pitch-linear frames. Never
disguise an array handle as a normal CUDA plane pointer. Preserve fixed capacities
and explicit synchronisation. Benchmark it end to end; the SDK's direct-transcode
benefits do not automatically transfer to a many-input compositor.

Preserve both storage representations independently:

- Linear frames retain normal device pointers, per-plane pitches, hardware-frame
  metadata and ownership. Existing consumers must not need an array descriptor to
  process them. `AV_PIX_FMT_CUDA` alone must never imply array-backed storage.
- Array frames carry an explicit, versioned storage descriptor and ownership;
  only consumers supporting that descriptor and its format may read them. Keep
  existing DMA-BUF descriptors compatible when extending the representation.
- Enable arrays only for an explicitly selected path after checking the device,
  runtime, pixel format and downstream consumers. Existing configurations and
  devices without array support continue using linear frames. An explicitly
  requested unsupported array path fails clearly. Do not silently allocate a
  second linear pool or convert storage after runtime exhaustion.
- If a future graph needs array-to-linear conversion, make it an explicit GPU
  operation with its fixed buffers included in admission. No CPU round trips or
  implicit conversion inserted into existing linear paths.

Before enabling the array mode, regression-test ordinary linear CUDA decoding,
filtering, compositing and encoding, existing DMA-BUF inputs, and mixed linear/
array inputs on supported consumers. Verify CPU-only and non-array device paths
still work. Compare linear-path memory, throughput and frame lifetime against the
pre-array baseline; adding array support must not change its default allocation
or introduce extra copies or synchronisation into that path.

Reuse the existing DMA-BUF array-frame representation rather than introducing a
new graph edge type. `drm_prime_to_cuda` already puts a reference-counted import
owner in `AVFrame.buf[0]` and a tagged `TextureFrameDesc` in `opaque_ref`. The
compositor reads that descriptor; the value in `data[0]` is a validity placeholder,
not a pointer that generic CUDA filters may dereference. The current descriptor
and texture kernels support packed RGB/RGBA only. Extend them with an explicit
format/version and per-plane texture information for NV12/P010, preserving existing
browser inputs. NVDEC supplies the fixed-pool frame owner instead of a DMA-BUF
import owner. Routing must keep array-backed frames away from unaware pixel readers.
This future phase requires a separate scope decision because extending existing
texture consumers would change low-level nodes. The current implementation must
preserve them and publish compatible pitch-linear CUDA frames.

### Decisions after measurement

- After measuring the native decoder at the baseline load, decide whether the
  resulting headroom justifies adoption and which higher output count to test.
- Whether mapped output merits promotion after the fixed-pool copy comparison.

### References

- [NVIDIA NVDEC capability table](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvdec-application-note/index.html): HEVC 4:2:2 support by GPU architecture.
- [NVIDIA SDK 13.0 overview](https://developer.nvidia.com/blog/nvidia-video-codec-sdk-13-0-powered-by-nvidia-blackwell/): Blackwell 4:2:2 decoding and NV16/P216 decoder output storage.
- [NVIDIA NVDEC programming guide](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.1/nvdec-video-decoder-api-prog-guide/index.html): parser requirements, allocation, reconfiguration, debug statistics and opaque surfaces.
- [NVIDIA SDK 13.1 overview](https://developer.nvidia.com/blog/nvidia-video-codec-sdk-13-1-zero-copy-transcode-av1-b-frames-and-frame-accurate-seek/): array sharing and decode statistics.
- [FFmpeg 8.1 NVDEC implementation](https://github.com/FFmpeg/FFmpeg/blob/n8.1/libavcodec/nvdec.c): surface counts and mapped/copied output paths.
- [FFmpeg 8.1 CUVID implementation](https://github.com/FFmpeg/FFmpeg/blob/n8.1/libavcodec/cuviddec.c): existing parser-backed decoder and exposed options.
