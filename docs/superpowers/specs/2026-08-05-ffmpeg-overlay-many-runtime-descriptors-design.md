# FFmpeg CUDA Overlay Runtime Descriptors

## Goal

Replace `ffmpeg-patches/0019-overlay_many_cuda-performance-improvements.patch`
with an independently reviewable `overlay_many_cuda` patch applied after patch
`0018`. Fold the overlay-related correctness changes from the current `0020`
into that patch while eliminating the generated kernel matrix for input counts
2 through 16.

Move the unrelated `scale_cuda` texture-addressing, Lanczos edge, and clipping
changes into a separate replacement `0020` patch.

The replacement targets the FFmpeg `n7.1.3` base documented in
`ffmpeg-patches/README.md`. It must remain a zero-copy CUDA pipeline and use
FFmpeg's existing CUDA context and stream.

FFmpeg 9 is a feature-comparison baseline only; it is not the base for these
patches. The overlay patch should be structured and documented for a future
upstream review, but direct submission to current FFmpeg will require a rebase.

## Patch Shape

The existing `0019` and `0020` files will be replaced by two patches with
independent subjects and scopes.

The new `0019` contains:

- the complete `overlay_many_cuda` replacement;
- deletion of `vf_overlay_many_cuda_generator.c`;
- addition of a normal, committed `vf_overlay_many_cuda.cu` source;
- the corresponding Makefile cleanup;
- all overlay bounds, output-frame-context, and edge fixes currently carried by
  patch `0020`.

The new `0020` contains only the `scale_cuda` texture-addressing, Lanczos edge,
and output-clipping fixes. The overlay patch must not depend on it, so the
overlay change can be reviewed independently and rebased upstream on its own.

## Overlay Kernel Interface

The host and CUDA source will share a fixed-layout descriptor ABI. It contains
destination planes, main-input planes, dimensions, the active overlay count,
and storage for at most 15 overlay plane descriptors. Fields use fixed-width
integers, explicit padding, and compile-time size and offset checks.

The complete launch packet is expected to be about 1.1 KiB, below the legacy
4 KiB CUDA kernel-parameter limit. Kernels take it as a
`__grid_constant__ const` value and index its overlay array using the runtime
layer count. This prevents a per-thread copy of the aggregate while retaining a
single launch with no descriptor allocation or upload. Validation will inspect
generated PTX and `ptxas` output for local-memory copies or spills.

This design requires CUDA Toolkit 11.7 or newer and CUDA compute capability 7.0
or newer. The hardware requirement refers to Volta-generation hardware or
newer, not the CUDA 7 toolkit. `__grid_constant__` is mandatory rather than a
conditional optimization. Configure and runtime behavior must report these
requirements clearly rather than failing with an unexplained compiler or
module-load error.

The PTX exports exactly three entry points:

1. YUV420P main plus YUVA420P overlays;
2. YUV444P main plus YUVA444P overlays;
3. YUV420P main plus YUVA444P overlays.

Each entry handles one base plus 1 to 15 overlays using a uniform runtime loop.
No NVRTC, device-side launch, or per-frame device allocation is introduced.

If CUDA-event benchmarks show that an exact one- or two-overlay specialization
is more than 3 percent faster than its generic format kernel, that
specialization may be included. It must use the same descriptor ABI. No other
count specialization is permitted, limiting the exported surface to three
generic kernels and at most six measured fast paths.

## Data Flow and Performance

For each output frame, the filter:

1. obtains all frames through the existing frame synchronizer;
2. omits unavailable secondary frames while preserving the z-order of the
   available overlays;
3. forwards the main frame unchanged only when no overlay is available;
4. otherwise allocates a distinct pooled CUDA output frame and copies frame
   properties;
5. packs device addresses and pitches into the fixed launch descriptor;
6. launches one fused kernel on FFmpeg's supplied CUDA stream;
7. forwards the output without synchronizing the stream.

Missing secondary inputs are handled independently. One missing overlay must
not suppress other available overlays. A missing main frame remains an error.
The filter never writes into the main input frame, even when that frame is
writable; keeping source and destination distinct preserves the descriptor's
non-aliasing contract.

All three format combinations use one launch and one output write. In
particular, the mixed YUV420P/YUVA444P path will no longer use separate Y, U,
and V launches. Even-coordinate threads perform the chroma work for 4:2:0
outputs.

The following optimizations from the existing patches are retained:

- separate source and destination frames, allowing read-only input pointers;
- fused Y, U, and V processing;
- integer alpha composition;
- `restrict`-compatible, non-aliasing source access;
- one CUDA context push/pop around the work;
- asynchronous execution on the existing stream;
- explicit bounds checks for partial edge blocks;
- a dedicated output hardware-frame context matching the main software format.

## Arithmetic and Edge Semantics

Eight-bit straight-alpha blending uses exact rounded division by 255:

```c
sum = alpha * foreground + (255 - alpha) * background;
rounded = sum + 128;
result = (rounded + (rounded >> 8)) >> 8;
```

This replaces the existing `>> 8` division-by-256 approximation. Alpha zero
must return the background exactly; alpha 255 must return the foreground
exactly. The result is rounded back to 8-bit after each overlay layer. Generic
and specialized kernels must therefore produce the same result as sequential
8-bit composition; they must not retain hidden higher precision across layers.

For YUVA420P overlays, chroma alpha is the rounded average of the valid luma
alpha samples in the corresponding 2x2 block. For YUVA444P overlays composed
onto YUV420P, every layer is composed independently at all four full-resolution
chroma positions. The four final results are averaged only after all layers
have been applied. Downsampling each overlay before composition is not an
equivalent operation and is not permitted.

Odd widths and heights use ceiling chroma dimensions. Edge 2x2 blocks include
only valid samples, avoiding out-of-bounds reads and avoiding a bias from
invented zero-valued samples.

## Validation and Error Handling

Host validation rejects unsupported format combinations, counts outside 2 to
16 total inputs, missing required planes, non-positive pitches, descriptor
overflow, or aliasing that violates the kernel's read-only assumptions.

CUDA module-load, symbol-lookup, context, allocation, and launch failures are
propagated. Context pop and frame cleanup still run on failure, preserving the
first error. A failed launch never forwards the newly allocated output frame.

Verification consists of:

- applying the complete patch series to a clean FFmpeg `n7.1.3` checkout;
- confirming that `0019` changes only `overlay_many_cuda` and its build wiring;
- confirming that `0020` changes only `scale_cuda`;
- checking that PTX exports the three generic overlay entry points plus only
  benchmark-approved one- or two-overlay specializations;
- checking the launch descriptor size and host/device offsets;
- exhaustive unit verification of the 8-bit blend formula against rounded
  integer division by 255;
- GPU comparisons against a CPU reference for all three format combinations,
  1, 2, 4, 8, and 15 overlays, alpha endpoints and random alpha, padded
  pitches, partial CUDA blocks, and odd dimensions;
- steady-state end-to-end filter benchmarks against the current patch stack at
  1080p and 2160p, with frame pools prewarmed and identical graphs;
- separate CUDA-event kernel timings used diagnostically rather than as the
  acceptance metric;
- checking `ptxas` register, spill, and local-memory reports.

CUDA compilation and runtime validation must run on the configured NVIDIA
host, not on a local machine without `nvidia-smi`. A measured regression above
3 percent for 1 to 3 overlays or 5 percent for 4 to 15 overlays blocks removal
of the old implementation until the cause is understood. The gate uses complete
steady-state filter cost, including descriptor packing, pooled output-frame
acquisition, and kernel execution. One- and two-overlay specializations are
accepted only under the benchmark rule above; a regression at another count
must be fixed in the generic implementation.

## Non-goals

The replacement does not add positioning, scaling, cropping, runtime layer
updates, more than 15 overlays, new pixel formats, CPU fallbacks, or CUDA
upload/download steps. It preserves the existing full-frame aligned-overlay
contract.
