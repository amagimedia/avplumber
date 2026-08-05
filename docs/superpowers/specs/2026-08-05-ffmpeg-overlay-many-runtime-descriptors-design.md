# FFmpeg CUDA Overlay Patch Consolidation

## Goal

Replace `ffmpeg-patches/0019-overlay_many_cuda-performance-improvements.patch`
and `ffmpeg-patches/0020-ffmpeg-match-host-cuda-overlay-edge-handling.patch`
with one patch applied after patch `0018`. The replacement must retain the
useful performance and correctness changes while eliminating the generated
kernel matrix for input counts 2 through 16.

The replacement targets the FFmpeg `n7.1.3` base documented in
`ffmpeg-patches/README.md`. It must remain a zero-copy CUDA pipeline and use
FFmpeg's existing CUDA context and stream.

## Patch Shape

The existing `0019` and `0020` files will be removed and replaced by one new
`0019` patch. It will contain:

- the complete `overlay_many_cuda` replacement;
- deletion of `vf_overlay_many_cuda_generator.c`;
- addition of a normal, committed `vf_overlay_many_cuda.cu` source;
- the corresponding Makefile cleanup;
- the `scale_cuda` texture-addressing, Lanczos edge, and output-clipping fixes
  currently carried by patch `0020`.

Keeping the `scale_cuda` changes in the replacement is intentional: the goal is
one behavioral replacement for both old patches, not merely an overlay-only
squash.

## Overlay Kernel Interface

The host and CUDA source will share a fixed-layout descriptor ABI. It contains
destination planes, main-input planes, dimensions, the active overlay count,
and storage for at most 15 overlay plane descriptors. Fields use fixed-width
integers, explicit padding, and compile-time size and offset checks.

The complete launch packet is expected to be about 1.1 KiB, below the legacy
4 KiB CUDA kernel-parameter limit. Overlay descriptors use constant indices in
device code so the compiler can issue direct parameter loads rather than copy
or dynamically index the aggregate in per-thread local memory. Validation will
inspect generated PTX and `ptxas` output for local-memory copies or spills.

The PTX exports exactly three entry points:

1. YUV420P main plus YUVA420P overlays;
2. YUV444P main plus YUVA444P overlays;
3. YUV420P main plus YUVA444P overlays.

Each entry handles one base plus 1 to 15 overlays. The device source expresses
the 15 possible layer operations once per format path, using uniform
count-controlled exits between layers. This keeps the current compile-time
unrolling advantage without emitting a separate function signature for every
input count. No NVRTC, device-side launch, or per-frame device allocation is
introduced.

## Data Flow and Performance

For each output frame, the filter:

1. obtains all frames through the existing frame synchronizer;
2. allocates a distinct CUDA output frame and copies frame properties;
3. packs device addresses and pitches into the fixed launch descriptor;
4. launches one fused kernel on FFmpeg's supplied CUDA stream;
5. forwards the output without synchronizing the stream.

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
exactly.

For YUVA420P overlays, chroma alpha is the rounded average of the valid luma
alpha samples in the corresponding 2x2 block. For YUVA444P overlays composed
onto YUV420P, four full-resolution chroma results are composed in layer order
and then averaged for the output chroma sample.

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
- confirming that the replacement patch is the only `0019` and no `0020`
  remains;
- checking that PTX exports exactly the three overlay entry points;
- checking the launch descriptor size and host/device offsets;
- exhaustive unit verification of the 8-bit blend formula against rounded
  integer division by 255;
- GPU comparisons against a CPU reference for all three format combinations,
  1, 2, 4, 8, and 15 overlays, alpha endpoints and random alpha, padded
  pitches, partial CUDA blocks, and odd dimensions;
- CUDA-event benchmarks against the current patch stack at 1080p and 2160p;
- checking `ptxas` register, spill, and local-memory reports.

CUDA compilation and runtime validation must run on the configured NVIDIA
host, not on a local machine without `nvidia-smi`. A measured regression above
3 percent for 1 to 3 overlays or 5 percent for 4 to 15 overlays blocks removal
of the old implementation until the cause is understood.

## Non-goals

The replacement does not add positioning, scaling, cropping, runtime layer
updates, more than 15 overlays, new pixel formats, CPU fallbacks, or CUDA
upload/download steps. It preserves the existing full-frame aligned-overlay
contract.
