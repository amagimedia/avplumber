# FFmpeg Patch Stack Consolidation Design

## Goal

Replace the 22-patch FFmpeg series in `deps/ffmpeg-patches` with seven
feature-family patches. The rewrite must preserve every existing FFmpeg source
change except the deliberately removed FFmpeg Whisper filter, which will be
implemented as an AVPlumber node instead.

This is a patch-history cleanup, not a CUDA, decoder, transport, or input-device
behavior change.

## Baseline and Target

The series applies to upstream FFmpeg tag `n7.1.5`, commit
`3a0867c2bfda4a4d4309ca1a8cbdc6175e67f587`.

Applying the current 22 patches on the Fedora validation host produces tree
`064865bafac8bc5e0b0ec0cc8081f6675d616472`.

The consolidation target is that tree with only current patch
`0012-port-whisper-STT-from-ffmpeg8.patch` reverted. The resulting target tree
is `52361f7251069ef74fbb41460e6e1b65d6f9947c`. Relative to the current tree,
the intentional removal is exactly:

- the Whisper configure options and dependency checks in `configure`;
- the Whisper object entry in `libavfilter/Makefile`;
- the Whisper filter declaration in `libavfilter/allfilters.c`;
- `libavfilter/af_whisper.c`.

No AVPlumber runtime code depends on the FFmpeg Whisper filter. Two historical
pipeline SVGs mention Whisper; they will be adjusted to show it as a separate
AVPlumber node rather than part of `filter_audio`. Implementing that node is
outside this change.

## New Series

The patches remain an ordered `git am` series, but each patch represents an
independently understandable feature family.

| New patch | Current patches consolidated | Responsibility |
| --- | --- | --- |
| `0001-swscale-aarch64-argb-yuva420p.patch` | `0001` | AArch64 fast unscaled ARGB-to-YUVA420P conversion |
| `0002-avfilter-cuda-composition-suite.patch` | `0002`, `0003`, `0005`, `0006`, `0009`, `0013`, `0014`, `0015`, `0017`, `0018`, `0019`, `0020`, `0022`, `0023` | CUDA pad, convert, crop, overlay, overlay-many, scale edge handling, transitions, procedural wipes, and their shared build/CUDA compatibility changes |
| `0003-avfilter-npp-cuda13-compat.patch` | `0021` | CUDA 13 stream-context compatibility for NPP filters |
| `0004-avcodec-nvdec-intra.patch` | `0004` | NVDEC intra-only handling |
| `0005-avformat-rtp-rfc4175.patch` | `0007`, `0008` | RFC 4175 4:2:0 handling and incomplete-frame skipping |
| `0006-avdevice-v4l2-compat.patch` | `0010` | V4L2 compatibility change |
| `0007-avdevice-ndi-v5.patch` | `0011` | NDI v5 input/output build registration and documentation |

Current Whisper patch `0012` has no replacement. Intermediate names such as
“missing files,” “continued,” and “fix for FFmpeg 7” disappear because their
changes become part of the complete owning feature.

NPP and NVDEC remain outside the CUDA composition patch. They have separate
dependencies and runtime roles, and keeping them independent makes the series
reviewable without fragmenting one feature across multiple patches.

## Provenance

Each new patch message will explain the complete feature and list the original
FFmpeg commit identifiers it replaces. Where a patch combines work by multiple
real contributors, its message will retain contributor trailers. Synthetic
local build identities will not be presented as human authors.

The repository history continues to preserve the original exported patch files,
but a reviewer should not need that history to understand the new series.

## Documentation and Verification Helper

`deps/ffmpeg-patches/README.md` will be rewritten around the seven categories.
It will remove the stale reference to nonexistent patch `0016`, document the
intentional Whisper removal, and state the exact base and target tree hashes.

A verification script beside the patches will accept an FFmpeg checkout,
create an isolated temporary worktree at the documented base commit, apply the
ordered series, and fail unless the resulting Git tree is exactly
`52361f7251069ef74fbb41460e6e1b65d6f9947c`. It must not alter the caller's
branch or contain machine-specific paths.

## Validation

All integration validation runs on the user-provided Fedora NVIDIA host.

1. Apply the current 22-patch series to clean FFmpeg `n7.1.5` and retain its
   tree hash as the baseline.
2. Apply the seven-patch series through the verification helper and require the
   documented target tree hash.
3. Compare the old and new trees and require that their only differences are
   the four listed Whisper integration changes.
4. Build the patched FFmpeg and AVPlumber CUDA-overlay container from a clean
   Docker build context.
5. Verify the expected CUDA filters and transition commands are registered.
6. Run the complete CUDA overlay matrix for every supported mode and overlay
   count 1 through 15.
7. Build or run the mixer validation path that exercises transition fades and
   procedural wipes.

The exact tree check proves retention for optional features that the basketball
container does not enable, including NPP, NDI, and V4L2. The clean container
build and GPU matrix prove the active CUDA composition path.

## Completion Criteria

The cleanup is complete only when:

- exactly seven `.patch` files remain;
- clean `git am` succeeds in filename order on the documented FFmpeg base;
- the result has the documented target tree hash;
- the old-to-new diff contains only the approved Whisper removal;
- README and pipeline diagrams describe the new boundaries accurately;
- Fedora-host build and CUDA runtime validation pass;
- no private host, key, credential, or instance path appears in committed files.
