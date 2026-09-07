# Mixer graph optimization

Approved scope: reduce actual graph/code complexity and CPU/GPU consumption against the current implementation, preserving frame continuity, output quality, low click-to-picture delay and all transition behavior. A grouped WebUI is a separate readability improvement and does not count as a smaller graph.

## Architecture

Real inputs fan out to the two permanent CUDA compositors. Each scene specifies per-source crop and destination rectangles. The compositor scales directly into the fixed output canvas, using each selected frame's dimensions and pitch. Equal-size copies retain the existing copy path. Dimension changes do not rebuild the graph; old queued frames retain their own references. Pixel formats remain explicitly validated.

Python graph construction stays in `avpmixer`; scene generation stays in the demo. C++ owns runtime routing, shared playout, preview readiness, Cut/Fade/media-Wipe and exact-picture interruption. Explicit FFmpeg preprocessing remains supported for existing callers. No framework graph-management or streaming decoder changes.

The first comparison retains input normalization and all FPS/timing stages. This isolates replacement of 62 geometry filters and their router with 16 input fanouts. Earlier estimates that also removed normalization are not the count for this first stage. Removing normalization requires a separate before/after validation within the same PR. All FPS stages remain.

## Verification and comparison

Freeze the complete uncommitted baseline source and runnable native/Python artifacts outside Git. Record fixture hashes, build flags, hardware and runtime options privately; publish only reproducible public conditions and results. Use the same 16 fixtures, host, 60 fps, output size, encoder/bitrate, latency, preview client, warm-up and measurement duration before and after. Run trials sequentially so benchmark control scripts cannot interfere.

Compare idle fullscreen and 4/8/16 grids plus a repeatable transition sequence. Report node and defined-queue counts; process CPU cores, GPU utilization, VRAM; per-tile duplicate/drop counts and output cadence; click-to-visible-picture min/max/mean/median/p95. Command acknowledgments are not display latency. Recording is either enabled for both sides or neither. Repeat trials and distinguish host-wide GPU use from process CPU use.

Tests cover fixed and changing resolution, aspect fit, crop/clipping, chroma alignment, source disappearance/recovery, preview prewarm, all Cut/Fade/Wipe interruption pairs, and explicit filter compatibility. Quality checks include patterned images and the same fixture frame at the same layout. No savings or frame-perfect claim until measured.

## Implementation sequence

1. Freeze baseline and run existing tests; collect clean comparable measurements.
2. Add independently testable rectangle resolution and CUDA resizing to `cuda_rect_overlay`, preserving legacy copy behavior.
3. Add filter-free source support to the existing mixer builder/orchestrator and simplify demo construction; retain A/B and transition prewarm.
4. Run CPU tests and remote GPU build/pixel tests, then same-workload repeated comparisons. Investigate regressions before further stage removal.
5. Render a generic grouped WebUI overview with expandable internals and bundled connections; retain the complete diagnostic graph. Validate accounting against native node/queue data.
6. Update public demo documentation with measured before/after results; keep media assets outside Git and combine changes into the requested PR.

## Second review

The first live build has 138 nodes and 156 defined queues; this is not a minimum. Sixteen normalization filters still enlarge all 640×360 fixtures to 1920×1080 before per-layer resizing. Removing them is the next candidate (122 nodes), subject to decoded-format support and parity for nested aspect padding. All input FPS stages remain, as explicitly requested.

A single existing router could replace 16 fanouts (15 fewer nodes), but its homogeneous-format contract conflicts with arbitrary input-size changes. Do not generalize it or combine this with the other removals without separate evidence. Keep A/B rendering and exact-picture snapshot functionality. The observed approximately 39% aggregate GPU utilization is not an established requirement.

Inline expansion of all input chains made the overview unreadable. Group navigation now uses a focused internal graph with bundled external boundaries and breadcrumbs; a family picker selects one input chain. Keep labels readable, serialize layout rebuilds and test real browser navigation rather than only graph accounting.

## Current user constraint

All FPS filters must remain. The authorized next target is 122 nodes by removing only the 16 normalization stages from the 16-input demo. Input timing/pacing, A/B prewarm, snapshots and transitions remain unchanged. Preserve the existing source framing through a virtual source canvas: resolve the same letterbox geometry without allocating an intermediate 1920×1080 frame. The larger-catalogue router retains normalization for its homogeneous-format contract.

## WebUI screenshot acceptance

Prioritize practical readability, including changes to node rendering and layout when needed. The grouped overview and every focused group must fit a maximized 1920×1080 browser view without overlapping boxes or text, clipped nodes, or edges crossing node interiors. Use measured, unscaled node/socket bounds and orthogonal routes. Mode switches must clear stale drilldown state. Synthetic group/boundary identities must never be sent as native node inspection commands. Verify all 16 input chains, all groups, repeated mode changes and native-count conservation in a browser. Keep a complete graph view for zoomed inspection; the grouped overview is the primary demo screenshot.
