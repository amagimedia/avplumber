# Unbounded YOLO Segmentation Side-Data Slots

## Goal

Generalize segmentation side-data handling so graph config can use an effectively unbounded number of `side_data_slot` values instead of the current hardcoded `0` and `1` only.

This change is needed because the current implementation supports only two segmentation channels on a frame:
- slot `0`: legacy/default masks
- slot `1`: one extra mask stream

That is enough for `court + players`, but it does not scale to additional segmentation producers or future composition patterns.

## Current State

The current design encodes the slot into the FFmpeg side-data type itself.

Files:
- `src/nodes/neural_net/common/yolo_side_data.hpp`
- `src/nodes/neural_net/yolo/infer_yolo.cpp`
- `src/nodes/neural_net/sport_specific/court_polygon.cpp`
- `src/nodes/neural_net/sport_specific/jersey_color_extract.cpp`
- `src/nodes/neural_net/draw/draw_segmask.cpp`
- `src/nodes/join_metadata.cpp`

Behavior today:
- slot `0` maps to `AV_FRAME_DATA_YOLO_SEG_MASKS` and `AV_FRAME_DATA_YOLO_SEG_MASKS_GPU`
- slot `1` maps to `AV_FRAME_DATA_YOLO_SEG_MASKS_SLOT1` and `AV_FRAME_DATA_YOLO_SEG_MASKS_GPU_SLOT1`
- any other slot currently falls back to slot `0`

That fallback is incorrect for generalized multi-stream use.

## Constraints

1. Slot count must be effectively unbounded from config.
2. GPU mask lifetime must remain correct across frame copies and `join_metadata`.
3. Existing graphs that use slot `0` or `1` must keep working.
4. `join_metadata` should remain simple and should not need slot-aware merge logic.
5. The new representation must support both CPU and GPU mask side data.
6. The number of masks in a slot is model-driven and variable per producer. It must never be hardcoded by model family or graph role.

## Recommended Design

Replace the current fixed slot-to-side-data-type mapping with a single side-data container format per mask kind:
- one side-data type for CPU segmentation masks
- one side-data type for GPU segmentation masks

Inside each side-data payload, store a slot-indexed container that can hold any number of slot entries.

This makes the FFmpeg side-data type stable while moving slot selection into the payload format, which is where it belongs.

## Data Model

### Side-data types

Keep one type for each storage class:
- CPU segmentation container
- GPU segmentation container

The exact enum values can reuse the current slot-0 identifiers for compatibility of the type namespace:
- CPU container type: current `AV_FRAME_DATA_YOLO_SEG_MASKS`
- GPU container type: current `AV_FRAME_DATA_YOLO_SEG_MASKS_GPU`

The old slot-1-specific enum values become legacy-read compatibility only.

### Container layout

Each side-data payload contains:
- a container header
- a sequence of slot entries

Suggested header fields:
- `version`
- `entry_count`

Suggested slot entry fields:
- `slot_id`
- `flags` or `kind`
- payload dimensions
- payload byte size
- payload offset or inline payload header

For CPU masks, the payload is inline float mask data.

For GPU masks, each slot entry contains the existing `GpuMaskSideDataHeader` fields:
- `gpu_ptr`
- `num_masks`
- `proto_w`
- `proto_h`
- `model_w`
- `model_h`

The container must support appending or replacing a single slot without disturbing other slots already attached to the frame.

Important:
- `num_masks` is part of each slot payload, not part of global slot configuration.
- Different slots may carry different mask counts at the same time.
- Example: a court model may emit `2` masks while a player model emits `10`, and the readers must consume the exact `num_masks` declared by the selected slot payload.
- No reader or writer may assume fixed counts such as `1`, `2`, `10`, or `max_det`; only the payload header is authoritative.

## Ownership Model

### CPU masks

CPU mask payloads live entirely inside the side-data buffer.

### GPU masks

GPU mask payloads keep an owning `AVBufferRef` per slot so the device allocation survives:
- frame propagation
- `av_frame_copy_props`
- `join_metadata`
- multiple readers on downstream frames

The implementation should preserve the current pattern where the side-data buffer owns the `AVBufferRef` that owns the CUDA allocation.

Important detail:
- GPU free callbacks must restore the correct CUDA context before `cuMemFree`, matching the safer pattern already used in `court_polygon.cpp`.

## Helper API

`src/nodes/neural_net/common/yolo_side_data.hpp` becomes the single API surface for segmentation side-data access.

Add helpers like:
- `attachYoloSegCpuSlot(...)`
- `attachYoloSegGpuSlot(...)`
- `findYoloSegCpuSlot(...)`
- `findYoloSegGpuSlot(...)`
- `readLegacyYoloSegCpuSlot(...)`
- `readLegacyYoloSegGpuSlot(...)`

Behavior:
- writers write only the new container format
- readers first check the new container format
- readers fall back to legacy slot `0`/`1` side data if needed

This preserves compatibility while converging the codebase on one representation.

The helper API must expose the parsed payload header back to callers so downstream code continues using per-slot `num_masks`, `proto_w`, `proto_h`, `model_w`, and `model_h` from the selected payload rather than reconstructing them from graph assumptions.

## Writer Changes

### `cuda_infer_yolo`

File:
- `src/nodes/neural_net/yolo/infer_yolo.cpp`

Changes:
- replace direct `av_frame_new_side_data_from_buf(... yoloSeg*SideDataType(slot) ...)` usage with helper calls into `yolo_side_data.hpp`
- write CPU and GPU mask payloads into the new slot-indexed container
- if a frame already has a container, replace only the matching slot entry

### `court_polygon`

File:
- `src/nodes/neural_net/sport_specific/court_polygon.cpp`

Changes:
- move court mask writing to the same helper API
- default court writer continues using slot `0`

## Reader Changes

### `jersey_color_extract`

File:
- `src/nodes/neural_net/sport_specific/jersey_color_extract.cpp`

Changes:
- replace direct `av_frame_get_side_data(... yoloSegGpuSideDataType(slot) ...)` with container lookup helper
- preserve existing `side_data_slot` parameter semantics

### `draw_segmask`

File:
- `src/nodes/neural_net/draw/draw_segmask.cpp`

Changes:
- same lookup migration as `jersey_color_extract`
- cached overlay logic continues to hold an `AVBufferRef` to the selected slot payload

## `join_metadata`

File:
- `src/nodes/join_metadata.cpp`

No special slot logic is needed.

Reason:
- the merged frame will just carry one CPU container side-data ref and one GPU container side-data ref
- `join_metadata` already copies side data by type and buffer ref

This is one of the main benefits of the container design.

## Backward Compatibility

Readers must support three cases during migration:
1. new container side data present
2. legacy slot `0` side data present
3. legacy slot `1` side data present

Writers should emit only the new container format.

Result:
- newly produced frames use the generalized representation
- older nodes or old captures remain readable during rollout

## Failure Handling

If a requested slot is missing, readers should behave exactly as they do today when side data is absent:
- `jersey_color_extract` passes the frame through unchanged
- `draw_segmask` logs debug output and draws nothing

If the slot payload is malformed:
- log a clear error
- treat it as missing
- do not crash

## Testing Plan

1. Unit-level validation by build and code inspection:
- compile all touched nodes
- confirm no remaining direct slot-type mapping in readers/writers except legacy fallback paths

2. Functional graph validation:
- court segmentation on slot `0`
- player segmentation on slot `1`
- confirm `jersey_color_extract` reads slot `1`
- confirm `draw_segmask` reads slot `1`

3. Compatibility validation:
- run with slot `0` only
- run with slot `1` only via legacy fallback if present

4. Remote validation:
- rebuild on the Fedora host using the documented clean build path from `CLAUDE.md`
- run the RTMP example
- verify masks are pixel masks, not bbox-like overlays
- verify no `CUDA_ERROR_ILLEGAL_ADDRESS`

## Implementation Scope

In scope:
- generalized slot container format
- helper API in `yolo_side_data.hpp`
- writer migration
- reader migration
- legacy read fallback
- CUDA free callback cleanup where needed

Out of scope:
- changing graph semantics beyond slot handling
- redesigning `join_metadata`
- changing detection metadata format
- adding non-segmentation slot namespaces

## Risks

1. GPU lifetime bugs if the new container mishandles `AVBufferRef` ownership.
2. Merge-time regressions if multiple slot updates accidentally drop earlier slot entries.
3. Compatibility regressions if legacy fallback is incomplete.

Mitigations:
- centralize all slot read/write logic in `yolo_side_data.hpp`
- keep readers simple and helper-driven
- validate with the existing `court + player` example first

## Acceptance Criteria

1. `side_data_slot` accepts arbitrary non-negative integers.
2. Slot `2+` no longer aliases to slot `0`.
3. Court and player segmentation can coexist without side-data collision.
4. Existing slot `0/1` graphs still run.
5. Different slots can carry different `num_masks` values simultaneously without hardcoded assumptions in readers or writers.
6. The RTMP example runs without CUDA illegal-address failures attributable to mask side-data handling.
