# Generalized YOLO Segmentation Side-Data Slots

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

Keep the current model of one `AVFrame` side-data entry per segmentation stream, but generalize the side-data type mapping so it is derived from the slot number instead of being special-cased for slot `0` and slot `1`.

This keeps the payload format unchanged and avoids introducing an extra container structure inside side data.

## Data Model

### Side-data types

Use a deterministic mapping from `slot` to side-data type:
- CPU type for slot `k`: `base_cpu + 2*k`
- GPU type for slot `k`: `base_gpu + 2*k`

Where the base types remain:
- `AV_FRAME_DATA_YOLO_SEG_MASKS`
- `AV_FRAME_DATA_YOLO_SEG_MASKS_GPU`

This preserves the current slot `0` and slot `1` numeric values while allowing additional slots within a centrally defined range.

### Payload layout

Payload layout stays exactly as it is today.

For GPU masks, the payload contains the existing `GpuMaskSideDataHeader` fields:
- `gpu_ptr`
- `num_masks`
- `proto_w`
- `proto_h`
- `model_w`
- `model_h`

Important:
- `num_masks` is part of each slot payload, not part of global slot configuration.
- Different slots may carry different mask counts at the same time.
- Example: a court model may emit `2` masks while a player model emits `10`, and the readers must consume the exact `num_masks` declared by the selected slot payload.
- No reader or writer may assume fixed counts such as `1`, `2`, `10`, or `max_det`; only the payload header is authoritative.

## Ownership Model

### CPU masks

CPU mask payloads live entirely inside the side-data buffer.

### GPU masks

GPU mask payloads keep an owning `AVBufferRef` so the device allocation survives:
- frame propagation
- `av_frame_copy_props`
- `join_metadata`
- multiple readers on downstream frames

The implementation should preserve the current pattern where each side-data buffer owns the `AVBufferRef` that owns the CUDA allocation.

Important detail:
- GPU free callbacks must restore the correct CUDA context before `cuMemFree`, matching the safer pattern already used in `court_polygon.cpp`.

## Helper API

`src/nodes/neural_net/common/yolo_side_data.hpp` becomes the single API surface for segmentation side-data access.

Add helpers like:
- `yoloSegIsValidSlot(...)`
- `yoloSegCpuSideDataType(slot)`
- `yoloSegGpuSideDataType(slot)`

Behavior:
- all readers and writers resolve side-data type through these helpers
- no node should special-case slot `1`
- slot limits are enforced centrally rather than scattered through the graph code

## Writer Changes

### `cuda_infer_yolo`

File:
- `src/nodes/neural_net/yolo/infer_yolo.cpp`

Changes:
- replace hardcoded slot-0/slot-1 assumptions with helper calls into `yolo_side_data.hpp`
- keep writing the existing CPU and GPU payload formats

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
  The implementation uses the generalized side-data type mapping helper, not a container lookup.

### `draw_segmask`

File:
- `src/nodes/neural_net/draw/draw_segmask.cpp`

Changes:
- same lookup migration as `jersey_color_extract`
- cached overlay logic continues to hold an `AVBufferRef` to the selected slot payload
- drawing logic must continue to support variable `num_masks` from the selected slot payload and must handle player-style batches such as 10 masks on the same frame without any hardcoded low mask-count assumption
- implementation should move toward batching the selected detections for a slot in a single launch flow per plane rather than assuming per-mask launch overhead is negligible
- the node contract is: if the selected slot payload declares `num_masks = N`, and metadata maps `N` mask indices to detections, the node must be able to render that full batch correctly for `N > 1`

## `join_metadata`

File:
- `src/nodes/join_metadata.cpp`

No special slot logic is needed.

Reason:
- each slot remains a normal side-data entry with its own type
- `join_metadata` already copies side data by exact type and buffer ref

## Backward Compatibility

All in-tree readers and writers are migrated together to the generalized mapping.

Result:
- slot `0` and slot `1` keep their existing type ids
- new slots derive their type ids from the same mapping
- no extra legacy compatibility layer is required

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
- confirm no remaining `slot == 1` special-casing in readers/writers

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
- generalized per-slot side-data type mapping
- helper API in `yolo_side_data.hpp`
- writer migration
- reader migration
- CUDA free callback cleanup where needed

Out of scope:
- changing graph semantics beyond slot handling
- redesigning `join_metadata`
- changing detection metadata format
- adding non-segmentation slot namespaces

Related but in scope for correctness:
- `draw_segmask` may need internal kernel-launch restructuring if that is required to reliably render larger mask batches such as player overlays

## Risks

1. GPU lifetime bugs if the new container mishandles `AVBufferRef` ownership.
2. Consumer races if a GPU mask buffer is published before its async copy is complete.
3. Range/validation regressions if slot bounds are not enforced consistently.

Mitigations:
- centralize all slot read/write logic in `yolo_side_data.hpp`
- keep readers simple and helper-driven
- validate with the existing `court + player` example first

## Acceptance Criteria

1. `side_data_slot` accepts centrally validated slot values without `slot == 1` special handling.
2. Slot `2+` no longer aliases to slot `0`.
3. Court and player segmentation can coexist without side-data collision.
4. Existing slot `0/1` graphs still run.
5. Different slots can carry different `num_masks` values simultaneously without hardcoded assumptions in readers or writers.
6. `draw_segmask` correctly renders multi-mask batches for player segmentation workloads, including cases around 10 masks per frame.
7. The RTMP example runs without CUDA illegal-address failures attributable to mask side-data handling.
