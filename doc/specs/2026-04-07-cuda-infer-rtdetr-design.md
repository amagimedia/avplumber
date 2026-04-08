# cuda_infer_rtdetr Design (v1)

## Summary

Add a new node `cuda_infer_rtdetr` to run RF-DETR TensorRT engines in avplumber while reusing as much of `CudaInferYoloBase` as possible.

v1 is intentionally strict:

- single model
- batch size 1
- fixed-shape engine only
- end-to-end output contract only (no raw-query decode)
- preprocessing must match current avplumber path (`NV12 -> RGB/BGR NCHW`, `[0,1]`)

The node emits detection metadata in the same schema used today so downstream nodes continue to work unchanged.

## Goals

- Reuse existing TensorRT/CUDA runtime plumbing from `CudaInferYoloBase`.
- Add a separate node type (`cuda_infer_rtdetr`) for clean operational separation.
- Preserve downstream compatibility (`draw_bbox`, `basketball_analysis`, `smooth_crop_viewport`).
- Keep v1 deterministic with strict contract validation and fail-fast init checks.

## Non-Goals (v1)

- No dynamic shape/profile support.
- No multi-model fusion in one node.
- No batch >1.
- No C++ NMS/postprocess for raw query tensors.
- No new preprocess normalization variants (mean/std) in v1.

## Output Contract (Required)

`output_contract: "rtdetr_e2e_v1"` is mandatory.

Required engine outputs:

- `boxes`: `[1, N, 4]` in `xyxy`
- `scores`: `[1, N]`
- `labels`: `[1, N]`

Assumptions:

- engine is end-to-end postprocessed (top-k/NMS already done)
- coordinates are model-pixel `xyxy` (not normalized)

Any mismatch at init must fail with explicit log errors.

## Architecture

### Reused from `CudaInferYoloBase`

- CUDA context acquisition from frame
- TensorRT engine deserialize + context create
- I/O tensor allocation + device buffers
- NV12 preprocess kernel launch
- `enqueueV3` inference
- output copy/sync and fp16-to-fp32 conversion

### New in RT-DETR path

- New node: `src/nodes/hwaccel/cuda_infer_rtdetr.cpp`
- New decoder: `src/nodes/hwaccel/rtdetr_decode_detection.hpp`
- Contract validation for output tensors in RT-DETR node init path
- Detection assembly into existing metadata JSON format

## Node Parameters

Global params (same behavior as YOLO where applicable):

- `src` (required)
- `dst` (required)
- `models` (required, exactly 1 item in v1)
- `conf_thresh` (default `0.25`)
- `max_det` (default `300`)
- `infer_every_n` (default `1`)
- `metadata_key_detection` (default `"yolo_detections"`)
- `debug_log_metadata` (default `false`)
- `debug_log_every_n` (default `30`)
- `input_format` (`"RGB"` or `"BGR"`, default RGB)

Model params (single item in `models`):

- `engine` (required)
- `output_contract` (required, must be `"rtdetr_e2e_v1"`)
- `class_names` (optional)
- `class_index_remap` (optional)
- `boxes_normalized` (optional, default `false`; if `true` in v1 -> fail init)

## Metadata Output Compatibility

Write detection metadata using same schema as current YOLO node:

- top-level: `coord_space`, `model_width`, `model_height`, `models`, `detections`
- each detection: `cls`, `conf`, `xyxy`, `model_index`, optional `engine_name`, optional `label`

Behavioral requirements:

- metadata key is written every processed inference frame
- zero-detection frames write empty `detections` array
- per-frame decode failure: pass frame through without dropping; write empty detections and log rate-limited error

## Error Handling

Fail fast during initialization on:

- missing/invalid `output_contract`
- unsupported output tensor count/shape/dtype
- dynamic dims
- batch size != 1
- normalized boxes configuration in v1

Runtime behavior:

- non-CUDA / unsupported frame format: pass through with log (existing behavior parity)
- decode failure on a frame: fail-open (frame continues), metadata reset to empty detections

## Logging

At init (debug-enabled), log once:

- engine name/path
- input tensor name/shape/dtype
- output tensor names/shapes/dtypes
- selected output contract

At runtime (debug cadence), log:

- detection count
- optional compact confidence histogram (same pattern as current node if useful)

## File-Level Change Plan

1. Create `src/nodes/hwaccel/rtdetr_decode_detection.hpp`
- Decode `boxes/scores/labels` into internal `Detection`.
- Apply `conf_thresh`.
- Apply `class_index_remap`.

2. Create `src/nodes/hwaccel/cuda_infer_rtdetr.cpp`
- Node create/param parsing.
- Enforce v1 constraints.
- Reuse `CudaInferYoloBase` lifecycle (`ensureInitialized`, preprocess, inference, sync).
- Build and write compatible metadata JSON.
- Register with `DECLNODE(cuda_infer_rtdetr, CudaInferRTDetr)`.

3. Minimal extension in `src/nodes/hwaccel/cuda_infer_yolo_base.hpp`
- Add RT-DETR decoder pointer slot on `ModelRunner` (minimal intrusive change), or equivalent generic hook needed by new node.
- Do not duplicate base runtime code.

4. Build wiring in `Makefile`
- Add new `cuda_infer_rtdetr.cpp` to TensorRT+CUDA+NVCC section.

5. Add example
- `examples/rtdetr/rtdetr_ball_only.avplumber`
- mirror YOLO ball-only topology, swapping inference node type.

6. Documentation
- Add `cuda_infer_rtdetr` section in `README.md` with params and contract.

## Test Plan

### Unit-ish decoder checks

- synthetic tensors for `boxes/scores/labels`
- verify thresholding, class remap, coordinate pass-through
- verify empty/low-confidence behavior

### Runtime smoke

- run `cuda_infer_rtdetr` on known-good RF-DETR `.plan`
- verify metadata key exists and renders with `draw_bbox`
- verify no frame drops from decoder-path errors

### Regression

- run existing YOLO example to ensure no behavior regressions from shared-base changes

## Acceptance Criteria

- Node loads fixed-shape RF-DETR end-to-end TensorRT engine and runs inference.
- Detections are emitted in existing metadata schema.
- Existing downstream nodes work without code changes.
- Contract mismatches fail at init with actionable errors.
- Runtime decode failures do not interrupt video flow.

## Future Extensions (Post-v1)

- dynamic shape/profile support
- optional mean/std preprocess mode
- raw-query decode path + optional C++ NMS
- multi-model aggregation in RT-DETR node
