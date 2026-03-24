# cuda_infer_yolo Refactor: Multi-Task YOLO (Detection + Segmentation + Pose)

## Goal

Refactor the `cuda_infer_yolo` node from a monolithic single-file detection-only node into a modular multi-task YOLO inference node supporting detection, segmentation (now), and pose estimation (future). Fix existing code duplication, unused code, and resource management issues along the way.

## Current Problems

- `decodeYoloOutput` has ~130 lines of duplicated decode logic (4 near-identical paths: 2 output formats x 2 memory layouts)
- `iou_thresh_` is parsed and stored but never used (no NMS implementation)
- `hwaccel_` is required at creation but never accessed after construction
- `class_names_per_model_` and `class_index_remap_per_model_` are parallel arrays that should live inside `ModelRunner`
- CUDA streams are per-model but executed sequentially (no parallelism benefit)
- No cleanup on partial initialization failure (GPU memory leaks on retry)
- No stream synchronization before cleanup in `cleanupModel`
- `buildDetectionMetadata` rebuilds the static model list every frame
- `shortEngineName` reimplements `std::filesystem::path::filename()`

## Architecture: Base + Decoders

### File Structure

```
src/nodes/hwaccel/
  cuda_infer_yolo_base.hpp    -- CudaInferYoloBase class
  cuda_infer_yolo_base.cpp    -- Base implementation
  cuda_infer_yolo.cpp         -- Node (DECLNODE, create(), process())
  yolo_decode_detection.hpp   -- DetectionDecoder
  yolo_decode_segmentation.hpp -- SegmentationDecoder
  yolo_decode_pose.hpp        -- PoseDecoder (future, not implemented now)
```

### Base Class: `CudaInferYoloBase` (`cuda_infer_yolo_base.hpp/.cpp`)

Handles all TensorRT and CUDA infrastructure. No YOLO-specific output interpretation.

**Owns:**

- `ModelRunner` struct containing:
  - Engine path, name, TRT runtime/engine/context
  - IO tensor names, byte sizes, device pointers, index map
  - Input tensor name, dims, dtype
  - Output tensor names and dims (vector -- supports multiple outputs, e.g., segmentation models have output0 for detections+coefficients and output1 for mask prototypes)
  - Output dtypes per tensor
  - Preprocess kernel function pointer, CUDA stream
  - Host output buffers per output tensor (float and half)
  - `OutputBoxFormat` enum (`EndToEndXYXY`, `RawCXCYWH`)
  - `TaskType` enum (`Detection`, `Segmentation`, `Pose`)
  - `class_names` (vector of strings)
  - `class_index_remap` (vector of ints)
  - Pointer to decoder instance
- CUDA context initialization from frame
- PTX preprocess module loading
- TRT engine loading (`parseEngine`)
- Binding allocation (`allocateBindings`) -- updated to enumerate all output tensors, not just the first. For detection models: one output tensor. For segmentation models: two output tensors (detections+coefficients and mask prototypes). Stored as vectors in `ModelRunner`.
- Input compatibility check (all models share same input size/dtype)
- Per-model preprocess kernel setup
- `runPreprocessNV12(frame, model)` -- launches preprocess kernel
- `runInference(model)` -- enqueueV3 + async D2H copy for all output tensors
- `syncModel(model)` -- stream synchronize + half-to-float conversion for all output tensors
- Cleanup with proper stream sync before resource destruction
- Rollback on partial init failure (clean up models 0..N-1 if model N fails). `cleanupModel` already null-checks all resources, so partially initialized models (e.g., stream never created) are handled safely.

**Removed from current code:**

- `hwaccel_` member (CUDA context obtained from frame)
- `iou_thresh_` (no NMS implementation exists)
- `shortEngineName` (replaced by `std::filesystem::path::filename()`)

**Fixed:**

- `cleanupModel` syncs stream before destroying it
- `ensureInitialized` rolls back on partial failure
- `buildDetectionMetadata` static model info cached at init time
- D2H copy float/half paths unified in `syncModel`

**Shared utilities (in base header):**

- `Detection` struct: x1, y1, x2, y2, conf, cls, model_index
- `halfToFloat` conversion
- `elementSize`, `volume` helpers
- `check_cu` / `CHECK_CU` macro (note: `cuda_overlay_base.hpp` has an identical copy -- consider consolidating into a shared CUDA utility header)
- `TRTLogger`

### Decoder Interface

All decoders implement the `YoloDecoder` abstract base:

```cpp
struct DetectionResult {
    std::vector<Detection> detections;
};

struct SegmentationResult : DetectionResult {
    // GPU mask at prototype resolution, wrapped in AVBufferRef for refcounted lifetime
    AVBufferRef* gpu_mask_buf = nullptr;  // owned by caller via refcount
    int mask_proto_w = 0;
    int mask_proto_h = 0;
    int num_masks = 0;  // matches detections.size()

    // CPU mask at mask_cpu_resolution (when cpu frame)
    std::vector<float> cpu_masks;  // N * cpu_res * cpu_res floats
    int cpu_mask_w = 0;
    int cpu_mask_h = 0;
};

// Common decode parameters, passed by the node to avoid repeating per-call
struct DecodeParams {
    int model_index;
    float conf_thresh;
    OutputBoxFormat box_format;
    const std::vector<int>& class_index_remap;
};
```

**There is no abstract `YoloDecoder` base class.** The node knows each model's `task_type` at init time and holds typed decoder instances directly:

- `ModelRunner` holds a `std::unique_ptr<DetectionDecoder>` or `std::unique_ptr<SegmentationDecoder>` (via a `std::variant` or simply two optional pointers, only one non-null).
- The node calls the decoder directly by type -- no virtual dispatch, no return-type downcasting.

```cpp
class DetectionDecoder {
public:
    // Decode detection output. host_outputs[0] is the single output tensor.
    DetectionResult decode(
        const std::vector<const float*>& host_outputs,
        const std::vector<nvinfer1::Dims>& output_dims,
        const DecodeParams& params
    );
};

class SegmentationDecoder {
public:
    // Initialize mask assembly kernel and GPU buffers.
    // Called once after engine is loaded. CUDA context must be current.
    bool init(const ModelRunner& model, CUcontext cu_ctx);

    // Decode segmentation output. host_outputs[0] is detections+coefficients,
    // host_outputs[1] is mask prototypes.
    // emit_gpu/emit_cpu control whether to assemble GPU/CPU masks this frame.
    SegmentationResult decode(
        const std::vector<const float*>& host_outputs,
        const std::vector<nvinfer1::Dims>& output_dims,
        const DecodeParams& params,
        bool emit_gpu_mask,
        bool emit_cpu_mask,
        int cpu_mask_resolution
    );

    // Free GPU resources (mask assembly kernel, buffers).
    // Must be called with CUDA context current, before destruction.
    void cleanup();
};
```

`DetectionDecoder` is stateless -- no `init()` or `cleanup()` needed. `SegmentationDecoder` owns GPU resources (mask assembly kernel, scratch buffers) and requires explicit `cleanup()` with CUDA context current before destruction. The destructor does not free GPU resources (CUDA context may not be current at destruction time).

**GPU mask lifetime on non-emission frames:** When `emit_gpu_mask` is false, the decoder skips GPU mask assembly entirely -- no `cuMemAlloc`, no kernel launch, no `AVBufferRef` created. This is more efficient than assembling and discarding. The node passes `emit_gpu_mask = (inference_frame_counter % mask_gpu_every_n == 0)` to the decoder.

### Detection Decoder: `yolo_decode_detection.hpp`

Replaces the current duplicated `decodeYoloOutput` with a single unified path.

**Decode flow:**

1. Determine layout from output dims: extract `count` (number of detections), `attrs` (attributes per detection), and memory layout (row-major vs column-major)
2. Build one accessor lambda: `at(detection_index, attr_index) -> float` -- handles both layouts
3. Single decode loop switching on `OutputBoxFormat`:
   - `EndToEndXYXY`: read x1, y1, x2, y2, conf, cls from attrs 0-5
   - `RawCXCYWH`: read cx, cy, w, h, then scan remaining attrs for best class score
4. Apply class index remapping
5. Return `std::vector<Detection>`

**Handles all current output dim cases:**
- 2D: `[attrs, count]` or `[count, attrs]`
- 3D: `[1, N, 6]`, `[1, 6, N]`, `[1, attrs, count]`, `[1, count, attrs]`

All unified through the accessor lambda -- no code duplication.

**Note:** The current code always uses center-box decoding for 2D output dims regardless of `OutputBoxFormat`. The refactored decoder will route 2D outputs through the `OutputBoxFormat` switch like all other dims. If no existing models produce 2D output with `EndToEndXYXY`, this is a no-op change. If any do, it is a bug fix (the current code incorrectly decodes them as center-box).

### Segmentation Decoder: `yolo_decode_segmentation.hpp`

Handles YOLO-seg model output (two tensors).

**Input tensors:**

- **output0**: detections + 32 mask coefficients per detection (same layout as detection but with 32 extra attrs after class scores)
- **output1**: mask prototypes `[1, 32, proto_h, proto_w]` (prototype resolution = input_size / 4, e.g., 240x240 for 960x960 input)

**Decode flow:**

1. Decode detections using the same accessor logic as DetectionDecoder, plus extract 32 mask coefficients per detection
2. Assemble masks on GPU: matrix multiply coefficients x prototypes -> per-detection mask at prototype resolution, sigmoid activation. This requires a new CUDA kernel in `yolo_mask_assemble.cu`, compiled to PTX alongside the existing `yolo_preprocess.cu`. The kernel performs: for each detection, dot-product of 32 coefficients against the 32 prototype channels at each spatial position, followed by sigmoid activation.
3. GPU path (every `mask_gpu_every_n` frames): assembled masks are allocated via `cuMemAlloc` and wrapped in an `AVBufferRef` with a custom release callback that calls `cuMemFree`. This `AVBufferRef` is attached to an `AVFrame` output on the `dst_seg` edge. Lifetime is managed by FFmpeg's reference counting -- when all downstream nodes release the frame, the release callback fires and the GPU buffer is freed. No separate `AVHWFramesContext` pool is needed since mask dimensions differ from video frame dimensions.
4. CPU path (every `mask_cpu_every_n` frames): bilinear downsample on GPU to `mask_cpu_resolution`, then `cuMemcpyDtoH` to host. Store as AVFrame side data on the passthrough frame.

**`mask_gpu_every_n` and `mask_cpu_every_n` count inference frames, not total frames.** If `infer_every_n=2` and `mask_cpu_every_n=3`, the CPU mask fires every 3rd inference, which is every 6th input frame.

### Pose Decoder: `yolo_decode_pose.hpp` (Future)

Not implemented now. The architecture accommodates it as a third `TaskType` with its own decoder. YOLO-pose outputs N keypoints per detection (each with x, y, confidence). The decoder would extract keypoints alongside detections and output them in metadata.

The detection and segmentation decoder designs are structured so pose slots in naturally: same base class, same `ModelRunner` dispatch by `task_type`, own metadata key.

## Node: `cuda_infer_yolo.cpp`

The avplumber node. Owns `DECLNODE`, `create()`, `process()`.

**Node type:** The node uses `NodeSingleInput<av::VideoFrame>` and manually manages two `EdgeSink<av::VideoFrame>` members: a required primary sink (`dst`) and an optional segmentation mask sink (`dst_seg`). This is simpler than using `NodeMultiOutput` because `dst_seg` is truly optional -- detection-only configs do not provide it. The `create()` method checks if `dst_seg` exists in params and only creates the second sink if present.

### Parameters

```
src                          -- input edge (960x960 NV12 CUDA frames)
dst                          -- primary output edge (passthrough frame with text metadata)
dst_seg                      -- optional segmentation mask output edge (AVFrame CUDA buffer)
conf_thresh: 0.25            -- confidence threshold
max_det: 300                 -- max detections after sort
infer_every_n: 1             -- run inference every Nth frame
input_format: "RGB"          -- RGB or BGR channel order
metadata_key_detection: "yolo_detections"
metadata_key_segmentation: "yolo_segmentation"
mask_gpu_every_n: 1          -- GPU mask output rate
mask_cpu_every_n: 2          -- CPU mask (side data) output rate
mask_cpu_resolution: 120     -- CPU mask downsample resolution (square)
debug_log_metadata: false
debug_log_every_n: 30
models: [                    -- array of model configs
  {
    engine: "path/to/model.plan",
    task_type: "detection",          -- required: "detection" or "segmentation"
    output_box_format: "end2end_xyxy", -- required: "end2end_xyxy" or "raw_cxcywh"
    class_names: ["ball"],
    class_index_remap: []
  }
]
```

All models must share the same input resolution and dtype (enforced at init). Different input sizes require separate node instances in the avplumber graph.

### process() Flow

1. Read frame from source, skip if `infer_every_n` says so
2. Validate: CUDA format, NV12 sw_format, correct dimensions
3. `ensureInitialized` (base class: parse engines, allocate bindings, setup preprocess)
4. For each model: `runPreprocessNV12` -> `runInference` (base class)
5. For each model: `syncModel`, dispatch to decoder by `task_type`:
   - Detection -> `std::vector<Detection>`
   - Segmentation -> `std::vector<Detection>` + mask data (GPU + optionally CPU)
6. Build detection metadata JSON -> `av_dict_set` on frame under `metadata_key_detection`
7. If segmentation models present and CPU frame (`mask_cpu_every_n`): attach downscaled mask as AVFrame side data
8. Put frame on primary output (`dst`)
9. If segmentation models present and GPU frame (`mask_gpu_every_n`): put mask AVFrame on `dst_seg`

### Detection Metadata Format (`yolo_detections`)

```json
{
  "coord_space": "model",
  "model_width": 960,
  "model_height": 960,
  "models": [
    {"model_index": 0, "engine": "ball.plan", "engine_name": "ball.plan"}
  ],
  "detections": [
    {
      "cls": 0,
      "conf": 0.85,
      "xyxy": [100, 200, 150, 250],
      "model_index": 0,
      "engine_name": "ball.plan",
      "label": "basketball"
    }
  ]
}
```

Static model list is cached at init time, not rebuilt every frame.

**Compatibility note:** The current code outputs `"version": 2` and `"thresholds"` (including `iou`) in the metadata. The refactored version drops `"version"` and the `"thresholds.iou"` field since `iou_thresh` is removed. Downstream consumers (`basketball_analysis`, `draw_bbox`, `draw_text`) have been checked -- they read only `"detections"`, `"model_width"`, `"model_height"`, and `"coord_space"`, none of which change.

### Segmentation CPU Metadata (AVFrame Side Data)

Stored as AVFrame side data using a project-local side data type constant cast to `AVFrameSideDataType`. Define `static const AVFrameSideDataType AV_FRAME_DATA_YOLO_SEG_MASKS = (AVFrameSideDataType)0x59534D00;` in the base header. This avoids collisions with upstream FFmpeg enum values (which are small sequential integers). The binary layout is:

```
Header (16 bytes):
  uint32_t  num_detections    -- number of masks (matches detection count)
  uint32_t  mask_width        -- mask_cpu_resolution
  uint32_t  mask_height       -- mask_cpu_resolution
  uint32_t  reserved          -- padding, set to 0

Masks (num_detections * mask_width * mask_height * sizeof(float) bytes):
  float[]   mask_0            -- mask_width * mask_height floats, row-major
  float[]   mask_1            -- same size
  ...
  float[]   mask_N-1          -- same size
```

Masks are in the same order as detections in the JSON metadata. To check if detection[i] is in a zone: read `mask_i = data + 16 + i * mask_width * mask_height * 4`, scale detection bbox center to mask coords, and look up the pixel value.

Coordinate mapping from mask to model space: `model_coord = mask_coord * (model_size / mask_cpu_resolution)`.

Analysis nodes read the raw floats directly -- no JSON parsing or base64 decoding.

### Segmentation GPU Output (dst_seg Edge)

AVFrame with CUDA pixel format containing the assembled masks at prototype resolution (e.g., 240x240 for 960x960 input). Carries PTS matching the input frame for downstream joining. Lifetime managed by AVFrame refcounting.

## Changes to Existing Nodes

### `join_metadata` (`src/nodes/join_metadata.cpp`)

Extended to copy AVFrame side data from auxiliary frame to primary frame, alongside existing text metadata copy (which uses `av_dict_copy`). Side data copying is done by iterating the auxiliary frame's side data entries and adding each one to the primary frame via `av_frame_new_side_data_from_buf` (preserving the `AVBufferRef` refcount). Only side data types not already present on the primary frame are copied, to avoid overwriting. This allows CPU segmentation masks (stored as side data on the YOLO output) to propagate to the full-resolution branch.

## Example Pipeline Update

The `examples/yolo/yolo_bbox.avplumber` file will be updated:

- Each model in the `models` array gets an explicit `task_type: "detection"`
- `metadata_key_out` replaced by `metadata_key_detection`
- `iou_thresh` param removed
- All other params and pipeline structure remain the same

Existing behavior is preserved -- the refactored node with detection-only models produces identical output.

## Constraints

- All models in one node instance share the same input resolution and dtype
- `output_box_format` is an explicit param per model (cannot be auto-detected due to ambiguity with 2-class models)
- No NMS implementation (end-to-end models have it baked in; raw models tolerate duplicates downstream)
- Pose decoder is a future addition -- not implemented, but the architecture accommodates it
- New `yolo_mask_assemble.cu` requires a Makefile PTX compilation rule (`.cu` -> `.ptx` -> `.ptx.h`), following the existing pattern for `yolo_preprocess.cu`
