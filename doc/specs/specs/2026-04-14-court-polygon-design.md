# Court Polygon Node Design

**Date:** 2026-04-14
**Branch:** neural_demo
**File:** `src/nodes/neural_net/sport_specific/court_polygon.cpp`
**Build gate:** `NEURAL_NET_SPECIFIC=1`

## Purpose

Replace the YOLO segmentation model for court mask generation with a lighter
approach: read 12 court-perimeter keypoints from a pose model, rasterize a
filled polygon, and emit side data in the exact same format the segmentation
model produces. The ball tracker and shot classifier consume the output with
zero changes.

## Context

The current pipeline runs a YOLO segmentation model (`court-segmentation`)
that outputs a pixel-level court mask. Two downstream nodes consume it:

- **shot_classifier** counts court pixels to compute a coverage ratio
  (`court_coverage`) used to classify wide vs closeup shots.
- **ball_tracker** samples horizontal rows of the mask at the ball's position
  to veto false-positive detections outside the court bounds.

Both read the mask from `AV_FRAME_DATA_YOLO_SEG_MASKS` CPU side data and
`yolo_seg` frame metadata. The `draw_segmask` visualization node reads the
GPU variant (`AV_FRAME_DATA_YOLO_SEG_MASKS_GPU`).

A new court-pose model is being trained with 12 perimeter keypoints (4
corners, 6 sideline intermediates, 2 baseline midpoints). This is smaller and
faster than the segmentation model. The `court_polygon` node bridges pose
output to the existing seg mask interface.

## Keypoints

The pose model outputs 12 keypoints in this order (new index = old 33-kpt index):

| New | Old | Court location |
|-----|-----|----------------|
| 0 | 0 | Left baseline, top corner |
| 1 | 3 | Left baseline, mid |
| 2 | 5 | Left baseline, bottom corner |
| 3 | 12 | Top sideline, left |
| 4 | 14 | Bottom sideline, left |
| 5 | 15 | Top sideline, center |
| 6 | 17 | Bottom sideline, center |
| 7 | 18 | Top sideline, right |
| 8 | 20 | Bottom sideline, right |
| 9 | 27 | Right baseline, top corner |
| 10 | 30 | Right baseline, mid |
| 11 | 32 | Right baseline, bottom corner |

## Polygon winding order

The default winding order traces the court perimeter clockwise:

```
0 -> 3 -> 5 -> 7 -> 9 -> 10 -> 11 -> 8 -> 6 -> 4 -> 2 -> 1 -> back to 0
```

Top edge: 0, 3, 5, 7, 9
Right edge: 9, 10, 11
Bottom edge: 11, 8, 6, 4, 2
Left edge: 2, 1, 0

Configurable via `winding_order` param if the keypoint layout changes.

## Node type

`NodeSISO<av::VideoFrame, av::VideoFrame>`, blocking. Registered as
`court_polygon` via `DECLNODE`.

## Input

Reads pose metadata from the frame under a configurable key (default
`yolo_pose`). Expected format (produced by `infer_yolo` with
`task_type: pose`):

```json
{
  "coord_space": "model",
  "model_width": 960,
  "model_height": 544,
  "num_keypoints": 12,
  "poses": [{
    "cls": 0,
    "conf": 0.92,
    "xyxy": [x1, y1, x2, y2],
    "keypoints": [x0, y0, c0, x1, y1, c1, ..., x11, y11, c11]
  }]
}
```

Takes the highest-confidence pose detection per frame.

## Processing

### 1. Parse keypoints

Extract the 12 keypoints from the flat array (stride 3: x, y, confidence).
Coordinates are in model space (e.g. 960x544).

### 2. Filter by confidence

Each keypoint has a confidence value. Discard keypoints below
`min_keypoint_conf` (default 0.3). If fewer than `min_visible_keypoints`
(default 6) remain, skip this frame (no polygon).

### 3. Build polygon

Order the surviving keypoints according to `winding_order`, skipping any
that were filtered out. The polygon connects the remaining points in order.

### 4. Validate polygon shape

Two checks before rasterization:

**Vertex count:** Require at least `min_visible_keypoints` (default 6)
vertices after confidence filtering. A triangle or thin quadrilateral from
3-4 keypoints is never a valid court outline.

**Convexity:** A court polygon seen from a broadcast camera is always
convex (it is a perspective-transformed rectangle). Verify that all
consecutive cross products along the winding have the same sign. If any
cross product has the opposite sign, the polygon is concave or
self-intersecting — discard it.

If either check fails, treat as no valid polygon (triggers hold logic).

### 5. Rasterize

CPU scanline fill at `mask_w` x `mask_h` (default 240x136):

- Map polygon vertices from model space to mask space:
  `mask_x = kpt_x / model_w * mask_w`, `mask_y = kpt_y / model_h * mask_h`
- For each row of the mask, compute edge intersections with the polygon,
  sort by x, fill between pairs with 1.0f. Everything else is 0.0f.

### 6. Area check

After rasterization, count filled pixels / total pixels. If the ratio is
below `min_court_area` (default 0.05), discard the mask — treat it the same
as a failed polygon. This handles closeup shots where the pose model
confidently detects a few keypoints but the resulting polygon is a tiny
sliver that would mislead downstream consumers. No dependency on
`shot_info` metadata or node ordering.

### 7. Attach CPU side data

`AV_FRAME_DATA_YOLO_SEG_MASKS` (type `0x59534D00`):

```
Offset  Size    Content
0       4       uint32_t num_masks = 1
4       4       uint32_t mask_w    = 240
8       4       uint32_t mask_h    = 136
12      4       uint32_t reserved  = 0
16      ...     float[mask_w * mask_h]  (1.0 inside, 0.0 outside)
```

Total size: `16 + mask_w * mask_h * sizeof(float)` bytes.

Allocated via `av_buffer_alloc()`, attached via
`av_frame_new_side_data_from_buf()`.

Ball tracker validates `sd->size >= 16 + num_masks * mask_w * mask_h *
sizeof(float)` before reading. A single-mask buffer passes this check.

### 8. Attach GPU side data

`AV_FRAME_DATA_YOLO_SEG_MASKS_GPU` (type `0x59534D01`):

A CPU-allocated `GpuMaskSideDataHeader`:

```
Offset  Size    Content
0       8       uint64_t gpu_ptr   = CUdeviceptr to float[mask_w * mask_h]
8       4       uint32_t num_masks = 1
12      4       uint32_t proto_w   = 240   (mask width)
16      4       uint32_t proto_h   = 136   (mask height)
20      4       uint32_t model_w   = 960   (from pose model_width)
24      4       uint32_t model_h   = 544   (from pose model_height)
```

GPU buffer allocated via `cuMemAlloc`, populated via `cuMemcpyHtoD` from
the CPU mask on every frame that produces a valid polygon.

The `AVBufferRef` wrapping the header uses a custom free callback that calls
`cuMemFree` on the GPU pointer and `av_free` on the header struct.

`draw_segmask` indexes into the mask as:
`mask_ptr = gpu_ptr + mask_index * proto_h * proto_w * sizeof(float)`

With `num_masks=1` and `mask_index=0` this gives the base pointer directly.

### 9. Write metadata

Writes `yolo_seg`-compatible JSON (configurable key, default `yolo_seg`):

```json
{
  "model_width": 960,
  "model_height": 544,
  "detections": [{
    "cls": 0,
    "conf": 0.92,
    "xyxy": [0, 0, 960, 544],
    "label": "court"
  }]
}
```

Downstream consumers must set `court_class_indices` to `{0}` (single class).

## Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `pose_metadata_key` | string | `"yolo_pose"` | Frame metadata key for pose input |
| `metadata_key_out` | string | `"yolo_seg"` | Frame metadata key for seg output |
| `mask_w` | int | `240` | Rasterized mask width |
| `mask_h` | int | `136` | Rasterized mask height |
| `min_keypoint_conf` | float | `0.3` | Per-keypoint confidence threshold |
| `min_visible_keypoints` | int | `6` | Minimum visible keypoints to form polygon |
| `min_court_area` | float | `0.05` | Minimum filled-pixel ratio to accept mask |
| `winding_order` | int[] | `[0,3,5,7,9,10,11,8,6,4,2,1]` | Keypoint indices for polygon perimeter |
| `debug_log_every_n` | int | `0` | Debug log interval (0 = off) |

## Pipeline integration

In an `.avplumber` script, replaces the court segmentation model. Before:

```
node.add {"name":"seg","type":"cuda_infer_yolo","task_type":"segmentation",
          "engine":"/home/tensorrt/court-segmentation_960x544.plan",...}
```

After:

```
node.add {"name":"pose","type":"cuda_infer_yolo","task_type":"pose",
          "engine":"/home/tensorrt/court-pose-small.plan",
          "num_classes":1,...}
node.add {"name":"court_poly","type":"court_polygon","src":"q_after_pose",
          "dst":"q_to_tracker",...}
```

Downstream `shot_classifier` and `ball_tracker` need only
`court_class_indices` set to `[0]` instead of `[0, 1]`.

## CUDA context

The node is not a draw node (does not subclass `CudaOverlayBase`) but needs
CUDA access for the GPU side data upload. It obtains the `CUcontext` from
the incoming frame's `hw_frames_ctx`, same pattern used by all GPU-aware
nodes:

```cpp
AVHWFramesContext* fctx = (AVHWFramesContext*)frm.raw()->hw_frames_ctx->data;
AVCUDADeviceContext* cuda_dev = (AVCUDADeviceContext*)fctx->device_ctx->hwctx;
cuCtxSetCurrent(cuda_dev->cuda_ctx);
```

If the frame is not a CUDA frame (no `hw_frames_ctx`), the node skips GPU
side data attachment and only emits CPU side data. This makes it usable in
CPU-only pipelines where `draw_segmask` is not in use.

The GPU buffer (`cuMemAlloc`) is allocated once and reused across frames.
It is reallocated only if `mask_w`/`mask_h` change (they are fixed params,
so in practice this never happens). On destruction, `cuMemFree` is called.

## Dependencies

- Includes `yolo_side_data.hpp` for side data type constants and
  `GpuMaskSideDataHeader`
- Includes `<cuda_loader/cuda_drvapi_dynlink_cuda.h>` for CUDA driver API
- Uses `cuMemAlloc`, `cuMemcpyHtoD`, `cuMemFree`, `cuCtxSetCurrent` via
  the existing `cuda_loader` dynamic loader in `deps/`
- Includes `<libavutil/hwcontext.h>` and `<libavutil/hwcontext_cuda.h>` for
  `AVCUDADeviceContext`
- No `.cu` kernels needed (all rasterization is CPU)

## Testing

No automated tests (per project convention). Verify manually:

1. Run the tracker-vod pipeline with the pose model + `court_polygon`
   replacing the seg model.
2. Confirm `shot_classifier` log output shows reasonable `court_coverage`
   values (comparable to the seg model).
3. Confirm `ball_tracker` court bounds veto still fires for out-of-court
   false positives.
4. Confirm `draw_segmask` renders the court overlay visually matching the
   actual court outline.
5. Test closeup handling: verify that closeup shots produce no mask
   (area check rejects partial court polygons).
