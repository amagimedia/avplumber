# neural_net/draw — CUDA Overlay Nodes

All draw nodes are SISO (single-input, single-output) operating on CUDA NV12 frames. Each uses paired luma/chroma CUDA kernels for correct NV12 color rendering.

## Shared base: `CudaOverlayBase`
Handles CUDA context init, PTX module loading, kernel function lookup, and coordinate mapping from model space to output frame space via `model_content_width/height/offset_x/offset_y` and `width/height`.

---

## Node: `draw_bbox`
Draw bounding boxes from detection metadata.

| Param | Default | Description |
|-------|---------|-------------|
| `metadata_key` / `metadata_keys` | "yolo_detections" | Detection source(s) |
| `bbox_thickness` | 2 | Line thickness in pixels |
| `min_conf` | 0.0 | Minimum confidence to draw |
| `allowed_labels` | [] | Label whitelist (empty=all) |
| `allowed_classes` | [] | Class ID whitelist |
| `model_colors` | {} | Color per model_index |
| `label_colors` | {} | Color per label string |

**Kernels:** `kDrawBBoxNV12Luma`, `kDrawBBoxNV12Chroma`

---

## Node: `draw_trail`
Draw motion trail lines from ball tracker trail points.

| Param | Default | Description |
|-------|---------|-------------|
| `metadata_key` | "yolo_detections" | Source with "trail" array |
| `color` | "red" | Trail line color |
| `thickness` | 2 | Line thickness |

Reads `trail` array of [x,y] points from metadata. Uses Bresenham line drawing.

**Kernels:** `kDrawTrailNV12Luma`, `kDrawTrailNV12Chroma`

---

## Node: `draw_keypoints`
Draw pose keypoint circles.

| Param | Default | Description |
|-------|---------|-------------|
| `metadata_key` | "yolo_pose" | Pose metadata source |
| `color` | "white" | Circle color |
| `radius` | 3 | Circle radius |
| `min_conf` | 0.0 | Minimum keypoint confidence |

**Kernels:** `kDrawKeypointsNV12Luma`, `kDrawKeypointsNV12Chroma`

---

## Node: `draw_segmask`
Overlay segmentation masks with alpha blending.

| Param | Default | Description |
|-------|---------|-------------|
| `metadata_key` | "yolo_detections" | Detection source |
| `mask_color` | "green" | Overlay color |
| `opacity` | 0.5 | Blend factor |
| `threshold` | 0.5 | Mask confidence threshold |
| `class_colors` | {} | Per-class colors |

Reads GPU mask side data (`AV_FRAME_DATA_YOLO_SEG_MASKS_GPU`). Bilinear samples mask at output resolution and blends.

**Kernels:** `kDrawSegMaskNV12Luma`, `kDrawSegMaskNV12Chroma`

---

## Node: `draw_text`
Render a text overlay from frame metadata.

| Param | Default | Description |
|-------|---------|-------------|
| `metadata_key` | required | Metadata source |
| `origin_x` / `origin_y` | 48 | Text position |
| `font_scale` | 5 | Font size |
| `draw_background` | true | Draw background box |

**Kernels:** `kDrawTextNV12Luma`, `kDrawTextNV12Chroma`
