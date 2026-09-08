# hog_diff

Pass-through CUDA video node that computes a HOG-descriptor L1 distance between
consecutive frames and writes it to frame metadata under the same `scene_diff`
contract as `luma_diff`, so it is a drop-in metric source for the recorder's
`LumaSceneCutDecider`.

The node expects CUDA video frames with an 8-bit luma plane, typically produced
by `scale_cuda=...:format=nv12`. It downscales the luma plane to
`analysis_width` x `analysis_height` on GPU, computes the HOG descriptor on that
small plane, and forwards the original full-resolution frame unchanged with
`scene_diff` metadata attached. It keeps a ring of recent HOG descriptors on
GPU and emits only compact JSON metadata. No CPU round-trip, no OpenCV.

Per frame it builds a Dalal-Triggs-style HOG descriptor on the luma plane
(Sobel gradients -> per-cell magnitude-weighted orientation histograms ->
block normalization -> concatenated descriptor) and reports the L1 distance
between two frames' descriptors as `mean_abs`.

## Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `src` | edge | required | Input `VideoFrame` edge. |
| `dst` | edge | required | Output `VideoFrame` edge. |
| `metadata_key` | string | `scene_diff` | Metadata key to write. |
| `strict_cuda` | bool | `true` | Throw on non-CUDA/unsupported input instead of writing skipped metadata. |
| `debug_log_every_n` | int | `0` | Log one sample every N frames. |
| `frames_lookahead` | int | `0` | Number of forward diffs (`scene_diff+1..+N`) to emit per frame. |
| `analysis_width` | int | `640` | Width of the internal luma downscale HOG runs on (full-res frame is forwarded). |
| `analysis_height` | int | `360` | Height of the internal luma downscale HOG runs on. |
| `cell_width` | int | `8` | Cell width in pixels. |
| `cell_height` | int | `8` | Cell height in pixels. |
| `block_size_cells` | int | `2` | Block size in cells (per side). |
| `block_stride_cells` | int | `2` | Block stride in cells. |
| `num_bins` | int | `9` | Number of orientation bins per cell histogram. |
| `signed_orientations` | bool | `false` | Use `[0,360)` orientations instead of unsigned `[0,180)`. |
| `block_norm` | string | `l2` | Block normalization: `l2`, `l1`, or `l2hys`. |
| `gamma` | float | `0.0` | Optional gamma correction applied to luma before gradients (`0` = off). |

## Metadata

Emitted for `scene_diff` (k=0) and `scene_diff+k` for k=1..`frames_lookahead`,
mirroring `luma_diff`:

```json
{
  "frame_index": 1,
  "lookahead": 0,
  "has_prev": true,
  "width": 1920,
  "height": 1080,
  "mean_abs": 0.0421,
  "mean_norm": 0.0087,
  "mean_signed": 0.0,
  "status": "ok"
}
```

`width`/`height` are the forwarded (full-resolution) frame dimensions; the HOG
descriptor itself is computed on the `analysis_width` x `analysis_height` plane.

`mean_abs` is the L1 distance between the two frames' HOG descriptors divided by
the descriptor length. `mean_norm` is the same L1 distance divided by the anchor
descriptor's L1 norm (a relative change ratio). `mean_signed` is always 0 for
HOG (emitted only for shape parity with `luma_diff`).

`has_prev` is false for the first frame after node start or after a size/geometry
change. Forward diff slots that run past EOF are emitted with
`status="no_forward_frame"` and `has_prev=false`, exactly like `luma_diff`.

`cuda_hog_diff` is accepted as a legacy node type alias.
