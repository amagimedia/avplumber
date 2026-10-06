# luma_diff

Pass-through CUDA video node that computes lightweight luma-difference metrics on
consecutive frames and writes them to frame metadata.

The node accepts linear CUDA frames with NV12, YUV420P, YUVJ420P or GRAY8 storage,
and native NV12 CUarray frames. Storage is detected per frame. CUarray luma is
sampled directly; the existing GPU history buffers and copy count are unchanged.
Input storage and timestamps pass through unchanged, with compact JSON metadata.

## Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `src` | edge | required | Input `VideoFrame` edge. |
| `dst` | edge | required | Output `VideoFrame` edge. |
| `metadata_key` | string | `scene_diff` | Metadata key to write. |
| `strict_cuda` | bool | `true` | Throw on non-CUDA/unsupported input instead of writing skipped metadata. |
| `debug_log_every_n` | int | `0` | Log one sample every N frames. |

## Metadata

```json
{
  "frame_index": 1,
  "has_prev": true,
  "width": 320,
  "height": 180,
  "mean_abs": 4.15,
  "mean_norm": 0.0012,
  "mean_signed": -0.31,
  "status": "ok"
}
```

`has_prev` is false for the first frame after node start or after a size change.

`cuda_scene_diff` remains accepted as a legacy node type alias.
