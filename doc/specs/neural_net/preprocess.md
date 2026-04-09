# neural_net/preprocess — CUDA Kernels

PTX-only directory — no nodes, just CUDA kernels compiled and embedded as C headers.

## `nv12_to_nchw.cu`

### `kNV12_to_NCHW_fp32` / `kNV12_to_NCHW_fp16`
NV12 frame → NCHW float/half tensor for TensorRT inference input.

- BT.709 limited-range YUV→RGB conversion
- Normalizes to [0.0, 1.0]
- `bgr` flag swaps R↔B channels
- Grid: (W+31)/32 × (H+7)/8, Block: 32×8

Used by: `cuda_infer_yolo`, `cuda_infer_rtdetr`, `reframer`

## `mask_assemble.cu`

### `kMaskAssemble`
Segmentation mask assembly: coefficient[det,32] · prototype[32,H,W] → mask[det,H,W] with sigmoid activation.

- Grid: (W+31)/32 × (H+7)/8 × num_dets, Block: 32×8

### `kMaskDownsample`
Bilinear downsample of assembled masks for CPU-side consumption.

- Grid: (dst_w+31)/32 × (dst_h+7)/8, Block: 32×8

Used by: `cuda_infer_yolo` (segmentation task)
