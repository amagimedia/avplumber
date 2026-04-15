// Draw a configurable-color bounding box onto an NV12 CUDA frame.
#include <stdint.h>
#include <cuda_runtime.h>

#include "draw_batch_shared.hpp"

namespace {
__device__ __forceinline__ bool inside_bbox_border(int x, int y,
                                                   int x1, int y1,
                                                   int x2, int y2,
                                                   int thickness) {
    if (x < x1 || x >= x2 || y < y1 || y >= y2) return false;
    return x < (x1 + thickness) || x >= (x2 - thickness)
        || y < (y1 + thickness) || y >= (y2 - thickness);
}
}

extern "C" __global__ void kDrawBBoxNV12Luma(
    uint8_t* __restrict__ y_plane, size_t pitch_y,
    int width, int height,
    const cuda_overlay::BatchedBBox* __restrict__ boxes,
    int num_boxes)
{
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) return;

    uint8_t* y_px = &y_plane[(size_t)y * pitch_y + (size_t)x];
    for (int i = 0; i < num_boxes; ++i) {
        const cuda_overlay::BatchedBBox box = boxes[i];
        if (inside_bbox_border(x, y, box.x1, box.y1, box.x2, box.y2, box.thickness)) {
            *y_px = (uint8_t)box.y_color;
        }
    }
}

extern "C" __global__ void kDrawBBoxNV12Chroma(
    uint8_t* __restrict__ uv_plane, size_t pitch_uv,
    int width, int height,
    const cuda_overlay::BatchedBBox* __restrict__ boxes,
    int num_boxes)
{
    const int uv_x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int uv_y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int uv_width = (width + 1) >> 1;
    const int uv_height = (height + 1) >> 1;
    if (uv_x >= uv_width || uv_y >= uv_height) return;

    const int px0 = uv_x << 1;
    const int py0 = uv_y << 1;
    const int px1 = px0 + 1;
    const int py1 = py0 + 1;

    uint8_t* row = uv_plane + (size_t)uv_y * pitch_uv;
    const size_t uv_idx = (size_t)(uv_x << 1);
    for (int i = 0; i < num_boxes; ++i) {
        const cuda_overlay::BatchedBBox box = boxes[i];
        const bool draw =
            inside_bbox_border(px0, py0, box.x1, box.y1, box.x2, box.y2, box.thickness) ||
            (px1 < width && inside_bbox_border(px1, py0, box.x1, box.y1, box.x2, box.y2, box.thickness)) ||
            (py1 < height && inside_bbox_border(px0, py1, box.x1, box.y1, box.x2, box.y2, box.thickness)) ||
            (px1 < width && py1 < height && inside_bbox_border(px1, py1, box.x1, box.y1, box.x2, box.y2, box.thickness));

        if (draw) {
            row[uv_idx + 0] = (uint8_t)box.u_color;
            row[uv_idx + 1] = (uint8_t)box.v_color;
        }
    }
}
