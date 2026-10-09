// Draw filled circles at keypoint positions on an NV12 CUDA frame.
#include <stdint.h>
#include <cuda_runtime.h>

#include "draw_batch_shared.hpp"
#include "draw_primitives.cuh"

using cuda_overlay::KeypointPos;

extern "C" __global__ void kDrawKeypointsNV12Luma(
    uint8_t* __restrict__ y_plane, size_t pitch_y,
    const KeypointPos* __restrict__ points, int num_points,
    int radius, int y_color,
    int frame_w, int frame_h)
{
    const int px = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int py = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (px >= frame_w || py >= frame_h) return;

    for (int i = 0; i < num_points; ++i) {
        if (inside_keypoint(px, py, points[i].x, points[i].y, radius)) {
            y_plane[(size_t)py * pitch_y + (size_t)px] = (uint8_t)y_color;
            return;
        }
    }
}

extern "C" __global__ void kDrawKeypointsNV12Chroma(
    uint8_t* __restrict__ uv_plane, size_t pitch_uv,
    const KeypointPos* __restrict__ points, int num_points,
    int radius, int u_color, int v_color,
    int frame_w, int frame_h)
{
    const int uv_x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int uv_y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int uv_frame_w = (frame_w + 1) >> 1;
    const int uv_frame_h = (frame_h + 1) >> 1;
    if (uv_x >= uv_frame_w || uv_y >= uv_frame_h) return;

    const int luma_x = (uv_x << 1);
    const int luma_y = (uv_y << 1);

    for (int i = 0; i < num_points; ++i) {
        if (inside_keypoint(luma_x, luma_y, points[i].x, points[i].y, radius)) {
            uint8_t* row = uv_plane + (size_t)uv_y * pitch_uv;
            row[(size_t)(uv_x << 1) + 0] = (uint8_t)u_color;
            row[(size_t)(uv_x << 1) + 1] = (uint8_t)v_color;
            return;
        }
    }
}
