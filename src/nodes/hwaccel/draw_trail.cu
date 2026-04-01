// Draw a trail (connected line segments) onto an NV12 CUDA frame.
#include <stdint.h>
#include <cuda_runtime.h>

// Line segment stored as (x0, y0, x1, y1)
struct LineSegment {
    int x0, y0, x1, y1;
};

namespace {
// Squared distance from point (px, py) to line segment (ax, ay)-(bx, by)
__device__ __forceinline__ float point_segment_dist_sq(int px, int py,
                                                        int ax, int ay,
                                                        int bx, int by) {
    float dx = (float)(bx - ax);
    float dy = (float)(by - ay);
    float len_sq = dx * dx + dy * dy;
    if (len_sq < 1e-6f) {
        // Degenerate segment (point)
        float ex = (float)(px - ax);
        float ey = (float)(py - ay);
        return ex * ex + ey * ey;
    }
    float t = ((float)(px - ax) * dx + (float)(py - ay) * dy) / len_sq;
    t = fmaxf(0.0f, fminf(1.0f, t));
    float proj_x = (float)ax + t * dx;
    float proj_y = (float)ay + t * dy;
    float ex = (float)px - proj_x;
    float ey = (float)py - proj_y;
    return ex * ex + ey * ey;
}

__device__ __forceinline__ bool on_trail(int px, int py,
                                          const LineSegment* segments, int num_segments,
                                          float thickness_sq) {
    for (int i = 0; i < num_segments; ++i) {
        float d2 = point_segment_dist_sq(px, py,
                                          segments[i].x0, segments[i].y0,
                                          segments[i].x1, segments[i].y1);
        if (d2 <= thickness_sq) return true;
    }
    return false;
}
} // namespace

extern "C" __global__ void kDrawTrailNV12Luma(
    uint8_t* __restrict__ y_plane, size_t pitch_y,
    int width, int height,
    const LineSegment* __restrict__ segments, int num_segments,
    float thickness_sq,
    int y_color)
{
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) return;

    if (on_trail(x, y, segments, num_segments, thickness_sq)) {
        y_plane[(size_t)y * pitch_y + (size_t)x] = (uint8_t)y_color;
    }
}

extern "C" __global__ void kDrawTrailNV12Chroma(
    uint8_t* __restrict__ uv_plane, size_t pitch_uv,
    int width, int height,
    const LineSegment* __restrict__ segments, int num_segments,
    float thickness_sq,
    int u_color, int v_color)
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

    const bool draw =
        on_trail(px0, py0, segments, num_segments, thickness_sq) ||
        (px1 < width && on_trail(px1, py0, segments, num_segments, thickness_sq)) ||
        (py1 < height && on_trail(px0, py1, segments, num_segments, thickness_sq)) ||
        (px1 < width && py1 < height && on_trail(px1, py1, segments, num_segments, thickness_sq));

    if (draw) {
        uint8_t* row = uv_plane + (size_t)uv_y * pitch_uv;
        row[(size_t)(uv_x << 1) + 0] = (uint8_t)u_color;
        row[(size_t)(uv_x << 1) + 1] = (uint8_t)v_color;
    }
}
