// NV12 CUDA frame -> NCHW float tensor (RGB/BGR), normalized to [0,1].
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float clamp01(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

__device__ __forceinline__ void nv12_to_rgb(
    uint8_t y8, uint8_t u8, uint8_t v8,
    float& r, float& g, float& b)
{
    // BT.709 limited-range approximation (suitable for typical video decode path).
    float y = (float)y8;
    float u = (float)u8 - 128.0f;
    float v = (float)v8 - 128.0f;

    float yf = 1.164384f * (y - 16.0f);
    float rf = yf + 1.792741f * v;
    float gf = yf - 0.213249f * u - 0.532909f * v;
    float bf = yf + 2.112402f * u;

    r = clamp01(rf / 255.0f);
    g = clamp01(gf / 255.0f);
    b = clamp01(bf / 255.0f);
}

extern "C" __global__ void kNV12_to_NCHW_fp32(
    const uint8_t* __restrict__ y_plane, size_t pitch_y,
    const uint8_t* __restrict__ uv_plane, size_t pitch_uv,
    float* __restrict__ out_nchw,
    int width, int height,
    int bgr_order)
{
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) return;

    const uint8_t* y_row = y_plane + (size_t)y * pitch_y;
    const int uvx = x >> 1;
    const int uvy = y >> 1;
    const uint8_t* uv_row = uv_plane + (size_t)uvy * pitch_uv;
    const uint8_t Y = y_row[x];
    const uint8_t U = uv_row[(uvx << 1) + 0];
    const uint8_t V = uv_row[(uvx << 1) + 1];

    float r, g, b;
    nv12_to_rgb(Y, U, V, r, g, b);

    const int plane_size = width * height;
    const int idx = y * width + x;
    if (bgr_order) {
        out_nchw[idx + 0 * plane_size] = b;
        out_nchw[idx + 1 * plane_size] = g;
        out_nchw[idx + 2 * plane_size] = r;
    } else {
        out_nchw[idx + 0 * plane_size] = r;
        out_nchw[idx + 1 * plane_size] = g;
        out_nchw[idx + 2 * plane_size] = b;
    }
}

extern "C" __global__ void kNV12_to_NCHW_fp16(
    const uint8_t* __restrict__ y_plane, size_t pitch_y,
    const uint8_t* __restrict__ uv_plane, size_t pitch_uv,
    __half* __restrict__ out_nchw,
    int width, int height,
    int bgr_order)
{
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) return;

    const uint8_t* y_row = y_plane + (size_t)y * pitch_y;
    const int uvx = x >> 1;
    const int uvy = y >> 1;
    const uint8_t* uv_row = uv_plane + (size_t)uvy * pitch_uv;
    const uint8_t Y = y_row[x];
    const uint8_t U = uv_row[(uvx << 1) + 0];
    const uint8_t V = uv_row[(uvx << 1) + 1];

    float r, g, b;
    nv12_to_rgb(Y, U, V, r, g, b);

    const int plane_size = width * height;
    const int idx = y * width + x;
    if (bgr_order) {
        out_nchw[idx + 0 * plane_size] = __float2half_rn(b);
        out_nchw[idx + 1 * plane_size] = __float2half_rn(g);
        out_nchw[idx + 2 * plane_size] = __float2half_rn(r);
    } else {
        out_nchw[idx + 0 * plane_size] = __float2half_rn(r);
        out_nchw[idx + 1 * plane_size] = __float2half_rn(g);
        out_nchw[idx + 2 * plane_size] = __float2half_rn(b);
    }
}

// UINT8 variant: emit raw 0..255 RGB (no /255 normalization). Used for engines
// whose ONNX graph bakes in normalization and expects a uint8 image input
// (e.g. DeepStream/Triton-style YOLO exports).
__device__ __forceinline__ unsigned char clamp255(float v) {
    v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
    return (unsigned char)(v + 0.5f);
}

__device__ __forceinline__ void nv12_to_rgb255(
    uint8_t y8, uint8_t u8, uint8_t v8,
    float& r, float& g, float& b)
{
    float y = (float)y8;
    float u = (float)u8 - 128.0f;
    float v = (float)v8 - 128.0f;

    float yf = 1.164384f * (y - 16.0f);
    r = yf + 1.792741f * v;
    g = yf - 0.213249f * u - 0.532909f * v;
    b = yf + 2.112402f * u;
}

extern "C" __global__ void kNV12_to_NCHW_u8(
    const uint8_t* __restrict__ y_plane, size_t pitch_y,
    const uint8_t* __restrict__ uv_plane, size_t pitch_uv,
    uint8_t* __restrict__ out_nchw,
    int width, int height,
    int bgr_order)
{
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) return;

    const uint8_t* y_row = y_plane + (size_t)y * pitch_y;
    const int uvx = x >> 1;
    const int uvy = y >> 1;
    const uint8_t* uv_row = uv_plane + (size_t)uvy * pitch_uv;
    const uint8_t Y = y_row[x];
    const uint8_t U = uv_row[(uvx << 1) + 0];
    const uint8_t V = uv_row[(uvx << 1) + 1];

    float r, g, b;
    nv12_to_rgb255(Y, U, V, r, g, b);

    const int plane_size = width * height;
    const int idx = y * width + x;
    if (bgr_order) {
        out_nchw[idx + 0 * plane_size] = clamp255(b);
        out_nchw[idx + 1 * plane_size] = clamp255(g);
        out_nchw[idx + 2 * plane_size] = clamp255(r);
    } else {
        out_nchw[idx + 0 * plane_size] = clamp255(r);
        out_nchw[idx + 1 * plane_size] = clamp255(g);
        out_nchw[idx + 2 * plane_size] = clamp255(b);
    }
}

