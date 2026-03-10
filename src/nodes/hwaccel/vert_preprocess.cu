// NV12 CUDA frame -> TCHW float tensor (grayscale or RGB/BGR), pixel values in [0,255].
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float clamp255(float v) {
    return v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
}

__device__ __forceinline__ void nv12_to_rgb255(
    uint8_t y8, uint8_t u8, uint8_t v8,
    float& r, float& g, float& b)
{
    // Match the current CUDA video path's BT.709 limited-range approximation.
    float y = (float)y8;
    float u = (float)u8 - 128.0f;
    float v = (float)v8 - 128.0f;

    float yf = 1.164384f * (y - 16.0f);
    float rf = yf + 1.792741f * v;
    float gf = yf - 0.213249f * u - 0.532909f * v;
    float bf = yf + 2.112402f * u;

    r = clamp255(rf);
    g = clamp255(gf);
    b = clamp255(bf);
}

__device__ __forceinline__ float rgb_to_gray255(float r, float g, float b) {
    return clamp255(0.299f * r + 0.587f * g + 0.114f * b);
}

extern "C" __global__ void kNV12_to_TCHW_rgb_fp32(
    const uint8_t* __restrict__ y_plane, size_t pitch_y,
    const uint8_t* __restrict__ uv_plane, size_t pitch_uv,
    float* __restrict__ out_tchw,
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
        out_tchw[idx + 0 * plane_size] = b;
        out_tchw[idx + 1 * plane_size] = g;
        out_tchw[idx + 2 * plane_size] = r;
    } else {
        out_tchw[idx + 0 * plane_size] = r;
        out_tchw[idx + 1 * plane_size] = g;
        out_tchw[idx + 2 * plane_size] = b;
    }
}

extern "C" __global__ void kNV12_to_TCHW_rgb_fp16(
    const uint8_t* __restrict__ y_plane, size_t pitch_y,
    const uint8_t* __restrict__ uv_plane, size_t pitch_uv,
    __half* __restrict__ out_tchw,
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
        out_tchw[idx + 0 * plane_size] = __float2half_rn(b);
        out_tchw[idx + 1 * plane_size] = __float2half_rn(g);
        out_tchw[idx + 2 * plane_size] = __float2half_rn(r);
    } else {
        out_tchw[idx + 0 * plane_size] = __float2half_rn(r);
        out_tchw[idx + 1 * plane_size] = __float2half_rn(g);
        out_tchw[idx + 2 * plane_size] = __float2half_rn(b);
    }
}

extern "C" __global__ void kNV12_to_TCHW_gray_fp32(
    const uint8_t* __restrict__ y_plane, size_t pitch_y,
    const uint8_t* __restrict__ uv_plane, size_t pitch_uv,
    float* __restrict__ out_tchw,
    int width, int height)
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
    out_tchw[y * width + x] = rgb_to_gray255(r, g, b);
}

extern "C" __global__ void kNV12_to_TCHW_gray_fp16(
    const uint8_t* __restrict__ y_plane, size_t pitch_y,
    const uint8_t* __restrict__ uv_plane, size_t pitch_uv,
    __half* __restrict__ out_tchw,
    int width, int height)
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
    out_tchw[y * width + x] = __float2half_rn(rgb_to_gray255(r, g, b));
}
