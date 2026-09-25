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

// Classifier-specific letterbox. Average RGB pixels over the source footprint
// (OpenCV INTER_AREA semantics) before packing the uint8 NCHW tensor. This
// avoids downscaling NV12 chroma/luma separately with scale_cuda: at 224 px
// that path changes the classifier's Inplay/Outplay probabilities materially.
extern "C" __global__ void kNV12_area_to_NCHW_u8(
    const uint8_t* __restrict__ y_plane, size_t pitch_y,
    const uint8_t* __restrict__ uv_plane, size_t pitch_uv,
    uint8_t* __restrict__ out_nchw,
    int source_w, int source_h,
    int output_w, int output_h,
    int content_w, int content_h,
    int offset_x, int offset_y,
    int bgr_order)
{
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= output_w || y >= output_h) return;
    const int idx = y * output_w + x;
    const int plane_size = output_w * output_h;
    const int cx = x - offset_x;
    const int cy = y - offset_y;
    if (cx < 0 || cx >= content_w || cy < 0 || cy >= content_h) {
        out_nchw[idx] = 0;
        out_nchw[idx + plane_size] = 0;
        out_nchw[idx + 2 * plane_size] = 0;
        return;
    }

    const float left = (float)cx * source_w / content_w;
    const float right = (float)(cx + 1) * source_w / content_w;
    const float top = (float)cy * source_h / content_h;
    const float bottom = (float)(cy + 1) * source_h / content_h;
    float red = 0.0f, green = 0.0f, blue = 0.0f;
    for (int sy = (int)top; sy < (int)ceilf(bottom); ++sy) {
        const float wy = fminf(bottom, (float)(sy + 1)) - fmaxf(top, (float)sy);
        const uint8_t* yr = y_plane + (size_t)sy * pitch_y;
        const uint8_t* uvr = uv_plane + (size_t)(sy >> 1) * pitch_uv;
        for (int sx = (int)left; sx < (int)ceilf(right); ++sx) {
            const float weight = wy * (fminf(right, (float)(sx + 1)) - fmaxf(left, (float)sx));
            const float Y = (float)yr[sx];
            const float U = (float)uvr[(sx & ~1)] - 128.0f;
            const float V = (float)uvr[(sx & ~1) + 1] - 128.0f;
            const float yf = 1.164384f * (Y - 16.0f);
            red += weight * (float)clamp255(yf + 1.792741f * V);
            green += weight * (float)clamp255(yf - 0.213249f * U - 0.532909f * V);
            blue += weight * (float)clamp255(yf + 2.112402f * U);
        }
    }
    const float inv_area = 1.0f / ((right - left) * (bottom - top));
    out_nchw[idx] = clamp255((bgr_order ? blue : red) * inv_area);
    out_nchw[idx + plane_size] = clamp255(green * inv_area);
    out_nchw[idx + 2 * plane_size] = clamp255((bgr_order ? red : blue) * inv_area);
}
