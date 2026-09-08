// NV12 CUDA frame -> CHW RGB float32 tensor, normalized to [0,1] (img/255.0).
//
// Model contract: RGB channel order, no ImageNet mean/std, plain /255 scaling.
//
// No resize is performed here: the caller (avplumber graph) must already have
// scaled the source frame to the model's exact input width/height (e.g. via
// scale_npp) before this node's preprocess kernel runs.
#include <stdint.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float clamp01(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

__device__ __forceinline__ void nv12_to_rgb01(
    uint8_t y8, uint8_t u8, uint8_t v8,
    float& r, float& g, float& b)
{
    // BT.709 limited-range approximation.
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

// Writes into a CHW (3, height, width) float32 buffer starting at out_chw — the
// caller positions out_chw at the correct tensor's device address (frame_a or
// frame_b), so a single kernel serves both inputs.
extern "C" __global__ void kNV12_to_CHW_rgb01_fp32(
    const uint8_t* __restrict__ y_plane, size_t pitch_y,
    const uint8_t* __restrict__ uv_plane, size_t pitch_uv,
    float* __restrict__ out_chw,
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
    nv12_to_rgb01(Y, U, V, r, g, b);

    const int plane_size = width * height;
    const int idx = y * width + x;
    out_chw[idx + 0 * plane_size] = r;
    out_chw[idx + 1 * plane_size] = g;
    out_chw[idx + 2 * plane_size] = b;
}
