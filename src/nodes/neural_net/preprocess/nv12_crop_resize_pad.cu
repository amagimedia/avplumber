// NV12 crop + bilinear resize + pad → RGB NCHW float32 normalized to [-1,1].
// Used by scoreboard_ocr to preprocess small bbox crops for PP-OCR TRT inference.
#include <stdint.h>
#include <cuda_runtime.h>

__device__ __forceinline__ float clamp_f(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ void nv12_to_rgb_norm(
    uint8_t y8, uint8_t u8, uint8_t v8,
    float& r, float& g, float& b)
{
    float y = (float)y8;
    float u = (float)u8 - 128.0f;
    float v = (float)v8 - 128.0f;
    float yf = 1.164384f * (y - 16.0f);
    r = clamp_f((yf + 1.792741f * v) / 127.5f - 1.0f, -1.0f, 1.0f);
    g = clamp_f((yf - 0.213249f * u - 0.532909f * v) / 127.5f - 1.0f, -1.0f, 1.0f);
    b = clamp_f((yf + 2.112402f * u) / 127.5f - 1.0f, -1.0f, 1.0f);
}

extern "C" __global__ void kNV12_crop_resize_pad_RGB(
    const uint8_t* __restrict__ y_plane, int y_pitch,
    const uint8_t* __restrict__ uv_plane, int uv_pitch,
    float* __restrict__ out_nchw,
    int src_x1, int src_y1, int src_w, int src_h,
    int dst_h, int dst_w_content, int dst_w_padded)
{
    const int dx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int dy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (dx >= dst_w_padded || dy >= dst_h) return;

    const int plane_size = dst_w_padded * dst_h;
    const int idx = dy * dst_w_padded + dx;

    if (dx >= dst_w_content) {
        out_nchw[idx + 0 * plane_size] = 0.0f;
        out_nchw[idx + 1 * plane_size] = 0.0f;
        out_nchw[idx + 2 * plane_size] = 0.0f;
        return;
    }

    float sx = (float)src_x1 + ((float)dx + 0.5f) * ((float)src_w / (float)dst_w_content) - 0.5f;
    float sy = (float)src_y1 + ((float)dy + 0.5f) * ((float)src_h / (float)dst_h) - 0.5f;
    sx = clamp_f(sx, (float)src_x1, (float)(src_x1 + src_w - 1));
    sy = clamp_f(sy, (float)src_y1, (float)(src_y1 + src_h - 1));

    int sx0 = (int)sx;
    int sy0 = (int)sy;
    int sx1 = sx0 + 1 < src_x1 + src_w ? sx0 + 1 : sx0;
    int sy1 = sy0 + 1 < src_y1 + src_h ? sy0 + 1 : sy0;
    float fx = sx - (float)sx0;
    float fy = sy - (float)sy0;

    float r00, g00, b00, r10, g10, b10, r01, g01, b01, r11, g11, b11;

#define SAMPLE_NV12(px, py, rv, gv, bv) do { \
    uint8_t Y = y_plane[(py) * y_pitch + (px)]; \
    int uvx = (px) >> 1; \
    int uvy = (py) >> 1; \
    uint8_t U = uv_plane[uvy * uv_pitch + (uvx << 1) + 0]; \
    uint8_t V = uv_plane[uvy * uv_pitch + (uvx << 1) + 1]; \
    nv12_to_rgb_norm(Y, U, V, rv, gv, bv); \
} while(0)

    SAMPLE_NV12(sx0, sy0, r00, g00, b00);
    SAMPLE_NV12(sx1, sy0, r10, g10, b10);
    SAMPLE_NV12(sx0, sy1, r01, g01, b01);
    SAMPLE_NV12(sx1, sy1, r11, g11, b11);

#undef SAMPLE_NV12

    float w00 = (1.0f - fx) * (1.0f - fy);
    float w10 = fx * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    out_nchw[idx + 0 * plane_size] = r00*w00 + r10*w10 + r01*w01 + r11*w11;
    out_nchw[idx + 1 * plane_size] = g00*w00 + g10*w10 + g01*w01 + g11*w11;
    out_nchw[idx + 2 * plane_size] = b00*w00 + b10*w10 + b01*w01 + b11*w11;
}
