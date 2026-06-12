#pragma once

#include <cuda_runtime.h>

__device__ __forceinline__ int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ float clamp01(float v) {
    return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
}

__device__ __forceinline__ void nv12_to_rgb709_limited(
    unsigned char y8, unsigned char u8, unsigned char v8,
    float& r, float& g, float& b)
{
    const float y = (float)y8;
    const float u = (float)u8 - 128.0f;
    const float v = (float)v8 - 128.0f;

    const float yf = 1.164384f * (y - 16.0f);
    r = clamp01((yf + 1.792741f * v) / 255.0f);
    g = clamp01((yf - 0.213249f * u - 0.532909f * v) / 255.0f);
    b = clamp01((yf + 2.112402f * u) / 255.0f);
}

__device__ __forceinline__ void sample_nv12_rgb(
    const unsigned char* __restrict__ y_plane, int pitch_y,
    const unsigned char* __restrict__ uv_plane, int pitch_uv,
    int width, int height, int x, int y,
    float& r, float& g, float& b)
{
    x = clamp_int(x, 0, width - 1);
    y = clamp_int(y, 0, height - 1);

    const unsigned char y8 = y_plane[y * pitch_y + x];
    const int uvx = x >> 1;
    const int uvy = y >> 1;
    const unsigned char* uv_row = uv_plane + uvy * pitch_uv;
    const unsigned char u8 = uv_row[(uvx << 1) + 0];
    const unsigned char v8 = uv_row[(uvx << 1) + 1];
    nv12_to_rgb709_limited(y8, u8, v8, r, g, b);
}

__device__ __forceinline__ void sample_nv12_rgb_bilinear(
    const unsigned char* __restrict__ y_plane, int pitch_y,
    const unsigned char* __restrict__ uv_plane, int pitch_uv,
    int width, int height, float sx, float sy,
    float& r, float& g, float& b)
{
    const int x0 = (int)floorf(sx);
    const int y0 = (int)floorf(sy);
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float fx = sx - (float)x0;
    const float fy = sy - (float)y0;

    float r00, g00, b00;
    float r10, g10, b10;
    float r01, g01, b01;
    float r11, g11, b11;
    sample_nv12_rgb(y_plane, pitch_y, uv_plane, pitch_uv, width, height, x0, y0, r00, g00, b00);
    sample_nv12_rgb(y_plane, pitch_y, uv_plane, pitch_uv, width, height, x1, y0, r10, g10, b10);
    sample_nv12_rgb(y_plane, pitch_y, uv_plane, pitch_uv, width, height, x0, y1, r01, g01, b01);
    sample_nv12_rgb(y_plane, pitch_y, uv_plane, pitch_uv, width, height, x1, y1, r11, g11, b11);

    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w10 = fx * (1.0f - fy);
    const float w01 = (1.0f - fx) * fy;
    const float w11 = fx * fy;
    r = r00 * w00 + r10 * w10 + r01 * w01 + r11 * w11;
    g = g00 * w00 + g10 * w10 + g01 * w01 + g11 * w11;
    b = b00 * w00 + b10 * w10 + b01 * w01 + b11 * w11;
}
