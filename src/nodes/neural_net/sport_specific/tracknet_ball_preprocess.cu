// TrackNet triplet preprocessor: three NV12 CUDA frames -> NCHW9 RGB tensor.
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ float clamp01(float v) {
    return clampf(v, 0.0f, 1.0f);
}

__device__ __forceinline__ float sample_luma(
    const uint8_t* __restrict__ plane, size_t pitch,
    int width, int height, float fx, float fy)
{
    fx = clampf(fx, 0.0f, (float)(width - 1));
    fy = clampf(fy, 0.0f, (float)(height - 1));

    const int x0 = (int)floorf(fx);
    const int y0 = (int)floorf(fy);
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const float wx = fx - (float)x0;
    const float wy = fy - (float)y0;

    const uint8_t* row0 = plane + (size_t)y0 * pitch;
    const uint8_t* row1 = plane + (size_t)y1 * pitch;
    const float v00 = (float)row0[x0];
    const float v01 = (float)row0[x1];
    const float v10 = (float)row1[x0];
    const float v11 = (float)row1[x1];
    const float top = v00 + (v01 - v00) * wx;
    const float bot = v10 + (v11 - v10) * wx;
    return top + (bot - top) * wy;
}

__device__ __forceinline__ float sample_chroma(
    const uint8_t* __restrict__ uv_plane, size_t pitch,
    int width, int height, float fx, float fy, int component)
{
    fx = clampf(fx, 0.0f, (float)(width - 1));
    fy = clampf(fy, 0.0f, (float)(height - 1));

    const int x0 = (int)floorf(fx);
    const int y0 = (int)floorf(fy);
    const int x1 = min(x0 + 1, width - 1);
    const int y1 = min(y0 + 1, height - 1);
    const float wx = fx - (float)x0;
    const float wy = fy - (float)y0;

    const uint8_t* row0 = uv_plane + (size_t)y0 * pitch;
    const uint8_t* row1 = uv_plane + (size_t)y1 * pitch;
    const float v00 = (float)row0[(x0 << 1) + component];
    const float v01 = (float)row0[(x1 << 1) + component];
    const float v10 = (float)row1[(x0 << 1) + component];
    const float v11 = (float)row1[(x1 << 1) + component];
    const float top = v00 + (v01 - v00) * wx;
    const float bot = v10 + (v11 - v10) * wx;
    return top + (bot - top) * wy;
}

__device__ __forceinline__ void nv12_to_rgb(
    float y8, float u8, float v8,
    float& r, float& g, float& b)
{
    // BT.709 limited-range approximation, matching the existing YOLO CUDA preprocess path.
    const float u = u8 - 128.0f;
    const float v = v8 - 128.0f;
    const float yf = 1.164384f * (y8 - 16.0f);
    r = clamp01((yf + 1.792741f * v) / 255.0f);
    g = clamp01((yf - 0.213249f * u - 0.532909f * v) / 255.0f);
    b = clamp01((yf + 2.112402f * u) / 255.0f);
}

__device__ __forceinline__ void sample_frame_rgb(
    const uint8_t* __restrict__ y_plane, size_t pitch_y,
    const uint8_t* __restrict__ uv_plane, size_t pitch_uv,
    int src_w, int src_h, int dst_w, int dst_h,
    int dst_x, int dst_y,
    float& r, float& g, float& b)
{
    const float sx = ((float)dst_x + 0.5f) * ((float)src_w / (float)dst_w) - 0.5f;
    const float sy = ((float)dst_y + 0.5f) * ((float)src_h / (float)dst_h) - 0.5f;
    const int uv_w = max(1, (src_w + 1) >> 1);
    const int uv_h = max(1, (src_h + 1) >> 1);
    const float uvx = sx * 0.5f;
    const float uvy = sy * 0.5f;

    const float yv = sample_luma(y_plane, pitch_y, src_w, src_h, sx, sy);
    const float u = sample_chroma(uv_plane, pitch_uv, uv_w, uv_h, uvx, uvy, 0);
    const float v = sample_chroma(uv_plane, pitch_uv, uv_w, uv_h, uvx, uvy, 1);
    nv12_to_rgb(yv, u, v, r, g, b);
}

__device__ __forceinline__ void store_tracknet_pixel(
    float* __restrict__ out, int frame_idx, int dst_idx, int plane_size,
    float r, float g, float b)
{
    const int c = frame_idx * 3;
    out[dst_idx + (c + 0) * plane_size] = (r - 0.485f) / 0.229f;
    out[dst_idx + (c + 1) * plane_size] = (g - 0.456f) / 0.224f;
    out[dst_idx + (c + 2) * plane_size] = (b - 0.406f) / 0.225f;
}

__device__ __forceinline__ void store_tracknet_pixel(
    __half* __restrict__ out, int frame_idx, int dst_idx, int plane_size,
    float r, float g, float b)
{
    const int c = frame_idx * 3;
    out[dst_idx + (c + 0) * plane_size] = __float2half_rn((r - 0.485f) / 0.229f);
    out[dst_idx + (c + 1) * plane_size] = __float2half_rn((g - 0.456f) / 0.224f);
    out[dst_idx + (c + 2) * plane_size] = __float2half_rn((b - 0.406f) / 0.225f);
}

template <typename OutT>
__device__ __forceinline__ void preprocess_triplet(
    const uint8_t* __restrict__ y0, size_t pitch_y0,
    const uint8_t* __restrict__ uv0, size_t pitch_uv0,
    const uint8_t* __restrict__ y1, size_t pitch_y1,
    const uint8_t* __restrict__ uv1, size_t pitch_uv1,
    const uint8_t* __restrict__ y2, size_t pitch_y2,
    const uint8_t* __restrict__ uv2, size_t pitch_uv2,
    OutT* __restrict__ out_nchw,
    int src_w, int src_h, int dst_w, int dst_h)
{
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= dst_w || y >= dst_h) return;

    const int plane_size = dst_w * dst_h;
    const int dst_idx = y * dst_w + x;

    float r, g, b;
    sample_frame_rgb(y0, pitch_y0, uv0, pitch_uv0, src_w, src_h, dst_w, dst_h, x, y, r, g, b);
    store_tracknet_pixel(out_nchw, 0, dst_idx, plane_size, r, g, b);

    sample_frame_rgb(y1, pitch_y1, uv1, pitch_uv1, src_w, src_h, dst_w, dst_h, x, y, r, g, b);
    store_tracknet_pixel(out_nchw, 1, dst_idx, plane_size, r, g, b);

    sample_frame_rgb(y2, pitch_y2, uv2, pitch_uv2, src_w, src_h, dst_w, dst_h, x, y, r, g, b);
    store_tracknet_pixel(out_nchw, 2, dst_idx, plane_size, r, g, b);
}

extern "C" __global__ void kTrackNetNV12TripletToNCHW9_fp32(
    const uint8_t* __restrict__ y0, size_t pitch_y0,
    const uint8_t* __restrict__ uv0, size_t pitch_uv0,
    const uint8_t* __restrict__ y1, size_t pitch_y1,
    const uint8_t* __restrict__ uv1, size_t pitch_uv1,
    const uint8_t* __restrict__ y2, size_t pitch_y2,
    const uint8_t* __restrict__ uv2, size_t pitch_uv2,
    float* __restrict__ out_nchw,
    int src_w, int src_h, int dst_w, int dst_h)
{
    preprocess_triplet(y0, pitch_y0, uv0, pitch_uv0,
                       y1, pitch_y1, uv1, pitch_uv1,
                       y2, pitch_y2, uv2, pitch_uv2,
                       out_nchw, src_w, src_h, dst_w, dst_h);
}

extern "C" __global__ void kTrackNetNV12TripletToNCHW9_fp16(
    const uint8_t* __restrict__ y0, size_t pitch_y0,
    const uint8_t* __restrict__ uv0, size_t pitch_uv0,
    const uint8_t* __restrict__ y1, size_t pitch_y1,
    const uint8_t* __restrict__ uv1, size_t pitch_uv1,
    const uint8_t* __restrict__ y2, size_t pitch_y2,
    const uint8_t* __restrict__ uv2, size_t pitch_uv2,
    __half* __restrict__ out_nchw,
    int src_w, int src_h, int dst_w, int dst_h)
{
    preprocess_triplet(y0, pitch_y0, uv0, pitch_uv0,
                       y1, pitch_y1, uv1, pitch_uv1,
                       y2, pitch_y2, uv2, pitch_uv2,
                       out_nchw, src_w, src_h, dst_w, dst_h);
}
