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

template <typename T>
__device__ __forceinline__ void store_pixel(
    T* out_nchw,
    int out_w,
    int out_h,
    int batch_index,
    int x,
    int y,
    float r,
    float g,
    float b)
{
    const int plane_size = out_w * out_h;
    const int sample_offset = batch_index * 3 * plane_size;
    const int idx = y * out_w + x;
    const float nr = r * 2.0f - 1.0f;
    const float ng = g * 2.0f - 1.0f;
    const float nb = b * 2.0f - 1.0f;
    out_nchw[sample_offset + 0 * plane_size + idx] = (T)nr;
    out_nchw[sample_offset + 1 * plane_size + idx] = (T)ng;
    out_nchw[sample_offset + 2 * plane_size + idx] = (T)nb;
}

template <>
__device__ __forceinline__ void store_pixel<__half>(
    __half* out_nchw,
    int out_w,
    int out_h,
    int batch_index,
    int x,
    int y,
    float r,
    float g,
    float b)
{
    const int plane_size = out_w * out_h;
    const int sample_offset = batch_index * 3 * plane_size;
    const int idx = y * out_w + x;
    const float nr = r * 2.0f - 1.0f;
    const float ng = g * 2.0f - 1.0f;
    const float nb = b * 2.0f - 1.0f;
    out_nchw[sample_offset + 0 * plane_size + idx] = __float2half_rn(nr);
    out_nchw[sample_offset + 1 * plane_size + idx] = __float2half_rn(ng);
    out_nchw[sample_offset + 2 * plane_size + idx] = __float2half_rn(nb);
}

template <typename T>
__device__ __forceinline__ void runPlayerMaskToNCHW(
    const uint8_t* __restrict__ y_plane,
    size_t pitch_y,
    const uint8_t* __restrict__ uv_plane,
    size_t pitch_uv,
    int frame_w,
    int frame_h,
    const float* __restrict__ masks,
    int proto_w,
    int proto_h,
    int model_w,
    int model_h,
    int body_region_mode,
    float torso_x_margin_rel,
    float torso_y_start_rel,
    float torso_y_end_rel,
    const int* __restrict__ bboxes_xyxy,
    const int* __restrict__ plane_indices,
    int out_w,
    int out_h,
    float mask_threshold,
    float bg_value,
    T* __restrict__ out_nchw)
{
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int sample = (int)blockIdx.z;
    if (x >= out_w || y >= out_h) return;

    const int* bbox = bboxes_xyxy + sample * 4;
    const int plane_index = plane_indices[sample];
    if (plane_index < 0) {
        store_pixel(out_nchw, out_w, out_h, sample, x, y, bg_value, bg_value, bg_value);
        return;
    }

    const float x1 = (float)bbox[0];
    const float y1 = (float)bbox[1];
    const float x2 = (float)bbox[2];
    const float y2 = (float)bbox[3];
    float crop_x1 = x1;
    float crop_y1 = y1;
    float crop_x2 = x2;
    float crop_y2 = y2;
    const float bw = max(1.0f, x2 - x1);
    const float bh = max(1.0f, y2 - y1);
    if (body_region_mode == 1) {
        crop_x1 = x1 + torso_x_margin_rel * bw;
        crop_x2 = x2 - torso_x_margin_rel * bw;
        crop_x1 = min(crop_x1, x2 - 1.0f);
        crop_x2 = max(crop_x2, crop_x1 + 1.0f);
        crop_y1 = y1 + torso_y_start_rel * bh;
        const float torso_h = max(1.0f, (torso_y_end_rel - torso_y_start_rel) * bh);
        crop_y2 = min(y2, crop_y1 + torso_h);
        crop_y1 = min(crop_y1, y2 - 1.0f);
        crop_y2 = max(crop_y2, crop_y1 + 1.0f);
    }
    const float crop_w = max(1.0f, crop_x2 - crop_x1);
    const float crop_h = max(1.0f, crop_y2 - crop_y1);
    const float side = max(crop_w, crop_h);
    const float cx = 0.5f * (crop_x1 + crop_x2);
    const float cy = 0.5f * (crop_y1 + crop_y2);
    const float crop_x0 = cx - 0.5f * side;
    const float crop_y0 = cy - 0.5f * side;

    const float src_x = crop_x0 + ((float)x + 0.5f) * side / (float)out_w;
    const float src_y = crop_y0 + ((float)y + 0.5f) * side / (float)out_h;

    if (src_x < 0.0f || src_y < 0.0f || src_x >= (float)frame_w || src_y >= (float)frame_h) {
        store_pixel(out_nchw, out_w, out_h, sample, x, y, bg_value, bg_value, bg_value);
        return;
    }

    const int mx = min(proto_w - 1, max(0, (int)(src_x * (float)proto_w / (float)model_w)));
    const int my = min(proto_h - 1, max(0, (int)(src_y * (float)proto_h / (float)model_h)));
    const float mask_value = masks[(size_t)plane_index * (size_t)proto_w * (size_t)proto_h + (size_t)my * (size_t)proto_w + (size_t)mx];
    if (mask_value < mask_threshold) {
        store_pixel(out_nchw, out_w, out_h, sample, x, y, bg_value, bg_value, bg_value);
        return;
    }

    const int sx = min(frame_w - 1, max(0, (int)src_x));
    const int sy = min(frame_h - 1, max(0, (int)src_y));

    const uint8_t* y_row = y_plane + (size_t)sy * pitch_y;
    const int uvx = sx >> 1;
    const int uvy = sy >> 1;
    const uint8_t* uv_row = uv_plane + (size_t)uvy * pitch_uv;
    const uint8_t Y = y_row[sx];
    const uint8_t U = uv_row[(uvx << 1) + 0];
    const uint8_t V = uv_row[(uvx << 1) + 1];

    float r, g, b;
    nv12_to_rgb(Y, U, V, r, g, b);
    store_pixel(out_nchw, out_w, out_h, sample, x, y, r, g, b);
}

extern "C" __global__ void kPlayerMaskToNCHW_fp32(
    const uint8_t* y_plane,
    size_t pitch_y,
    const uint8_t* uv_plane,
    size_t pitch_uv,
    int frame_w,
    int frame_h,
    const float* masks,
    int proto_w,
    int proto_h,
    int model_w,
    int model_h,
    int body_region_mode,
    float torso_x_margin_rel,
    float torso_y_start_rel,
    float torso_y_end_rel,
    const int* bboxes_xyxy,
    const int* plane_indices,
    int out_w,
    int out_h,
    float mask_threshold,
    float bg_value,
    float* out_nchw)
{
    runPlayerMaskToNCHW<float>(
        y_plane, pitch_y, uv_plane, pitch_uv,
        frame_w, frame_h, masks, proto_w, proto_h, model_w, model_h,
        body_region_mode, torso_x_margin_rel, torso_y_start_rel, torso_y_end_rel,
        bboxes_xyxy, plane_indices, out_w, out_h, mask_threshold, bg_value, out_nchw);
}

extern "C" __global__ void kPlayerMaskToNCHW_fp16(
    const uint8_t* y_plane,
    size_t pitch_y,
    const uint8_t* uv_plane,
    size_t pitch_uv,
    int frame_w,
    int frame_h,
    const float* masks,
    int proto_w,
    int proto_h,
    int model_w,
    int model_h,
    int body_region_mode,
    float torso_x_margin_rel,
    float torso_y_start_rel,
    float torso_y_end_rel,
    const int* bboxes_xyxy,
    const int* plane_indices,
    int out_w,
    int out_h,
    float mask_threshold,
    float bg_value,
    __half* out_nchw)
{
    runPlayerMaskToNCHW<__half>(
        y_plane, pitch_y, uv_plane, pitch_uv,
        frame_w, frame_h, masks, proto_w, proto_h, model_w, model_h,
        body_region_mode, torso_x_margin_rel, torso_y_start_rel, torso_y_end_rel,
        bboxes_xyxy, plane_indices, out_w, out_h, mask_threshold, bg_value, out_nchw);
}
