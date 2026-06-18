#include <stdint.h>

__device__ __forceinline__ float clampf_doctr(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ void nv12_rgb01_doctr(
    const uint8_t* y_plane, int y_pitch,
    const uint8_t* uv_plane, int uv_pitch,
    int x, int y, float& r, float& g, float& b)
{
    uint8_t yy = y_plane[y * y_pitch + x];
    int uvx = x >> 1;
    int uvy = y >> 1;
    uint8_t uu = uv_plane[uvy * uv_pitch + (uvx << 1) + 0];
    uint8_t vv = uv_plane[uvy * uv_pitch + (uvx << 1) + 1];

    float yf = 1.164384f * ((float)yy - 16.0f);
    float u = (float)uu - 128.0f;
    float v = (float)vv - 128.0f;
    r = clampf_doctr(yf + 1.792741f * v, 0.0f, 255.0f) / 255.0f;
    g = clampf_doctr(yf - 0.213249f * u - 0.532909f * v, 0.0f, 255.0f) / 255.0f;
    b = clampf_doctr(yf + 2.112402f * u, 0.0f, 255.0f) / 255.0f;
}

extern "C" __global__ void kNV12_doctr_crop_resize_pad_f32(
    const uint8_t* __restrict__ y_plane, int y_pitch,
    const uint8_t* __restrict__ uv_plane, int uv_pitch,
    float* __restrict__ out_nchw,
    const int* __restrict__ boxes_xywh,
    int batch, int dst_h, int dst_w,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b)
{
    int dx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int dy = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    int bi = (int)blockIdx.z;
    if (bi >= batch || dx >= dst_w || dy >= dst_h) return;

    const int* box = boxes_xywh + bi * 4;
    int sx0 = box[0];
    int sy0 = box[1];
    int sw = box[2];
    int sh = box[3];
    const int plane = dst_w * dst_h;
    const int batch_stride = plane * 3;
    const int out_idx = bi * batch_stride + dy * dst_w + dx;

    float scale = fminf((float)dst_w / fmaxf(1.0f, (float)sw),
                        (float)dst_h / fmaxf(1.0f, (float)sh));
    int content_w = max(1, min(dst_w, (int)floorf((float)sw * scale + 0.5f)));
    int content_h = max(1, min(dst_h, (int)floorf((float)sh * scale + 0.5f)));
    int pad_x = (dst_w - content_w) / 2;
    int pad_y = (dst_h - content_h) / 2;

    float r = 0.0f, g = 0.0f, b = 0.0f;
    if (sw > 0 && sh > 0 && dx >= pad_x && dx < pad_x + content_w &&
        dy >= pad_y && dy < pad_y + content_h) {
        float sx = (float)sx0 + ((float)(dx - pad_x) + 0.5f) * ((float)sw / (float)content_w) - 0.5f;
        float sy = (float)sy0 + ((float)(dy - pad_y) + 0.5f) * ((float)sh / (float)content_h) - 0.5f;
        int px = (int)floorf(sx + 0.5f);
        int py = (int)floorf(sy + 0.5f);
        px = max(sx0, min(sx0 + sw - 1, px));
        py = max(sy0, min(sy0 + sh - 1, py));
        nv12_rgb01_doctr(y_plane, y_pitch, uv_plane, uv_pitch, px, py, r, g, b);
    }

    out_nchw[out_idx + 0 * plane] = (r - mean_r) / std_r;
    out_nchw[out_idx + 1 * plane] = (g - mean_g) / std_g;
    out_nchw[out_idx + 2 * plane] = (b - mean_b) / std_b;
}

