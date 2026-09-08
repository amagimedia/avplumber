#include <cuda_runtime.h>

// Bilinear interpolation. This lightweight
// sampler does not widen its support when downscaling. Each lane is sampled
// separately, including interleaved UV.
extern "C" __global__ void scale_plane(
    const unsigned char *src, int src_pitch, int sx, int sy, int sw, int sh,
    unsigned char *dst, int dst_pitch, int dx, int dy, int dw, int dh,
    int canvas_w, int canvas_h, int lanes) {
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    const int x = dx + ox, y = dy + oy;
    if (ox >= dw || oy >= dh || x < 0 || y < 0 || x >= canvas_w || y >= canvas_h) return;
    const float fx = (ox + 0.5f) * sw / dw - 0.5f;
    const float fy = (oy + 0.5f) * sh / dh - 0.5f;
    const int ix = int(floorf(fx)), iy = int(floorf(fy));
    const float tx = fx - ix, ty = fy - iy;
    const int x0 = sx + max(0, min(ix, sw - 1));
    const int x1 = sx + max(0, min(ix + 1, sw - 1));
    const int y0 = sy + max(0, min(iy, sh - 1));
    const int y1 = sy + max(0, min(iy + 1, sh - 1));
    for (int c = 0; c < lanes; ++c) {
        const float a = src[y0 * src_pitch + x0 * lanes + c];
        const float b = src[y0 * src_pitch + x1 * lanes + c];
        const float d = src[y1 * src_pitch + x0 * lanes + c];
        const float e = src[y1 * src_pitch + x1 * lanes + c];
        const float top = a + tx * (b - a), bottom = d + tx * (e - d);
        dst[y * dst_pitch + x * lanes + c] = (unsigned char)(top + ty * (bottom - top) + 0.5f);
    }
}

// Packed 8-bit RGB source (any channel order, 3 or 4 bytes per pixel) onto an
// NV12 canvas in one pass: each thread produces one 2x2 luma block and its
// interleaved Cb/Cr pair (BT.709 limited range), sampling RGB bilinearly like
// scale_plane. dst_y/dst_uv are the canvas planes at their own pitches.
__device__ __forceinline__ void sample_rgb(
    const unsigned char *src, int src_pitch, int sx, int sy, int sw, int sh,
    int step, int r_off, int g_off, int b_off, float fx, float fy, float *rgb) {
    const int ix = int(floorf(fx)), iy = int(floorf(fy));
    const float tx = fx - ix, ty = fy - iy;
    const int x0 = sx + max(0, min(ix, sw - 1)), x1 = sx + max(0, min(ix + 1, sw - 1));
    const int y0 = sy + max(0, min(iy, sh - 1)), y1 = sy + max(0, min(iy + 1, sh - 1));
    const unsigned char *p00 = src + y0 * src_pitch + x0 * step, *p01 = src + y0 * src_pitch + x1 * step;
    const unsigned char *p10 = src + y1 * src_pitch + x0 * step, *p11 = src + y1 * src_pitch + x1 * step;
    const int offs[3] = {r_off, g_off, b_off};
    for (int c = 0; c < 3; ++c) {
        const float top = p00[offs[c]] + tx * (float(p01[offs[c]]) - p00[offs[c]]);
        const float bottom = p10[offs[c]] + tx * (float(p11[offs[c]]) - p10[offs[c]]);
        rgb[c] = top + ty * (bottom - top);
    }
}

extern "C" __global__ void rgb_to_nv12(
    const unsigned char *src, int src_pitch, int sx, int sy, int sw, int sh,
    int step, int r_off, int g_off, int b_off,
    unsigned char *dst_y, int y_pitch, unsigned char *dst_uv, int uv_pitch,
    int dx, int dy, int dw, int dh, int canvas_w, int canvas_h) {
    const int ox = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int oy = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    const int x = dx + ox, y = dy + oy;
    if (ox >= dw || oy >= dh || x < 0 || y < 0 || x >= canvas_w || y >= canvas_h) return;
    const float xs = float(sw) / dw, ys = float(sh) / dh;
    float sum[3] = {0.f, 0.f, 0.f};
    int n = 0;
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < 2; ++i) {
            if (ox + i >= dw || oy + j >= dh || x + i >= canvas_w || y + j >= canvas_h) continue;
            float rgb[3];
            sample_rgb(src, src_pitch, sx, sy, sw, sh, step, r_off, g_off, b_off,
                       (ox + i + 0.5f) * xs - 0.5f, (oy + j + 0.5f) * ys - 0.5f, rgb);
            const float luma = 16.f + 0.1826f * rgb[0] + 0.6142f * rgb[1] + 0.0620f * rgb[2];
            dst_y[(y + j) * y_pitch + x + i] = (unsigned char)min(max(luma + 0.5f, 0.f), 255.f);
            sum[0] += rgb[0]; sum[1] += rgb[1]; sum[2] += rgb[2];
            ++n;
        }
    }
    const float r = sum[0] / n, g = sum[1] / n, b = sum[2] / n;
    const float cb = 128.f - 0.1006f * r - 0.3386f * g + 0.4392f * b;
    const float cr = 128.f + 0.4392f * r - 0.3989f * g - 0.0403f * b;
    unsigned char *uv = dst_uv + (y / 2) * uv_pitch + (x / 2) * 2;
    uv[0] = (unsigned char)min(max(cb + 0.5f, 0.f), 255.f);
    uv[1] = (unsigned char)min(max(cr + 0.5f, 0.f), 255.f);
}
