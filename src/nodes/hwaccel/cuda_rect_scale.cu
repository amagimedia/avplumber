#include <cuda_runtime.h>

// Logical-sample access shared by every kernel here: a plane holds either
// bytes or little-endian 16-bit words whose meaningful bits sit above `shift`
// (P210/P010 store 10-bit codes as word >> 6; planar 10-bit uses shift 0).
__device__ __forceinline__ float load_sample(const unsigned char *p, int sample_bytes, int shift) {
    return sample_bytes == 2 ? float(*(const unsigned short *)p >> shift) : float(*p);
}

// Round-to-nearest store (+0.5 on a non-negative value); padding bits stay zero.
__device__ __forceinline__ void store_sample(unsigned char *p, int sample_bytes, int shift, float v) {
    if (sample_bytes == 2)
        *(unsigned short *)p = (unsigned short)((unsigned short)(v + 0.5f) << shift);
    else
        *p = (unsigned char)(v + 0.5f);
}

// Bilinear interpolation. This lightweight
// sampler does not widen its support when downscaling. Each lane is sampled
// separately, including interleaved UV. Coordinates count lane groups; pitches
// count bytes.
extern "C" __global__ void scale_plane(
    const unsigned char *src, int src_pitch, int sx, int sy, int sw, int sh,
    unsigned char *dst, int dst_pitch, int dx, int dy, int dw, int dh,
    int canvas_w, int canvas_h, int lanes, int sample_bytes, int shift) {
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
        const float a = load_sample(src + y0 * src_pitch + (x0 * lanes + c) * sample_bytes, sample_bytes, shift);
        const float b = load_sample(src + y0 * src_pitch + (x1 * lanes + c) * sample_bytes, sample_bytes, shift);
        const float d = load_sample(src + y1 * src_pitch + (x0 * lanes + c) * sample_bytes, sample_bytes, shift);
        const float e = load_sample(src + y1 * src_pitch + (x1 * lanes + c) * sample_bytes, sample_bytes, shift);
        const float top = a + tx * (b - a), bottom = d + tx * (e - d);
        store_sample(dst + y * dst_pitch + (x * lanes + c) * sample_bytes, sample_bytes, shift,
                     top + ty * (bottom - top));
    }
}

// Packed 8-bit RGB source (any channel order, 3 or 4 bytes per pixel) onto a
// semiplanar YUV canvas in one pass: each thread produces one chroma-footprint
// block of luma (2x2 for NV12, 2x1 for P210, 1x1 for 444) and its interleaved
// Cb/Cr pair (BT.709 limited range), sampling RGB bilinearly like scale_plane.
// The 8-bit matrix result is scaled by dst_scale on deeper canvases, so SDR
// graphics promote exactly like SDR video (16 -> 64); this path stays SDR.
// dst_y/dst_uv are the canvas planes at their own pitches.
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

extern "C" __global__ void rgb_to_yuv(
    const unsigned char *src, int src_pitch, int sx, int sy, int sw, int sh,
    int step, int r_off, int g_off, int b_off,
    unsigned char *dst_y, int y_pitch, unsigned char *dst_uv, int uv_pitch,
    int dx, int dy, int dw, int dh, int canvas_w, int canvas_h,
    int dst_sb, int dst_shift, int dst_scale, int sub_x, int sub_y) {
    const int bw = 1 << sub_x, bh = 1 << sub_y;
    const int ox = (blockIdx.x * blockDim.x + threadIdx.x) * bw;
    const int oy = (blockIdx.y * blockDim.y + threadIdx.y) * bh;
    const int x = dx + ox, y = dy + oy;
    if (ox >= dw || oy >= dh || x < 0 || y < 0 || x >= canvas_w || y >= canvas_h) return;
    const float xs = float(sw) / dw, ys = float(sh) / dh;
    const float maxv = 256.f * dst_scale - 1.f;
    float sum[3] = {0.f, 0.f, 0.f};
    int n = 0;
    for (int j = 0; j < bh; ++j) {
        for (int i = 0; i < bw; ++i) {
            if (ox + i >= dw || oy + j >= dh || x + i >= canvas_w || y + j >= canvas_h) continue;
            float rgb[3];
            sample_rgb(src, src_pitch, sx, sy, sw, sh, step, r_off, g_off, b_off,
                       (ox + i + 0.5f) * xs - 0.5f, (oy + j + 0.5f) * ys - 0.5f, rgb);
            const float luma = (16.f + 0.1826f * rgb[0] + 0.6142f * rgb[1] + 0.0620f * rgb[2]) * dst_scale;
            store_sample(dst_y + (y + j) * y_pitch + (x + i) * dst_sb, dst_sb, dst_shift,
                         min(max(luma, 0.f), maxv));
            sum[0] += rgb[0]; sum[1] += rgb[1]; sum[2] += rgb[2];
            ++n;
        }
    }
    const float r = sum[0] / n, g = sum[1] / n, b = sum[2] / n;
    const float cb = (128.f - 0.1006f * r - 0.3386f * g + 0.4392f * b) * dst_scale;
    const float cr = (128.f + 0.4392f * r - 0.3989f * g - 0.0403f * b) * dst_scale;
    unsigned char *uv = dst_uv + (y >> sub_y) * uv_pitch + (x >> sub_x) * 2 * dst_sb;
    store_sample(uv, dst_sb, dst_shift, min(max(cb, 0.f), maxv));
    store_sample(uv + dst_sb, dst_sb, dst_shift, min(max(cr, 0.f), maxv));
}

// Packed 8-bit RGBA over an existing semiplanar YUV canvas, BT.709 limited
// range: the source's own alpha decides how much of it survives, so a media
// wipe or a transparent graphic composites in one pass with no separate
// overlay filter, no format round trip and no CPU resize. Same chroma-block
// ownership as rgb_to_yuv, so chroma is read-modify-written exactly once per
// block.
extern "C" __global__ void rgba_over_yuv(
    const unsigned char *src, int src_pitch, int sx, int sy, int sw, int sh,
    int step, int r_off, int g_off, int b_off, int a_off,
    unsigned char *dst_y, int y_pitch, unsigned char *dst_uv, int uv_pitch,
    int dx, int dy, int dw, int dh, int canvas_w, int canvas_h,
    int dst_sb, int dst_shift, int dst_scale, int sub_x, int sub_y) {
    const int bw = 1 << sub_x, bh = 1 << sub_y;
    const int ox = (blockIdx.x * blockDim.x + threadIdx.x) * bw;
    const int oy = (blockIdx.y * blockDim.y + threadIdx.y) * bh;
    const int x = dx + ox, y = dy + oy;
    if (ox >= dw || oy >= dh || x < 0 || y < 0 || x >= canvas_w || y >= canvas_h) return;
    const float xs = float(sw) / dw, ys = float(sh) / dh;
    const float maxv = 256.f * dst_scale - 1.f;
    float sum[3] = {0.f, 0.f, 0.f}, sum_a = 0.f;
    int n = 0;
    for (int j = 0; j < bh; ++j) {
        for (int i = 0; i < bw; ++i) {
            if (ox + i >= dw || oy + j >= dh || x + i >= canvas_w || y + j >= canvas_h) continue;
            float rgb[3];
            const float fx = (ox + i + 0.5f) * xs - 0.5f, fy = (oy + j + 0.5f) * ys - 0.5f;
            sample_rgb(src, src_pitch, sx, sy, sw, sh, step, r_off, g_off, b_off, fx, fy, rgb);
            // Alpha is sampled on the same bilinear footprint as the colour.
            const int ix = int(floorf(fx)), iy = int(floorf(fy));
            const float tx = fx - ix, ty = fy - iy;
            const int x0 = sx + max(0, min(ix, sw - 1)), x1 = sx + max(0, min(ix + 1, sw - 1));
            const int y0 = sy + max(0, min(iy, sh - 1)), y1 = sy + max(0, min(iy + 1, sh - 1));
            const float a00 = src[y0 * src_pitch + x0 * step + a_off], a01 = src[y0 * src_pitch + x1 * step + a_off];
            const float a10 = src[y1 * src_pitch + x0 * step + a_off], a11 = src[y1 * src_pitch + x1 * step + a_off];
            const float a = ((a00 + tx * (a01 - a00)) + ty * ((a10 + tx * (a11 - a10)) - (a00 + tx * (a01 - a00)))) / 255.f;

            const float luma = (16.f + 0.1826f * rgb[0] + 0.6142f * rgb[1] + 0.0620f * rgb[2]) * dst_scale;
            unsigned char *py = dst_y + (y + j) * y_pitch + (x + i) * dst_sb;
            store_sample(py, dst_sb, dst_shift,
                         min(max(a * luma + (1.f - a) * load_sample(py, dst_sb, dst_shift), 0.f), maxv));
            sum[0] += a * rgb[0]; sum[1] += a * rgb[1]; sum[2] += a * rgb[2];
            sum_a += a;
            ++n;
        }
    }
    if (n == 0 || sum_a <= 0.f) return;
    // Premultiplied average keeps a transparent corner of the block from
    // dragging the blended chroma toward black.
    const float r = sum[0] / sum_a, g = sum[1] / sum_a, b = sum[2] / sum_a, a = sum_a / n;
    const float cb = (128.f - 0.1006f * r - 0.3386f * g + 0.4392f * b) * dst_scale;
    const float cr = (128.f + 0.4392f * r - 0.3989f * g - 0.0403f * b) * dst_scale;
    unsigned char *uv = dst_uv + (y >> sub_y) * uv_pitch + (x >> sub_x) * 2 * dst_sb;
    store_sample(uv, dst_sb, dst_shift,
                 min(max(a * cb + (1.f - a) * load_sample(uv, dst_sb, dst_shift), 0.f), maxv));
    store_sample(uv + dst_sb, dst_sb, dst_shift,
                 min(max(a * cr + (1.f - a) * load_sample(uv + dst_sb, dst_sb, dst_shift), 0.f), maxv));
}
