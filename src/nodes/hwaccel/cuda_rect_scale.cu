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

// Cross-depth plane scaler: read an 8-bit (or narrower) source plane, bilinear
// scale it into the destination region and store at the canvas depth, scaling
// logical codes by `mul` (4 for 8->10-bit SDR promotion, 16->64 / 235->940).
// Separate src/dst sample bytes and shifts let one body cover NV12->P210 luma
// and interleaved chroma; the chroma footprint difference (4:2:0 -> 4:2:2) is
// just the src/dst region sizes, so the same bilinear handles the resample.
extern "C" __global__ void convert_scale_plane(
    const unsigned char *src, int src_pitch, int sx, int sy, int sw, int sh,
    int src_bytes, int src_shift,
    unsigned char *dst, int dst_pitch, int dx, int dy, int dw, int dh,
    int canvas_w, int canvas_h, int lanes, int dst_bytes, int dst_shift, float mul) {
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
        const float a = load_sample(src + y0 * src_pitch + (x0 * lanes + c) * src_bytes, src_bytes, src_shift);
        const float b = load_sample(src + y0 * src_pitch + (x1 * lanes + c) * src_bytes, src_bytes, src_shift);
        const float d = load_sample(src + y1 * src_pitch + (x0 * lanes + c) * src_bytes, src_bytes, src_shift);
        const float e = load_sample(src + y1 * src_pitch + (x1 * lanes + c) * src_bytes, src_bytes, src_shift);
        const float top = a + tx * (b - a), bottom = d + tx * (e - d);
        store_sample(dst + y * dst_pitch + (x * lanes + c) * dst_bytes, dst_bytes, dst_shift,
                     (top + ty * (bottom - top)) * mul);
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

// Embed SDR RGB graphics at the same display white as tonemap_cuda. Transfer
// codes match that filter: HLG=0, PQ=1, SDR=2. Alpha stays separate.
__device__ static inline float pq_code(float nits) {
    const float p = powf(fmaxf(nits, 0.f) / 10000.f, 0.1593017578125f);
    return powf((0.8359375f + 18.8515625f * p) / (1.f + 18.6875f * p), 78.84375f);
}

__device__ static inline void convert_graphic_rgb(float *rgb, int transfer, float white, float peak) {
    if (transfer == 2) return;
    float r = powf(fmaxf(rgb[0] / 255.f, 0.f), 2.4f) * white;
    float g = powf(fmaxf(rgb[1] / 255.f, 0.f), 2.4f) * white;
    float b = powf(fmaxf(rgb[2] / 255.f, 0.f), 2.4f) * white;
    rgb[0] = 0.627404f * r + 0.329283f * g + 0.043313f * b;
    rgb[1] = 0.069097f * r + 0.919540f * g + 0.011362f * b;
    rgb[2] = 0.016391f * r + 0.088013f * g + 0.895595f * b;
    if (transfer == 1) {
        for (int i = 0; i < 3; ++i) rgb[i] = 255.f * pq_code(rgb[i]);
        return;
    }
    const float luma = 0.2627f * rgb[0] + 0.6780f * rgb[1] + 0.0593f * rgb[2];
    const float gamma = fmaxf(1.f, 1.2f + 0.42f * log10f(peak / 1000.f));
    const float gain = luma > 0.f ? 12.f * powf(luma / peak, 1.f / gamma) / luma : 0.f;
    for (int i = 0; i < 3; ++i) {
        const float c = fmaxf(rgb[i] * gain, 0.f);
        rgb[i] = 255.f * (c <= 1.f ? 0.5f * sqrtf(c) : 0.17883277f * logf(c - 0.28466892f) + 0.55991073f);
    }
}

__device__ static inline float graphic_luma(const float *rgb, int transfer) {
    return transfer == 2 ? 16.f + 0.1826f * rgb[0] + 0.6142f * rgb[1] + 0.0620f * rgb[2]
        : 16.f + (219.f / 255.f) * (0.2627f * rgb[0] + 0.6780f * rgb[1] + 0.0593f * rgb[2]);
}

__device__ static inline void graphic_chroma(float r, float g, float b, int transfer, float &cb, float &cr) {
    if (transfer == 2) {
        cb = 128.f - 0.1006f * r - 0.3386f * g + 0.4392f * b;
        cr = 128.f + 0.4392f * r - 0.3989f * g - 0.0403f * b;
    } else {
        const float y = 0.2627f * r + 0.6780f * g + 0.0593f * b;
        cb = 128.f + (224.f / 255.f) * (b - y) / 1.8814f;
        cr = 128.f + (224.f / 255.f) * (r - y) / 1.4746f;
    }
}

extern "C" __global__ void rgb_to_yuv(
    const unsigned char *src, int src_pitch, int sx, int sy, int sw, int sh,
    int step, int r_off, int g_off, int b_off,
    unsigned char *dst_y, int y_pitch, unsigned char *dst_uv, int uv_pitch,
    int dx, int dy, int dw, int dh, int canvas_w, int canvas_h,
    int dst_sb, int dst_shift, int dst_scale, int sub_x, int sub_y,
    int transfer = 2, float white = 203.f, float peak = 1000.f) {
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
            convert_graphic_rgb(rgb, transfer, white, peak);
            const float luma = graphic_luma(rgb, transfer) * dst_scale;
            store_sample(dst_y + (y + j) * y_pitch + (x + i) * dst_sb, dst_sb, dst_shift,
                         min(max(luma, 0.f), maxv));
            sum[0] += rgb[0]; sum[1] += rgb[1]; sum[2] += rgb[2];
            ++n;
        }
    }
    const float r = sum[0] / n, g = sum[1] / n, b = sum[2] / n;
    float cb, cr;
    graphic_chroma(r, g, b, transfer, cb, cr);
    cb *= dst_scale; cr *= dst_scale;
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
    int dst_sb, int dst_shift, int dst_scale, int sub_x, int sub_y,
    int transfer = 2, float white = 203.f, float peak = 1000.f, int premultiplied = 0) {
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

            // Unassociate before the nonlinear transfer conversion. Interpolating
            // the premultiplied samples first also preserves transparent edges.
            if (premultiplied)
                for (int c = 0; c < 3; ++c)
                    rgb[c] = a > 0.f ? min(rgb[c] / a, 255.f) : 0.f;
            convert_graphic_rgb(rgb, transfer, white, peak);
            const float luma = graphic_luma(rgb, transfer) * dst_scale;
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
    float cb, cr;
    graphic_chroma(r, g, b, transfer, cb, cr);
    cb *= dst_scale; cr *= dst_scale;
    unsigned char *uv = dst_uv + (y >> sub_y) * uv_pitch + (x >> sub_x) * 2 * dst_sb;
    store_sample(uv, dst_sb, dst_shift,
                 min(max(a * cb + (1.f - a) * load_sample(uv, dst_sb, dst_shift), 0.f), maxv));
    store_sample(uv + dst_sb, dst_sb, dst_shift,
                 min(max(a * cr + (1.f - a) * load_sample(uv + dst_sb, dst_sb, dst_shift), 0.f), maxv));
}

// ---------------------------------------------------------------------------
// Single-launch composite. One launch draws the whole canvas: gridDim.z selects
// the plane (0 luma, 1 interleaved chroma), each thread owns one lane group of
// that plane, walks the draw-ordered rect table bottom to top and produces the
// same value the per-layer kernels above would have left there: bilinear scale
// or blit for canvas-format sources, code promotion for lower-depth sources,
// fused RGB(A) conversion with the same chroma-block averaging, alpha blending
// against what lower layers produced, and the clear value where nothing draws.
// Every intermediate value is rounded exactly as the per-layer store would have
// been, so the output is bit-identical to clear + N layer launches.
// ---------------------------------------------------------------------------
#include "cuda_rect_table.h"

// Emulate store_sample followed by load_sample: the rounded code as a float.
__device__ __forceinline__ float stored_code(float v) { return truncf(v + 0.5f); }

__device__ __forceinline__ bool rect_overlaps(int ax, int ay, int aw, int ah, int bx, int by, int bw, int bh) {
    return ax < bx + bw && bx < ax + aw && ay < by + bh && by < ay + ah;
}

// Bilinear sample of lane `c` of a lane-group plane, identical to scale_plane / convert_scale_plane.
// The vertical terms depend only on the row, so the caller computes them once per thread.
struct RowTaps { int y0, y1; float ty; };
__device__ __forceinline__ RowTaps row_taps(int sy, int sh, int oy, int dh) {
    const float fy = (oy + 0.5f) * sh / dh - 0.5f;
    const int iy = int(floorf(fy));
    RowTaps t;
    t.ty = fy - iy;
    t.y0 = sy + max(0, min(iy, sh - 1));
    t.y1 = sy + max(0, min(iy + 1, sh - 1));
    return t;
}
__device__ __forceinline__ float sample_lane(const unsigned char *src, int src_pitch, int sx, int sw,
                                             int lanes, int c, int bytes, int shift, int ox, int dw, RowTaps t) {
    const float fx = (ox + 0.5f) * sw / dw - 0.5f;
    const int ix = int(floorf(fx));
    const float tx = fx - ix;
    const int x0 = sx + max(0, min(ix, sw - 1));
    const int x1 = sx + max(0, min(ix + 1, sw - 1));
    const float a = load_sample(src + t.y0 * src_pitch + (x0 * lanes + c) * bytes, bytes, shift);
    const float b = load_sample(src + t.y0 * src_pitch + (x1 * lanes + c) * bytes, bytes, shift);
    const float d = load_sample(src + t.y1 * src_pitch + (x0 * lanes + c) * bytes, bytes, shift);
    const float e = load_sample(src + t.y1 * src_pitch + (x1 * lanes + c) * bytes, bytes, shift);
    const float top = a + tx * (b - a), bottom = d + tx * (e - d);
    return top + t.ty * (bottom - top);
}

// Bilinear alpha at the same footprint sample_rgb uses, as rgba_over_yuv computes it.
__device__ __forceinline__ float sample_alpha(const unsigned char *src, int src_pitch, int sx, int sy, int sw, int sh,
                                              int step, int a_off, float fx, float fy) {
    const int ix = int(floorf(fx)), iy = int(floorf(fy));
    const float tx = fx - ix, ty = fy - iy;
    const int x0 = sx + max(0, min(ix, sw - 1)), x1 = sx + max(0, min(ix + 1, sw - 1));
    const int y0 = sy + max(0, min(iy, sh - 1)), y1 = sy + max(0, min(iy + 1, sh - 1));
    const float a00 = src[y0 * src_pitch + x0 * step + a_off], a01 = src[y0 * src_pitch + x1 * step + a_off];
    const float a10 = src[y1 * src_pitch + x0 * step + a_off], a11 = src[y1 * src_pitch + x1 * step + a_off];
    return ((a00 + tx * (a01 - a00)) + ty * ((a10 + tx * (a11 - a10)) - (a00 + tx * (a01 - a00)))) / 255.f;
}

// kRgb=false compiles a lean YUV-only body (no transfer math, far fewer registers) for the
// common case of scenes without packed-RGB layers; the host picks the entry point per frame.
//
// Each thread owns AVP_RECT_PX consecutive lane groups of one row (4 luma samples, or 2 chroma
// pairs): layer geometry is fetched once per thread, the per-sample math is exactly the
// per-layer kernels', and the results leave in one vector store.
template <bool kRgb>
__device__ __forceinline__ void composite_body(
    const AvpRectLayer *__restrict__ layers, int n,
    unsigned char *dst_y, int y_pitch, unsigned char *dst_uv, int uv_pitch,
    int canvas_w, int canvas_h, int chroma_w, int chroma_h,
    int dst_sb, int dst_shift, int dst_scale, int sub_x, int sub_y,
    int clear_y, int clear_uv, int transfer, float white, float peak) {
    const int plane = blockIdx.z;
    const int px = plane ? AVP_RECT_PX / 2 : AVP_RECT_PX;          // lane groups per thread
    const int pw = plane ? chroma_w : canvas_w, ph = plane ? chroma_h : canvas_h;
    const int tile_w = blockDim.x * px;
    const int tile_x = blockIdx.x * tile_w, tile_y = blockIdx.y * blockDim.y;
    if (tile_x >= pw || tile_y >= ph) return;   // block-uniform: no thread reaches the barrier below

    // Per-block culling: one bit per layer whose rect touches this tile.
    __shared__ unsigned int hit[AVP_RECT_MAX_LAYERS / 32];
    const int threads = blockDim.x * blockDim.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int w = tid; w < AVP_RECT_MAX_LAYERS / 32; w += threads) hit[w] = 0;
    __syncthreads();
    for (int i = tid; i < n; i += threads) {
        const AvpRectLayer &L = layers[i];
        if (rect_overlaps(tile_x, tile_y, tile_w, blockDim.y, L.dx[plane], L.dy[plane], L.dw[plane], L.dh[plane]))
            atomicOr(&hit[i >> 5], 1u << (i & 31));
    }
    __syncthreads();

    const int X0 = tile_x + threadIdx.x * px, Y = tile_y + threadIdx.y;
    if (X0 >= pw || Y >= ph) return;
    const int lanes = plane ? 2 : 1;
    const int bw = 1 << sub_x, bh = 1 << sub_y;
    const float maxv = 256.f * dst_scale - 1.f;
    float v0[AVP_RECT_PX], v1[AVP_RECT_PX];
#pragma unroll
    for (int p = 0; p < AVP_RECT_PX; ++p) { v0[p] = plane ? float(clear_uv) : float(clear_y); v1[p] = float(clear_uv); }

    for (int w = 0; w < AVP_RECT_MAX_LAYERS / 32; ++w) {
        unsigned int bits = hit[w];
        while (bits) {
            const int bit = __ffs(bits) - 1;
            bits &= bits - 1;
            const AvpRectLayer &L = layers[w * 32 + bit];
            const int ldx = L.dx[plane], ldy = L.dy[plane], ldw = L.dw[plane], ldh = L.dh[plane];
            const int oy = Y - ldy;
            if (oy < 0 || oy >= ldh || X0 + px <= ldx || X0 >= ldx + ldw) continue;

            if (L.kind == AVP_RECT_KIND_YUV || L.kind == AVP_RECT_KIND_PROMOTE) {
                const unsigned char *src = (const unsigned char *)L.src[plane];
                const int pitch = L.src_pitch[plane];
                const int sx = L.sx[plane], sy = L.sy[plane], sw = L.sw[plane], sh = L.sh[plane];
                const int bytes = L.kind == AVP_RECT_KIND_YUV ? dst_sb : L.src_bytes;
                const int shift = L.kind == AVP_RECT_KIND_YUV ? dst_shift : L.src_shift;
                const float mul = L.mul;
                const RowTaps taps = row_taps(sy, sh, oy, ldh);
#pragma unroll
                for (int p = 0; p < AVP_RECT_PX; ++p) {
                    if (p >= px) break;
                    const int ox = X0 + p - ldx;
                    if (ox < 0 || ox >= ldw) continue;
                    for (int c = 0; c < lanes; ++c) {
                        float sv = sample_lane(src, pitch, sx, sw, lanes, c, bytes, shift, ox, ldw, taps);
                        if (L.kind == AVP_RECT_KIND_PROMOTE) sv *= mul;
                        if (c == 0) v0[p] = stored_code(sv); else v1[p] = stored_code(sv);
                    }
                }
                continue;
            }

            if (!kRgb) continue;   // host never puts RGB entries in a YUV-only table
            // Packed RGB(A): plane 0 is one luma sample; plane 1 is one chroma block of bw x bh luma
            // positions averaged, with the same skip rules as rgb_to_yuv / rgba_over_yuv.
            const bool blend = L.kind == AVP_RECT_KIND_RGBA;
            const unsigned char *src = (const unsigned char *)L.src[0];   // packed RGB: one plane for both passes
            const int pitch = L.src_pitch[0];
            const int sx = L.sx[0], sy = L.sy[0], sw = L.sw[0], sh = L.sh[0];
            const int dx = L.dx[0], dy = L.dy[0], dw = L.dw[0], dh = L.dh[0];
            const float xs = float(sw) / dw, ys = float(sh) / dh;
            const int step = L.step, r_off = L.r_off, g_off = L.g_off, b_off = L.b_off, a_off = L.a_off;
            const int premultiplied = L.premultiplied;
#pragma unroll 1
            for (int p = 0; p < px; ++p) {
                const int X = X0 + p;
                const int ox = X - ldx;
                if (ox < 0 || ox >= ldw) continue;
                if (plane == 0) {
                    float rgb[3];
                    const float fx = (ox + 0.5f) * xs - 0.5f, fy = (oy + 0.5f) * ys - 0.5f;
                    sample_rgb(src, pitch, sx, sy, sw, sh, step, r_off, g_off, b_off, fx, fy, rgb);
                    if (!blend) {
                        convert_graphic_rgb(rgb, transfer, white, peak);
                        const float luma = graphic_luma(rgb, transfer) * dst_scale;
                        v0[p] = stored_code(min(max(luma, 0.f), maxv));
                    } else {
                        const float a = sample_alpha(src, pitch, sx, sy, sw, sh, step, a_off, fx, fy);
                        if (premultiplied)
                            for (int c = 0; c < 3; ++c)
                                rgb[c] = a > 0.f ? min(rgb[c] / a, 255.f) : 0.f;
                        convert_graphic_rgb(rgb, transfer, white, peak);
                        const float luma = graphic_luma(rgb, transfer) * dst_scale;
                        v0[p] = stored_code(min(max(a * luma + (1.f - a) * v0[p], 0.f), maxv));
                    }
                    continue;
                }
                // Chroma: this thread's block origin in luma coordinates, relative to the layer.
                const int box = (X << sub_x) - dx, boy = (Y << sub_y) - dy;
                const int x = dx + box, y = dy + boy;
                float sum[3] = {0.f, 0.f, 0.f}, sum_a = 0.f;
                int cnt = 0;
                for (int j = 0; j < bh; ++j) {
                    for (int i2 = 0; i2 < bw; ++i2) {
                        if (box + i2 >= dw || boy + j >= dh || x + i2 >= canvas_w || y + j >= canvas_h) continue;
                        float rgb[3];
                        const float fx = (box + i2 + 0.5f) * xs - 0.5f, fy = (boy + j + 0.5f) * ys - 0.5f;
                        sample_rgb(src, pitch, sx, sy, sw, sh, step, r_off, g_off, b_off, fx, fy, rgb);
                        if (!blend) {
                            convert_graphic_rgb(rgb, transfer, white, peak);
                            sum[0] += rgb[0]; sum[1] += rgb[1]; sum[2] += rgb[2];
                        } else {
                            const float a = sample_alpha(src, pitch, sx, sy, sw, sh, step, a_off, fx, fy);
                            if (premultiplied)
                                for (int c = 0; c < 3; ++c)
                                    rgb[c] = a > 0.f ? min(rgb[c] / a, 255.f) : 0.f;
                            convert_graphic_rgb(rgb, transfer, white, peak);
                            sum[0] += a * rgb[0]; sum[1] += a * rgb[1]; sum[2] += a * rgb[2];
                            sum_a += a;
                        }
                        ++cnt;
                    }
                }
                float cb, cr;
                if (!blend) {
                    const float r = sum[0] / cnt, g = sum[1] / cnt, b = sum[2] / cnt;
                    graphic_chroma(r, g, b, transfer, cb, cr);
                    cb *= dst_scale; cr *= dst_scale;
                    v0[p] = stored_code(min(max(cb, 0.f), maxv));
                    v1[p] = stored_code(min(max(cr, 0.f), maxv));
                } else {
                    if (cnt == 0 || sum_a <= 0.f) continue;
                    const float r = sum[0] / sum_a, g = sum[1] / sum_a, b = sum[2] / sum_a, a = sum_a / cnt;
                    graphic_chroma(r, g, b, transfer, cb, cr);
                    cb *= dst_scale; cr *= dst_scale;
                    v0[p] = stored_code(min(max(a * cb + (1.f - a) * v0[p], 0.f), maxv));
                    v1[p] = stored_code(min(max(a * cr + (1.f - a) * v1[p], 0.f), maxv));
                }
            }
        }
    }

    // Store: 4 consecutive samples (4 luma, or 2 chroma pairs) in one aligned vector write when the
    // group lies inside the plane; scalar stores for a tail group. Canvas pitches are 256-aligned.
    unsigned char *row = plane ? dst_uv + Y * uv_pitch : dst_y + Y * y_pitch;
    const int valid = min(px, pw - X0);
    if (valid == px) {
        if (dst_sb == 1) {
            uchar4 out;
            if (plane == 0)
                out = make_uchar4((unsigned char)(v0[0] + 0.5f), (unsigned char)(v0[1] + 0.5f),
                                  (unsigned char)(v0[2] + 0.5f), (unsigned char)(v0[3] + 0.5f));
            else
                out = make_uchar4((unsigned char)(v0[0] + 0.5f), (unsigned char)(v1[0] + 0.5f),
                                  (unsigned char)(v0[1] + 0.5f), (unsigned char)(v1[1] + 0.5f));
            *(uchar4 *)(row + X0 * lanes) = out;
        } else {
            ushort4 out;
            if (plane == 0)
                out = make_ushort4((unsigned short)((unsigned short)(v0[0] + 0.5f) << dst_shift),
                                   (unsigned short)((unsigned short)(v0[1] + 0.5f) << dst_shift),
                                   (unsigned short)((unsigned short)(v0[2] + 0.5f) << dst_shift),
                                   (unsigned short)((unsigned short)(v0[3] + 0.5f) << dst_shift));
            else
                out = make_ushort4((unsigned short)((unsigned short)(v0[0] + 0.5f) << dst_shift),
                                   (unsigned short)((unsigned short)(v1[0] + 0.5f) << dst_shift),
                                   (unsigned short)((unsigned short)(v0[1] + 0.5f) << dst_shift),
                                   (unsigned short)((unsigned short)(v1[1] + 0.5f) << dst_shift));
            *(ushort4 *)(row + X0 * lanes * 2) = out;
        }
        return;
    }
    for (int p = 0; p < valid; ++p) {
        unsigned char *at = row + (X0 + p) * lanes * dst_sb;
        store_sample(at, dst_sb, dst_shift, v0[p]);
        if (plane) store_sample(at + dst_sb, dst_sb, dst_shift, v1[p]);
    }
}

#define AVP_COMPOSITE_ARGS \
    const AvpRectLayer *__restrict__ layers, int n, \
    unsigned char *dst_y, int y_pitch, unsigned char *dst_uv, int uv_pitch, \
    int canvas_w, int canvas_h, int chroma_w, int chroma_h, \
    int dst_sb, int dst_shift, int dst_scale, int sub_x, int sub_y, \
    int clear_y, int clear_uv, int transfer, float white, float peak
#define AVP_COMPOSITE_PASS \
    layers, n, dst_y, y_pitch, dst_uv, uv_pitch, canvas_w, canvas_h, chroma_w, chroma_h, \
    dst_sb, dst_shift, dst_scale, sub_x, sub_y, clear_y, clear_uv, transfer, white, peak

extern "C" __global__ void __launch_bounds__(256) composite_planes(AVP_COMPOSITE_ARGS) {
    composite_body<true>(AVP_COMPOSITE_PASS);
}
extern "C" __global__ void __launch_bounds__(256) composite_planes_yuv(AVP_COMPOSITE_ARGS) {
    composite_body<false>(AVP_COMPOSITE_PASS);
}
