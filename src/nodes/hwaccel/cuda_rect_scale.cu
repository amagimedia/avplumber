// CUDA compositor kernel: one launch composes a whole canvas from a rect table
// (see cuda_rect_table.h and CudaRectDraw). Semiplanar YUV canvases (NV12,
// P010, P210) take same-format YUV, lower-depth semiplanar and packed RGB(A)
// sources; packed 8-bit canvases (rgb0, rgba) take same-format sources.
#include <cuda_runtime.h>
#include "cuda_rect_table.h"
#include "graphic_color.h"

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

// Bilinear RGB (8-bit codes) of a packed source; the caller converts to the
// canvas transfer and depth.
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

// ---------------------------------------------------------------------------
// Single-launch composite. One launch draws the whole canvas: gridDim.z selects
// the plane (0 luma or packed pixels, 1 interleaved chroma of a semiplanar
// canvas), each thread owns AVP_RECT_PX lane groups of one row, walks the
// draw-ordered rect table bottom to top and produces: bilinear scale or blit for
// canvas-format sources, code promotion for lower-depth semiplanar sources,
// fused RGB(A) conversion onto YUV with chroma-block averaging, alpha blending
// against what lower layers produced, and the clear value where nothing draws.
// Every intermediate value is rounded as a stored sample would be, so the result
// equals clear + one launch per layer (tests/cuda/legacy_rect_kernels.cuh).
// ---------------------------------------------------------------------------

// Emulate store_sample followed by load_sample: the rounded code as a float.
__device__ __forceinline__ float stored_code(float v) { return truncf(v + 0.5f); }

__device__ __forceinline__ bool rect_overlaps(int ax, int ay, int aw, int ah, int bx, int by, int bw, int bh) {
    return ax < bx + bw && bx < ax + aw && ay < by + bh && by < ay + ah;
}

// Bilinear sample of lane `c` of a lane-group plane (canvas-format and promoted sources).
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
__device__ __forceinline__ float load_lane(unsigned long long storage, bool textured, int pitch,
                                           int x, int y, int lanes, int c, int bytes, int shift) {
    if (!textured)
        return load_sample((const unsigned char *)storage + y * pitch + (x * lanes + c) * bytes, bytes, shift);
    // Point sampling returns exact stored codes. Keep the same software bilinear
    // arithmetic as linear inputs; CUDA's hardware interpolation has lower precision.
    if (bytes == 1) {
        if (lanes == 1) return float(tex2D<unsigned char>(storage, x + 0.5f, y + 0.5f));
        const uchar2 pair = tex2D<uchar2>(storage, x + 0.5f, y + 0.5f);
        return float(c ? pair.y : pair.x);
    }
    if (lanes == 1) return float(tex2D<unsigned short>(storage, x + 0.5f, y + 0.5f) >> shift);
    const ushort2 pair = tex2D<ushort2>(storage, x + 0.5f, y + 0.5f);
    return float((c ? pair.y : pair.x) >> shift);
}

__device__ __forceinline__ float sample_lane(unsigned long long src, bool textured, int src_pitch, int sx, int sw,
                                             int lanes, int c, int bytes, int shift, int ox, int dw, RowTaps t) {
    const float fx = (ox + 0.5f) * sw / dw - 0.5f;
    const int ix = int(floorf(fx));
    const float tx = fx - ix;
    const int x0 = sx + max(0, min(ix, sw - 1));
    const int x1 = sx + max(0, min(ix + 1, sw - 1));
    const float a = load_lane(src, textured, src_pitch, x0, t.y0, lanes, c, bytes, shift);
    const float b = load_lane(src, textured, src_pitch, x1, t.y0, lanes, c, bytes, shift);
    const float d = load_lane(src, textured, src_pitch, x0, t.y1, lanes, c, bytes, shift);
    const float e = load_lane(src, textured, src_pitch, x1, t.y1, lanes, c, bytes, shift);
    const float top = a + tx * (b - a), bottom = d + tx * (e - d);
    return top + t.ty * (bottom - top);
}

// Bilinear alpha at the footprint sample_rgb uses.
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

// Texture-backed packed RGBA (zero-copy DMA-BUF): the same four taps as sample_rgb /
// sample_alpha, fetched as exact texels (point sampling, unnormalized coordinates), so the
// result matches the pointer path bit for bit.
__device__ __forceinline__ float texel_channel(uchar4 v, int off) {
    return off == 0 ? float(v.x) : off == 1 ? float(v.y) : off == 2 ? float(v.z) : float(v.w);
}
__device__ __forceinline__ void sample_rgb_tex(unsigned long long tex, int sx, int sy, int sw, int sh,
                                               int r_off, int g_off, int b_off, float fx, float fy, float *rgb) {
    const int ix = int(floorf(fx)), iy = int(floorf(fy));
    const float tx = fx - ix, ty = fy - iy;
    const int x0 = sx + max(0, min(ix, sw - 1)), x1 = sx + max(0, min(ix + 1, sw - 1));
    const int y0 = sy + max(0, min(iy, sh - 1)), y1 = sy + max(0, min(iy + 1, sh - 1));
    const uchar4 p00 = tex2D<uchar4>(tex, x0 + 0.5f, y0 + 0.5f), p01 = tex2D<uchar4>(tex, x1 + 0.5f, y0 + 0.5f);
    const uchar4 p10 = tex2D<uchar4>(tex, x0 + 0.5f, y1 + 0.5f), p11 = tex2D<uchar4>(tex, x1 + 0.5f, y1 + 0.5f);
    const int offs[3] = {r_off, g_off, b_off};
    for (int c = 0; c < 3; ++c) {
        const float a = texel_channel(p00, offs[c]), b = texel_channel(p01, offs[c]);
        const float d = texel_channel(p10, offs[c]), e = texel_channel(p11, offs[c]);
        const float top = a + tx * (b - a);
        const float bottom = d + tx * (e - d);
        rgb[c] = top + ty * (bottom - top);
    }
}
__device__ __forceinline__ float sample_alpha_tex(unsigned long long tex, int sx, int sy, int sw, int sh,
                                                  int a_off, float fx, float fy) {
    const int ix = int(floorf(fx)), iy = int(floorf(fy));
    const float tx = fx - ix, ty = fy - iy;
    const int x0 = sx + max(0, min(ix, sw - 1)), x1 = sx + max(0, min(ix + 1, sw - 1));
    const int y0 = sy + max(0, min(iy, sh - 1)), y1 = sy + max(0, min(iy + 1, sh - 1));
    const float a00 = texel_channel(tex2D<uchar4>(tex, x0 + 0.5f, y0 + 0.5f), a_off);
    const float a01 = texel_channel(tex2D<uchar4>(tex, x1 + 0.5f, y0 + 0.5f), a_off);
    const float a10 = texel_channel(tex2D<uchar4>(tex, x0 + 0.5f, y1 + 0.5f), a_off);
    const float a11 = texel_channel(tex2D<uchar4>(tex, x1 + 0.5f, y1 + 0.5f), a_off);
    return ((a00 + tx * (a01 - a00)) + ty * ((a10 + tx * (a11 - a10)) - (a00 + tx * (a01 - a00)))) / 255.f;
}

// Each thread owns AVP_RECT_PX consecutive lane groups of one row (4 luma samples, or 2 chroma
// pairs): layer geometry is fetched once per thread and the results leave in one vector store.
// kRgb=false compiles a lean YUV-only body (no transfer math, far fewer registers) for scenes
// without packed-RGB layers; the host picks the entry point per frame. kLanes and kChroma are
// compile-time so the per-sample loops unroll and the accumulators stay in registers:
// <1,false> luma, <2,true> chroma pairs, <4,false> packed RGB canvas. kOpacity=true also
// weights each blended RGBA layer's alpha by its table `mul` (a key fade); only the
// composite_planes_opacity entry sets it, so the other entries keep their code.
template <bool kRgb, int kLanes, bool kChroma, bool kOpacity = false, bool kArrays = false>
__device__ __forceinline__ void composite_body(
    const AvpRectLayer *__restrict__ layers, int n,
    unsigned char *dst_y, int y_pitch, unsigned char *dst_uv, int uv_pitch,
    int canvas_w, int canvas_h, int chroma_w, int chroma_h,
    int dst_sb, int dst_shift, int dst_scale, int sub_x, int sub_y,
    int clear_y, int clear_uv, int transfer, float white, float peak) {
    constexpr int plane = kChroma ? 1 : 0;
    constexpr int px = kChroma ? AVP_RECT_PX / 2 : AVP_RECT_PX;   // lane groups per thread
    constexpr int lanes = kLanes;
    const int pw = plane ? chroma_w : canvas_w, ph = plane ? chroma_h : canvas_h;
    const int tile_w = blockDim.x * px;
    const int tile_x = blockIdx.x * tile_w, tile_y = blockIdx.y * blockDim.y;
    if (tile_x >= pw || tile_y >= ph) return;   // block-uniform: no thread reaches the barrier below

    // Per-block culling: one bit per layer whose rect touches this tile.
    extern __shared__ unsigned int hit[];
    const int threads = blockDim.x * blockDim.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int words = (n + 31) / 32;
    for (int w = tid; w < words; w += threads) hit[w] = 0;
    __syncthreads();
    for (int i = tid; i < n; i += threads) {
        const AvpRectLayer &L = layers[i];
        if (rect_overlaps(tile_x, tile_y, tile_w, blockDim.y, L.dx[plane], L.dy[plane], L.dw[plane], L.dh[plane]))
            atomicOr(&hit[i >> 5], 1u << (i & 31));
    }
    __syncthreads();

    const int X0 = tile_x + threadIdx.x * px, Y = tile_y + threadIdx.y;
    if (X0 >= pw || Y >= ph) return;
    const int bw = 1 << sub_x, bh = 1 << sub_y;
    const float maxv = 256.f * dst_scale - 1.f;
    // acc[p][0] is luma or lane 0; acc[p][1..] the other lanes of the group.
    float acc[px][lanes];
#pragma unroll
    for (int p = 0; p < px; ++p)
#pragma unroll
        for (int c = 0; c < lanes; ++c) acc[p][c] = kChroma ? float(clear_uv) : float(clear_y);

    for (int w = 0; w < words; ++w) {
        unsigned int bits = hit[w];
        while (bits) {
            const int bit = __ffs(bits) - 1;
            bits &= bits - 1;
            const AvpRectLayer &L = layers[w * 32 + bit];
            const int ldx = L.dx[plane], ldy = L.dy[plane], ldw = L.dw[plane], ldh = L.dh[plane];
            const int oy = Y - ldy;
            if (oy < 0 || oy >= ldh || X0 + px <= ldx || X0 >= ldx + ldw) continue;

            if (L.kind == AVP_RECT_KIND_YUV || L.kind == AVP_RECT_KIND_PROMOTE) {
                const unsigned long long src = L.src[plane];
                const int pitch = L.src_pitch[plane];
                const int sx = L.sx[plane], sy = L.sy[plane], sw = L.sw[plane], sh = L.sh[plane];
                const int bytes = L.kind == AVP_RECT_KIND_YUV ? dst_sb : L.src_bytes;
                const int shift = L.kind == AVP_RECT_KIND_YUV ? dst_shift : L.src_shift;
                const float mul = L.mul;
                const RowTaps taps = row_taps(sy, sh, oy, ldh);
#pragma unroll
                for (int p = 0; p < px; ++p) {
                    const int ox = X0 + p - ldx;
                    if (ox < 0 || ox >= ldw) continue;
#pragma unroll
                    for (int c = 0; c < lanes; ++c) {
                        float sv = sample_lane(src, kArrays && L.yuv_texture, pitch, sx, sw,
                                               lanes, c, bytes, shift, ox, ldw, taps);
                        if (L.kind == AVP_RECT_KIND_PROMOTE) sv *= mul;
                        acc[p][c] = stored_code(sv);
                    }
                }
                continue;
            }

            if (!kRgb) continue;   // host never puts RGB entries in a YUV-only table
            // Packed RGB(A): plane 0 is one luma sample; plane 1 is one chroma block of bw x bh luma
            // positions averaged, skipping positions outside the rect or the canvas.
            const bool blend = L.kind == AVP_RECT_KIND_RGBA || L.kind == AVP_RECT_KIND_RGBA_TEX;
            const bool textured = L.kind >= AVP_RECT_KIND_RGB_TEX;
            const unsigned long long tex = L.src[0];
            const unsigned char *src = (const unsigned char *)L.src[0];   // packed RGB: one plane for both passes
            const int pitch = L.src_pitch[0];
            const int sx = L.sx[0], sy = L.sy[0], sw = L.sw[0], sh = L.sh[0];
            const int dx = L.dx[0], dy = L.dy[0], dw = L.dw[0], dh = L.dh[0];
            const float xs = float(sw) / dw, ys = float(sh) / dh;
            const int step = L.step, r_off = L.r_off, g_off = L.g_off, b_off = L.b_off, a_off = L.a_off;
            const int premultiplied = L.premultiplied;
            const float opacity = kOpacity ? L.mul : 1.f;
#pragma unroll 1
            for (int p = 0; p < px; ++p) {
                const int X = X0 + p;
                const int ox = X - ldx;
                if (ox < 0 || ox >= ldw) continue;
                if (!kChroma) {
                    float rgb[3];
                    const float fx = (ox + 0.5f) * xs - 0.5f, fy = (oy + 0.5f) * ys - 0.5f;
                    if (textured) sample_rgb_tex(tex, sx, sy, sw, sh, r_off, g_off, b_off, fx, fy, rgb);
                    else sample_rgb(src, pitch, sx, sy, sw, sh, step, r_off, g_off, b_off, fx, fy, rgb);
                    if (!blend) {
                        convert_graphic_rgb(rgb, transfer, white, peak);
                        const float luma = graphic_luma(rgb, transfer) * dst_scale;
                        acc[p][0] = stored_code(min(max(luma, 0.f), maxv));
                    } else {
                        const float a = textured ? sample_alpha_tex(tex, sx, sy, sw, sh, a_off, fx, fy)
                                                 : sample_alpha(src, pitch, sx, sy, sw, sh, step, a_off, fx, fy);
                        if (premultiplied)
                            for (int c = 0; c < 3; ++c)
                                rgb[c] = a > 0.f ? min(rgb[c] / a, 255.f) : 0.f;
                        convert_graphic_rgb(rgb, transfer, white, peak);
                        const float luma = graphic_luma(rgb, transfer) * dst_scale;
                        // Opacity weights coverage only: the colour above was un-premultiplied
                        // by the sampled alpha, so a fading key keeps its colours.
                        const float w = kOpacity ? a * opacity : a;
                        acc[p][0] = stored_code(min(max(w * luma + (1.f - w) * acc[p][0], 0.f), maxv));
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
                        if (textured) sample_rgb_tex(tex, sx, sy, sw, sh, r_off, g_off, b_off, fx, fy, rgb);
                        else sample_rgb(src, pitch, sx, sy, sw, sh, step, r_off, g_off, b_off, fx, fy, rgb);
                        if (!blend) {
                            convert_graphic_rgb(rgb, transfer, white, peak);
                            sum[0] += rgb[0]; sum[1] += rgb[1]; sum[2] += rgb[2];
                        } else {
                            const float a = textured ? sample_alpha_tex(tex, sx, sy, sw, sh, a_off, fx, fy)
                                                     : sample_alpha(src, pitch, sx, sy, sw, sh, step, a_off, fx, fy);
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
                    acc[p][0] = stored_code(min(max(cb, 0.f), maxv));
                    acc[p][1] = stored_code(min(max(cr, 0.f), maxv));
                } else {
                    if (cnt == 0 || sum_a <= 0.f) continue;
                    // The colour is alpha-weighted, so opacity scales only the block coverage.
                    const float r = sum[0] / sum_a, g = sum[1] / sum_a, b = sum[2] / sum_a;
                    const float a = kOpacity ? sum_a / cnt * opacity : sum_a / cnt;
                    graphic_chroma(r, g, b, transfer, cb, cr);
                    cb *= dst_scale; cr *= dst_scale;
                    acc[p][0] = stored_code(min(max(a * cb + (1.f - a) * acc[p][0], 0.f), maxv));
                    acc[p][1] = stored_code(min(max(a * cr + (1.f - a) * acc[p][1], 0.f), maxv));
                }
            }
        }
    }

    // Store: the group's samples in one aligned vector write for the two common layouts
    // (4 luma samples; 2 chroma pairs), scalar stores otherwise (tail groups, packed canvases).
    unsigned char *row = kChroma ? dst_uv + Y * uv_pitch : dst_y + Y * y_pitch;
    const int valid = min(px, pw - X0);
    if (valid == px && lanes <= 2) {
        if (dst_sb == 1) {
            uchar4 out;
            if (!kChroma)
                out = make_uchar4((unsigned char)(acc[0][0] + 0.5f), (unsigned char)(acc[1 % px][0] + 0.5f),
                                  (unsigned char)(acc[2 % px][0] + 0.5f), (unsigned char)(acc[3 % px][0] + 0.5f));
            else
                out = make_uchar4((unsigned char)(acc[0][0] + 0.5f), (unsigned char)(acc[0][1 % lanes] + 0.5f),
                                  (unsigned char)(acc[1 % px][0] + 0.5f), (unsigned char)(acc[1 % px][1 % lanes] + 0.5f));
            *(uchar4 *)(row + X0 * lanes) = out;
        } else {
            ushort4 out;
            if (!kChroma)
                out = make_ushort4((unsigned short)((unsigned short)(acc[0][0] + 0.5f) << dst_shift),
                                   (unsigned short)((unsigned short)(acc[1 % px][0] + 0.5f) << dst_shift),
                                   (unsigned short)((unsigned short)(acc[2 % px][0] + 0.5f) << dst_shift),
                                   (unsigned short)((unsigned short)(acc[3 % px][0] + 0.5f) << dst_shift));
            else
                out = make_ushort4((unsigned short)((unsigned short)(acc[0][0] + 0.5f) << dst_shift),
                                   (unsigned short)((unsigned short)(acc[0][1 % lanes] + 0.5f) << dst_shift),
                                   (unsigned short)((unsigned short)(acc[1 % px][0] + 0.5f) << dst_shift),
                                   (unsigned short)((unsigned short)(acc[1 % px][1 % lanes] + 0.5f) << dst_shift));
            *(ushort4 *)(row + X0 * lanes * 2) = out;
        }
        return;
    }
#pragma unroll
    for (int p = 0; p < px; ++p) {
        if (p >= valid) break;
#pragma unroll
        for (int c = 0; c < lanes; ++c)
            store_sample(row + ((X0 + p) * lanes + c) * dst_sb, dst_sb, dst_shift, acc[p][c]);
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

// Semiplanar YUV canvas, any layer kind (gridDim.z = 2: luma, chroma).
extern "C" __global__ void __launch_bounds__(256) composite_planes(AVP_COMPOSITE_ARGS) {
    if (blockIdx.z) composite_body<true, 2, true>(AVP_COMPOSITE_PASS);
    else composite_body<true, 1, false>(AVP_COMPOSITE_PASS);
}
// Semiplanar YUV canvas, no packed-RGB layers in the table (lean, the common case).
extern "C" __global__ void __launch_bounds__(256) composite_planes_yuv(AVP_COMPOSITE_ARGS) {
    if (blockIdx.z) composite_body<false, 2, true>(AVP_COMPOSITE_PASS);
    else composite_body<false, 1, false>(AVP_COMPOSITE_PASS);
}
// Semiplanar YUV canvas where a blended RGBA layer is faded (table mul < 1): composite_planes
// plus the opacity weight. The host picks it only for such frames.
extern "C" __global__ void __launch_bounds__(256) composite_planes_opacity(AVP_COMPOSITE_ARGS) {
    if (blockIdx.z) composite_body<true, 2, true, true>(AVP_COMPOSITE_PASS);
    else composite_body<true, 1, false, true>(AVP_COMPOSITE_PASS);
}
// Packed 8-bit RGB canvas with 4 bytes per pixel (rgb0/bgr0/rgba/bgra), same-format sources (gridDim.z = 1).
extern "C" __global__ void __launch_bounds__(256) composite_planes_packed4(AVP_COMPOSITE_ARGS) {
    composite_body<false, 4, false>(AVP_COMPOSITE_PASS);
}

// Keep array fetches out of the linear-only kernels' instruction/register budget.
// These variants accept a mix of pointer and array YUV layers in the same draw.
extern "C" __global__ void __launch_bounds__(256) composite_planes_array(AVP_COMPOSITE_ARGS) {
    if (blockIdx.z) composite_body<true, 2, true, false, true>(AVP_COMPOSITE_PASS);
    else composite_body<true, 1, false, false, true>(AVP_COMPOSITE_PASS);
}
extern "C" __global__ void __launch_bounds__(256) composite_planes_yuv_array(AVP_COMPOSITE_ARGS) {
    if (blockIdx.z) composite_body<false, 2, true, false, true>(AVP_COMPOSITE_PASS);
    else composite_body<false, 1, false, false, true>(AVP_COMPOSITE_PASS);
}
extern "C" __global__ void __launch_bounds__(256) composite_planes_opacity_array(AVP_COMPOSITE_ARGS) {
    if (blockIdx.z) composite_body<true, 2, true, true, true>(AVP_COMPOSITE_PASS);
    else composite_body<true, 1, false, true, true>(AVP_COMPOSITE_PASS);
}
