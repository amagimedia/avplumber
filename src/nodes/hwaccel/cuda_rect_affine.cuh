#pragma once
// The affine entries' drawing (composite_planes_yuv_affine, composite_planes_affine): included by
// cuda_rect_scale.cu after its samplers and filters (LanePlane, the cubic, Hermite and multisample
// filters, sample_rgb/sample_alpha and their texture twins), which it uses as they are.

// ---------------------------------------------------------------------------
// What only the affine entries draw (kAffine): layers whose table entry has `affine` set, through
// the per-plane inverse map m (compositor_affine.hpp). Each sample maps its centre, tests it against
// the crop's edges with an analytic 1-sample soft edge (coverage), samples with the layer's filter at
// the mapped position and blends by the coverage; a sample with coverage 1 is written as the filter
// entries write it. The map and coverage use explicit fused multiply-adds, so a CPU reference can
// reproduce the tap positions exactly (tests/cuda/test_rect_affine.cu).
// ---------------------------------------------------------------------------

// Coverage of the destination sample whose centre maps to (u, v): the distance to the nearer edge
// on each axis, in destination samples, as a 1-sample ramp.
__device__ __forceinline__ float affine_coverage(float u, float v, const float *e, const float *aa) {
    const float cu = fminf(fmaxf(__fmaf_rn(fminf(u - e[0], e[2] - u), aa[0], 0.5f), 0.f), 1.f);
    const float cv = fminf(fmaxf(__fmaf_rn(fminf(v - e[1], e[3] - v), aa[1], 0.5f), 0.f), 1.f);
    return cu * cv;
}

// Every lane of a YUV/PROMOTE/DEMOTE layer's plane at source position (fx, fy), under its filter.
template <int kLanes>
__device__ __forceinline__ void affine_lanes(const AvpRectLayer &L, const LanePlane &P, int plane, float fx, float fy,
                                             float *out) {
    if (L.filter == AVP_RECT_FILTER_BICUBIC) {
        const CubicFilter F(cubic_taps_at(P.sy, P.sh, fy, L.filter_param), L.filter_param);
        const CubicTaps cols = cubic_taps_at(P.sx, P.sw, fx, L.filter_param);
#pragma unroll
        for (int c = 0; c < kLanes; ++c) out[c] = F.sample(P, cols, c);
    } else if (L.filter == AVP_RECT_FILTER_BICUBIC_A0) {
        const CubicA0Filter F(CubicA0Filter::taps_at(P.sy, P.sh, fy));
        const RowTaps cols = CubicA0Filter::taps_at(P.sx, P.sw, fx);
#pragma unroll
        for (int c = 0; c < kLanes; ++c) out[c] = F.sample(P, cols, c);
    } else if (L.filter == AVP_RECT_FILTER_MULTISAMPLE4) {
        const MultisampleFilter<4> F(fy, L.footprint[plane][0], L.footprint[plane][1]);
#pragma unroll
        for (int c = 0; c < kLanes; ++c) out[c] = F.sample(P, fx, c);
    } else if (L.filter == AVP_RECT_FILTER_MULTISAMPLE8) {
        const MultisampleFilter<8> F(fy, L.footprint[plane][0], L.footprint[plane][1]);
#pragma unroll
        for (int c = 0; c < kLanes; ++c) out[c] = F.sample(P, fx, c);
    } else if (L.filter == AVP_RECT_FILTER_AREA) {
        const AreaFilter F(fy, L.footprint[plane][0], L.footprint[plane][1]);
#pragma unroll
        for (int c = 0; c < kLanes; ++c) out[c] = F.sample(P, fx, c);
    } else {
        const RowTaps rows = row_taps_at(P.sy, P.sh, fy);
#pragma unroll
        for (int c = 0; c < kLanes; ++c) out[c] = P.bilinear(fx, rows, c);
    }
}

// The lane groups of one thread from an affine YUV/PROMOTE/DEMOTE layer.
template <int kPx, int kLanes>
__device__ __forceinline__ void draw_affine_yuv(float (*acc)[kLanes], const AvpRectLayer &L, const LanePlane &P, int plane,
                                                int X0, int Y, int ldx, int ldw, float gain, float maxv) {
    const float *m = L.m[plane];
    const float y = Y + 0.5f, uy = __fmaf_rn(m[1], y, m[2]), vy = __fmaf_rn(m[4], y, m[5]);
#pragma unroll
    for (int p = 0; p < kPx; ++p) {
        const int X = X0 + p;
        if (X < ldx || X >= ldx + ldw) continue;
        const float x = X + 0.5f, u = __fmaf_rn(m[0], x, uy), v = __fmaf_rn(m[3], x, vy);
        const float cov = affine_coverage(u, v, L.edge[plane], L.aa[plane]) * (1.f - L.transparency);
        if (!(cov > 0.f)) continue;
        float s[kLanes];
        // The samplers take positions relative to the taps' rect, as the plain path does.
        affine_lanes<kLanes>(L, P, plane, u - 0.5f - P.sx, v - 0.5f - P.sy, s);
#pragma unroll
        for (int c = 0; c < kLanes; ++c) {
            const float value = fminf(fmaxf(s[c] * gain, 0.f), maxv);
            acc[p][c] = stored_code(cov >= 1.f ? value : cov * value + (1.f - cov) * acc[p][c]);
        }
    }
}

// The lane groups of one thread from an affine packed RGB(A) layer. Each luma position maps through
// plane 0; its weight is coverage x alpha (alpha 1 for opaque RGB), so an opaque key blends at its
// soft edge. Chroma averages its block's positions by weight, as the plain path does for alpha.
template <int kPx, int kLanes, bool kChroma, bool kOpacity>
__device__ __forceinline__ void draw_affine_rgb(float (*acc)[kLanes], const AvpRectLayer &L, int X0, int Y, int ldx, int ldw,
                                                int sub_x, int sub_y, int canvas_w, int canvas_h, int transfer,
                                                float white, float peak, int dst_scale, float maxv) {
    const bool blend = L.kind == AVP_RECT_KIND_RGBA || L.kind == AVP_RECT_KIND_RGBA_TEX;
    const bool textured = L.kind >= AVP_RECT_KIND_RGB_TEX;
    const unsigned long long tex = L.src[0];
    const unsigned char *src = (const unsigned char *)L.src[0];
    const int pitch = L.src_pitch[0], sx = L.sx[0], sy = L.sy[0], sw = L.sw[0], sh = L.sh[0];
    const float opacity = kOpacity ? L.mul : 1.f;
    const float *m = L.m[0];
    // Weight of luma position (lx, ly), its colour converted in `rgb`; 0 when it draws nothing.
    auto weight = [&](int lx, int ly, float *rgb) -> float {
        const float x = lx + 0.5f, y = ly + 0.5f;
        const float u = __fmaf_rn(m[0], x, __fmaf_rn(m[1], y, m[2])), v = __fmaf_rn(m[3], x, __fmaf_rn(m[4], y, m[5]));
        const float cov = affine_coverage(u, v, L.edge[0], L.aa[0]) * (1.f - L.transparency);
        if (!(cov > 0.f)) return 0.f;
        const float fx = u - 0.5f - sx, fy = v - 0.5f - sy;   // relative to the taps' rect
        const float a = !blend ? 1.f : (textured ? sample_alpha_tex(tex, sx, sy, sw, sh, L.a_off, fx, fy)
                                                 : sample_alpha(src, pitch, sx, sy, sw, sh, L.step, L.a_off, fx, fy));
        if (a <= 0.f) return 0.f;
        if (textured) sample_rgb_tex(tex, sx, sy, sw, sh, L.r_off, L.g_off, L.b_off, fx, fy, rgb);
        else sample_rgb(src, pitch, sx, sy, sw, sh, L.step, L.r_off, L.g_off, L.b_off, fx, fy, rgb);
        if (blend && L.premultiplied)
            for (int c = 0; c < 3; ++c) rgb[c] = min(rgb[c] / a, 255.f);
        convert_graphic_rgb(rgb, transfer, white, peak);
        return cov * a;
    };
#pragma unroll 1
    for (int p = 0; p < kPx; ++p) {
        const int X = X0 + p;
        if (X < ldx || X >= ldx + ldw) continue;
        if (!kChroma) {
            float rgb[3];
            const float w = weight(X, Y, rgb) * opacity;
            if (w <= 0.f) continue;
            const float luma = graphic_luma(rgb, transfer) * dst_scale;
            acc[p][0] = stored_code(min(max(w * luma + (1.f - w) * acc[p][0], 0.f), maxv));
            continue;
        }
        const int bw = 1 << sub_x, bh = 1 << sub_y;
        float sum[3] = {0.f, 0.f, 0.f}, sum_a = 0.f;
        int cnt = 0;
        for (int j = 0; j < bh; ++j)
            for (int i = 0; i < bw; ++i) {
                const int lx = (X << sub_x) + i, ly = (Y << sub_y) + j;
                if (lx >= canvas_w || ly >= canvas_h) continue;
                ++cnt;   // positions outside the quad count in the block's coverage
                float rgb[3];
                const float w = weight(lx, ly, rgb);
                if (w <= 0.f) continue;
                sum[0] += w * rgb[0]; sum[1] += w * rgb[1]; sum[2] += w * rgb[2];
                sum_a += w;
            }
        if (cnt == 0 || sum_a <= 0.f) continue;
        float cb, cr;
        graphic_chroma(sum[0] / sum_a, sum[1] / sum_a, sum[2] / sum_a, transfer, cb, cr);
        const float a = sum_a / cnt * opacity;
        cb *= dst_scale; cr *= dst_scale;
        acc[p][0] = stored_code(min(max(a * cb + (1.f - a) * acc[p][0], 0.f), maxv));
        acc[p][1 % kLanes] = stored_code(min(max(a * cr + (1.f - a) * acc[p][1 % kLanes], 0.f), maxv));
    }
}
