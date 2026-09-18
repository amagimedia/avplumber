#pragma once
// Compositor rules on top of libavutil pixel descriptors: storage words and
// padding, chroma alignment, per-plane byte regions, per-plane clear values and
// which source formats a canvas accepts. Plane counts come straight from
// av_pix_fmt_count_planes. No CUDA, no node state.
#include <algorithm>
#include <cstdint>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
}

namespace avp::mixer {

/// Chroma subsampling as luma alignment (2 for 4:2:0 horizontally); 1 for unknown formats.
inline int chromaXAlign(AVPixelFormat fmt) {
    int h = 0, v = 0;
    return av_pix_fmt_get_chroma_sub_sample(fmt, &h, &v) < 0 ? 1 : 1 << h;
}

inline int chromaYAlign(AVPixelFormat fmt) {
    int h = 0, v = 0;
    return av_pix_fmt_get_chroma_sub_sample(fmt, &h, &v) < 0 ? 1 : 1 << v;
}

inline int alignCoord(int v, int a) {
    if (a <= 1)
        return v;
    return v & ~(a - 1);
}

inline bool clipRect(int &x, int &y, int &rw, int &rh, int lim_w, int lim_h) {
    if (rw <= 0 || rh <= 0 || lim_w <= 0 || lim_h <= 0)
        return false;
    int x2 = x + rw;
    int y2 = y + rh;
    x = std::max(0, std::min(x, lim_w));
    y = std::max(0, std::min(y, lim_h));
    x2 = std::max(0, std::min(x2, lim_w));
    y2 = std::max(0, std::min(y2, lim_h));
    rw = x2 - x;
    rh = y2 - y;
    return rw > 0 && rh > 0;
}

/// Bytes per stored logical sample on every plane of fmt: 1 for 8-bit, 2 for deeper storage.
inline int sampleBytes(AVPixelFormat fmt) {
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(fmt);
    return d && d->comp[0].depth + d->comp[0].shift > 8 ? 2 : 1;
}

/// Least-significant padding bits below each stored sample (6 for P210/P010, else 0).
inline int storageShift(AVPixelFormat fmt) {
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(fmt);
    return d ? d->comp[0].shift : 0;
}

/// Rectangle in luma/packed pixel units -> byte offset region for a given plane (for memcpy2D).
/// Horizontal extents are av_image_get_linesize of the left and right edges (chroma subsampling
/// and the component step, NV12: 2, P210: 4, come from the descriptor); the left edge must be
/// chroma-aligned. Rows follow log2_chroma_h on the chroma planes of multi-plane formats.
inline void lumaRectToPlaneRegion(AVPixelFormat fmt, int lx, int ly, int lw, int lh, int plane, int &bx,
                                  int &by, int &bw_bytes, int &bh) {
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(fmt);
    const int left = av_image_get_linesize(fmt, lx, plane), right = av_image_get_linesize(fmt, lx + lw, plane);
    if (!d || left < 0 || right < 0) {
        bx = by = bw_bytes = bh = 0;
        return;
    }
    const bool chroma = av_pix_fmt_count_planes(fmt) > 1 && (plane == 1 || plane == 2);
    const int sy = chroma ? d->log2_chroma_h : 0;
    bx = left;
    bw_bytes = right - left;
    by = ly >> sy;
    bh = AV_CEIL_RSHIFT(ly + lh, sy) - by;
}

// Returns true when src_fmt can be overlaid onto canvas_fmt by treating the source as fully opaque:
// canvas must have a separate alpha plane, source must not, and all other plane layouts must match.
inline bool isAlphaCompatible(AVPixelFormat src_fmt, AVPixelFormat canvas_fmt) {
    const AVPixFmtDescriptor *sd = av_pix_fmt_desc_get(src_fmt);
    const AVPixFmtDescriptor *cd = av_pix_fmt_desc_get(canvas_fmt);
    if (!sd || !cd) return false;
    if (!(cd->flags & AV_PIX_FMT_FLAG_ALPHA)) return false;
    if (sd->flags & AV_PIX_FMT_FLAG_ALPHA) return false;
    if (sd->nb_components != cd->nb_components - 1) return false;
    if (sd->log2_chroma_w != cd->log2_chroma_w) return false;
    if (sd->log2_chroma_h != cd->log2_chroma_h) return false;
    if (sd->comp[0].depth != cd->comp[0].depth || sd->comp[0].shift != cd->comp[0].shift) return false;
    if (av_pix_fmt_count_planes(src_fmt) != av_pix_fmt_count_planes(canvas_fmt) - 1) return false;
    return true;
}

// Packed 8-bit RGB with 3 or 4 bytes per pixel (rgb0, bgr0, rgba, bgra, rgb24, ...): the compositor
// converts such sources onto an NV12 canvas on the GPU, so browser pages and video mix freely.
inline bool isPackedRgb8(AVPixelFormat fmt, int &step, int &r_off, int &g_off, int &b_off) {
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(fmt);
    if (!d || !(d->flags & AV_PIX_FMT_FLAG_RGB) || (d->flags & (AV_PIX_FMT_FLAG_PLANAR | AV_PIX_FMT_FLAG_BITSTREAM)))
        return false;
    if (d->nb_components < 3) return false;
    for (int c = 0; c < 3; ++c)
        if (d->comp[c].depth != 8 || d->comp[c].plane != 0) return false;
    step = d->comp[0].step;
    r_off = d->comp[0].offset;
    g_off = d->comp[1].offset;
    b_off = d->comp[2].offset;
    return step == 3 || step == 4;
}

/// Byte offset of alpha inside a packed 8-bit pixel, or -1 when there is none.
inline int packedAlphaOffset(AVPixelFormat fmt) {
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(fmt);
    if (!d || !(d->flags & AV_PIX_FMT_FLAG_ALPHA) || d->nb_components < 4) return -1;
    const AVComponentDescriptor &a = d->comp[3];
    return (a.depth == 8 && a.plane == 0) ? a.offset : -1;
}

// The fused RGB(A) path renders onto semiplanar YUV canvases (NV12, P210): two
// planes, no alpha, interleaved Cb/Cr. Planar-chroma canvases reject packed RGB
// sources instead of growing a third store path here. SDR graphics are embedded
// into the explicitly selected SDR/HLG/PQ canvas color contract.
inline bool isRgbToYuvConvertible(AVPixelFormat src_fmt, AVPixelFormat canvas_fmt) {
    const AVPixFmtDescriptor *cd = av_pix_fmt_desc_get(canvas_fmt);
    int step, r, g, b;
    return cd && !(cd->flags & (AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_ALPHA)) &&
           cd->nb_components == 3 && av_pix_fmt_count_planes(canvas_fmt) == 2 &&
           isPackedRgb8(src_fmt, step, r, g, b);
}

// A lower-depth semiplanar YUV source (e.g. NV12 from NVDEC) drawn onto a deeper
// semiplanar canvas (P210): the fused scaler promotes its codes by 2^(dbits-sbits)
// (NV12->P210 is <<2, 16->64 / 235->940) and resamples the chroma footprint,
// so an 8-bit clip mixes onto a 10-bit program with no separate convert node.
// SDR only, like the RGB path; the source keeps its own subsampling.
inline bool isYuvPromoteConvertible(AVPixelFormat src_fmt, AVPixelFormat canvas_fmt) {
    const AVPixFmtDescriptor *sd = av_pix_fmt_desc_get(src_fmt);
    const AVPixFmtDescriptor *cd = av_pix_fmt_desc_get(canvas_fmt);
    if (!sd || !cd || src_fmt == canvas_fmt) return false;
    if ((sd->flags | cd->flags) & (AV_PIX_FMT_FLAG_RGB | AV_PIX_FMT_FLAG_ALPHA)) return false;
    if (sd->nb_components != 3 || cd->nb_components != 3) return false;
    if (av_pix_fmt_count_planes(src_fmt) != 2 || av_pix_fmt_count_planes(canvas_fmt) != 2) return false;   // semiplanar UV
    return sd->comp[0].depth <= cd->comp[0].depth;
}

/// Every source format the canvas accepts: identical, opaque-onto-alpha, packed RGB, or promotable YUV.
inline bool canvasAccepts(AVPixelFormat src_fmt, AVPixelFormat canvas_fmt) {
    return src_fmt == canvas_fmt || isAlphaCompatible(src_fmt, canvas_fmt) ||
           isRgbToYuvConvertible(src_fmt, canvas_fmt) || isYuvPromoteConvertible(src_fmt, canvas_fmt);
}

// Returns the plane index of the alpha component for planar formats, or -1 if there is none /
// the format is packed (all components on plane 0).
inline int alphaPlaneIndex(AVPixelFormat fmt) {
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(fmt);
    if (!d || !(d->flags & AV_PIX_FMT_FLAG_ALPHA)) return -1;
    int max_plane = -1;
    for (int i = 0; i < d->nb_components; ++i)
        max_plane = std::max(max_plane, d->comp[i].plane);
    return max_plane > 0 ? max_plane : -1;
}

inline uint16_t blackLumaValue(AVPixelFormat fmt, const AVFrame *color_src) {
    if (color_src && color_src->color_range == AVCOL_RANGE_JPEG)
        return 0;
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(fmt);
    return (uint16_t)(16 << ((d ? d->comp[0].depth : 8) - 8));
}

/// Logical sample value that clears `plane` to opaque black; false when the format has no
/// single-value clear (packed YUV, packed RGBA).
inline bool planeClearValue(AVPixelFormat fmt, const AVFrame *color_src, int plane, uint16_t &value) {
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(fmt);
    if (!d)
        return false;
    const int depth = d->comp[0].depth;

    const int alpha_p = alphaPlaneIndex(fmt);
    if (plane == alpha_p) {
        value = (uint16_t)((1 << depth) - 1);   // 10-bit opaque is 1023, not 255 << 2.
        return true;
    }

    if (d->flags & AV_PIX_FMT_FLAG_RGB) {
        // Packed RGB without alpha can be cleared with all-zero bytes.
        // Packed RGBA needs a byte pattern to make opaque black; leave it unsupported here.
        if ((d->flags & AV_PIX_FMT_FLAG_ALPHA) && alpha_p < 0)
            return false;
        value = 0;
        return true;
    }

    const int planes = av_pix_fmt_count_planes(fmt);
    if (planes == 1) {
        if (d->nb_components == 1) {
            value = blackLumaValue(fmt, color_src);
            return true;
        }
        // Packed YUV (e.g. yuyv422) requires a repeating Y/Cb/Y/Cr pattern.
        return false;
    }

    // Chroma sits on plane 1 (semiplanar NV12/NV21/P210) or planes 1-2 (planar).
    value = (plane == 1 || plane == 2) ? (uint16_t)(1 << (depth - 1)) : blackLumaValue(fmt, color_src);
    return true;
}

}
