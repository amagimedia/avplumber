#include "../node_common.hpp"
#include "../../hwaccel.hpp"
#include "../../hwaccel/CompositorGeometry.hpp"
#ifdef HAVE_CUDA_RECT_SCALE
#include "../../../objs/src/nodes/hwaccel/cuda_rect_scale.ptx.h"
#endif
#include "../../SharedTimeline.hpp"
#include "../../mixer/Playout.hpp"
#include "../../mixer/MonotonicClock.hpp"
#include <sstream>
#include <cuda_loader/cuda_drvapi_dynlink_cuda.h>

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/pixdesc.h>
}

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

static int check_cu(CUresult err, const char *func) {
    if (err == CUDA_SUCCESS)
        return 0;
    const char *err_name = nullptr;
    const char *err_string = nullptr;
    if (cuGetErrorName && cuGetErrorString) {
        cuGetErrorName(err, &err_name);
        cuGetErrorString(err, &err_string);
    }
    logstream << "cuda_rect_overlay: " << func << " failed: " << (err_name ? err_name : "?") << ": "
              << (err_string ? err_string : "?");
    return -1;
}

#define CHECK_CU(x) check_cu((x), #x)

struct LayerSpec {
    int dst_x = 0;
    int dst_y = 0;
    int crop_x = 0;
    int crop_y = 0;
    int dst_w = 0;
    int dst_h = 0;
    bool fit = false;
    int source_canvas_w = 0;
    int source_canvas_h = 0;
    int crop_w= 0;
    int crop_h = 0;
    int z = 0;   // draw order: lower first, ties by source index

    bool operator==(const LayerSpec &other) const {
        return dst_x == other.dst_x && dst_y == other.dst_y && z == other.z &&
               crop_x == other.crop_x && crop_y == other.crop_y &&
               crop_w == other.crop_w && crop_h == other.crop_h &&
               dst_w == other.dst_w && dst_h == other.dst_h && fit == other.fit &&
               source_canvas_w == other.source_canvas_w && source_canvas_h == other.source_canvas_h;
    }
};

struct DrawOp {
    const av::VideoFrame *src = nullptr;
    int src_w = 0;
    int src_h = 0;
    LayerSpec layer;

    bool operator==(const DrawOp &other) const {
        return (src != nullptr) == (other.src != nullptr) &&
               src_w == other.src_w && src_h == other.src_h &&
               layer == other.layer;
    }
};

static bool frameUsable(const av::VideoFrame &f) {
    return !f.isNull() && f.isComplete() && f.raw() && f.pts().isValid();
}


static void parseLayerFromJson(const Parameters &obj, LayerSpec &out) {
    out.dst_x = obj.value("dst_x", 0);
    out.dst_y = obj.value("dst_y", 0);
    out.dst_w = obj.value("dst_w", 0);
    out.dst_h = obj.value("dst_h", 0);
    out.z = obj.value("z", 0);
    const std::string fit = obj.value("fit", std::string("stretch"));
    if (fit != "stretch" && fit != "contain")
        throw Error("cuda_rect_overlay: fit must be stretch or contain");
    out.fit = fit == "contain";
    out.source_canvas_w = out.source_canvas_h = 0;
    if (obj.contains("source_canvas")) {
        const auto &canvas = obj.at("source_canvas");
        out.source_canvas_w = canvas.at("w").get<int>();
        out.source_canvas_h = canvas.at("h").get<int>();
        if (!out.fit || out.dst_w <= 0 || out.dst_h <= 0 || out.source_canvas_w <= 0 || out.source_canvas_h <= 0)
            throw Error("cuda_rect_overlay: source_canvas requires positive dimensions and fit=contain");
    }
    if (out.dst_w < 0 || out.dst_h < 0 || (out.dst_w == 0) != (out.dst_h == 0))
        throw Error("cuda_rect_overlay: dst_w and dst_h must both be positive or both omitted");
#ifndef HAVE_CUDA_RECT_SCALE
    if (out.dst_w || out.dst_h)
        throw Error("cuda_rect_overlay: destination sizing requires HAVE_NVCC=1");
#endif
    if (obj.contains("crop") && obj["crop"].is_object()) {
        const auto &c = obj["crop"];
        out.crop_x = c.value("x", 0);
        out.crop_y = c.value("y", 0);
        // 0 means "use remaining source width/height from crop_x/crop_y" (resolved per-frame in processComposite).
        out.crop_w = c.value("w", 0);
        out.crop_h = c.value("h", 0);
    }
    // No crop object → crop_x/y/w/h all stay 0 (full source frame from origin).
}

static std::vector<LayerSpec> parseLayersArray(const Parameters &arr) {
    std::vector<LayerSpec> layers;
    if (!arr.is_array())
        throw Error("cuda_rect_overlay: layers must be an array");
    for (const auto &item : arr) {
        if (!item.is_object())
            throw Error("cuda_rect_overlay: layers entries must be objects");
        LayerSpec s;
        parseLayerFromJson(item, s);
        layers.push_back(s);
    }
    return layers;
}

static std::vector<LayerSpec> parseLayersParam(const Parameters &params) {
    if (!params.contains("layers") || !params["layers"].is_array())
        throw Error("cuda_rect_overlay: layers array required (one entry per input in src order)");
    return parseLayersArray(params["layers"]);
}

static int chromaXAlign(AVPixelFormat sw_fmt) {
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(sw_fmt);
    if (!desc || desc->log2_chroma_w < 0)
        return 1;
    return 1 << desc->log2_chroma_w;
}

static int chromaYAlign(AVPixelFormat sw_fmt) {
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(sw_fmt);
    if (!desc || desc->log2_chroma_h < 0)
        return 1;
    return 1 << desc->log2_chroma_h;
}

static int alignCoord(int v, int a) {
    if (a <= 1)
        return v;
    return v & ~(a - 1);
}

static bool clipRect(int &x, int &y, int &rw, int &rh, int lim_w, int lim_h) {
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

static int numPlanes(AVPixelFormat fmt) {
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(fmt);
    if (!d)
        return 0;
    int np = 0;
    for (int i = 0; i < d->nb_components; ++i)
        np = std::max(np, d->comp[i].plane + 1);
    return np;
}

/// Rectangle in luma/packed pixel units -> byte offset region for a given plane (for memcpy2D).
static void lumaRectToPlaneRegion(AVPixelFormat fmt, int lx, int ly, int lw, int lh, int plane, int &bx,
                                 int &by, int &bw_bytes, int &bh) {
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(fmt);
    if (!d) {
        bx = by = bw_bytes = bh = 0;
        return;
    }
    if (fmt == AV_PIX_FMT_NV12) {
        if (plane == 0) {
            bx = lx;
            by = ly;
            bw_bytes = lw;
            bh = lh;
            return;
        }
        const int sx0 = lx >> d->log2_chroma_w;
        const int sy0 = ly >> d->log2_chroma_h;
        const int sx1 = AV_CEIL_RSHIFT(lx + lw, d->log2_chroma_w);
        const int sy1 = AV_CEIL_RSHIFT(ly + lh, d->log2_chroma_h);
        bx = sx0 * 2;
        by = sy0;
        bw_bytes = (sx1 - sx0) * 2;
        bh = sy1 - sy0;
        return;
    }
    const int np = numPlanes(fmt);
    if (np == 1) {
        int step = 1;
        for (int i = 0; i < d->nb_components; ++i)
            step = std::max(step, d->comp[i].step);
        bx = lx * step;
        by = ly;
        bw_bytes = lw * step;
        bh = lh;
        return;
    }
    // Planar YUV (+ alpha): chroma on planes 1–2 follows log2_chroma_*; other planes match luma grid.
    const int sx = (plane == 1 || plane == 2) ? d->log2_chroma_w : 0;
    const int sy = (plane == 1 || plane == 2) ? d->log2_chroma_h : 0;
    const int x0 = lx >> sx;
    const int y0 = ly >> sy;
    const int x1 = AV_CEIL_RSHIFT(lx + lw, sx);
    const int y1 = AV_CEIL_RSHIFT(ly + lh, sy);
    int step = 1;
    for (int i = 0; i < d->nb_components; ++i) {
        if (d->comp[i].plane == plane) {
            step = std::max(1, d->comp[i].step);
            break;
        }
    }
    bx = x0 * step;
    by = y0;
    bw_bytes = (x1 - x0) * step;
    bh = y1 - y0;
}

static bool memcpy2d_async(CUstream stream, CUdeviceptr dst, size_t dst_pitch, size_t dst_x_off_bytes,
                           CUdeviceptr src, size_t src_pitch, size_t src_x_off_bytes, size_t width_bytes,
                           size_t height) {
    CUDA_MEMCPY2D cpy{};
    cpy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cpy.srcDevice = src + (CUdeviceptr)src_x_off_bytes;
    cpy.srcPitch = src_pitch;
    cpy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cpy.dstDevice = dst + (CUdeviceptr)dst_x_off_bytes;
    cpy.dstPitch = dst_pitch;
    cpy.WidthInBytes = width_bytes;
    cpy.Height = height;
    return CHECK_CU(cuMemcpy2DAsync(&cpy, stream)) == 0;
}

static bool blitLayerPlanes(CUstream stream, AVPixelFormat sw_fmt, const AVFrame *src, const AVFrame *dst,
                            int src_luma_x, int src_luma_y, int lw, int lh, int dst_luma_x, int dst_luma_y) {
    const int planes = numPlanes(sw_fmt);
    for (int p = 0; p < planes && p < AV_NUM_DATA_POINTERS; ++p) {
        if (!src->data[p] || !dst->data[p])
            continue;
        int sx, sy, sw_bytes, sh;
        lumaRectToPlaneRegion(sw_fmt, src_luma_x, src_luma_y, lw, lh, p, sx, sy, sw_bytes, sh);
        int dx, dy, dw_bytes, dh;
        lumaRectToPlaneRegion(sw_fmt, dst_luma_x, dst_luma_y, lw, lh, p, dx, dy, dw_bytes, dh);
        if (sw_bytes <= 0 || sh <= 0 || dw_bytes <= 0 || dh <= 0)
            continue;
        const size_t src_pitch = (size_t)src->linesize[p];
        const size_t dst_pitch = (size_t)dst->linesize[p];
        CUdeviceptr sbase = (CUdeviceptr)(uintptr_t)src->data[p];
        CUdeviceptr dbase = (CUdeviceptr)(uintptr_t)dst->data[p];
        const size_t src_off = (size_t)sy * src_pitch + (size_t)sx;
        const size_t dst_off = (size_t)dy * dst_pitch + (size_t)dx;
        if (!memcpy2d_async(stream, dbase, dst_pitch, dst_off, sbase, src_pitch, src_off, (size_t)sw_bytes,
                            (size_t)sh))
            return false;
    }
    return true;
}

// Returns true when src_fmt can be overlaid onto canvas_fmt by treating the source as fully opaque:
// canvas must have a separate alpha plane, source must not, and all other plane layouts must match.
static bool isAlphaCompatible(AVPixelFormat src_fmt, AVPixelFormat canvas_fmt) {
    const AVPixFmtDescriptor *sd = av_pix_fmt_desc_get(src_fmt);
    const AVPixFmtDescriptor *cd = av_pix_fmt_desc_get(canvas_fmt);
    if (!sd || !cd) return false;
    if (!(cd->flags & AV_PIX_FMT_FLAG_ALPHA)) return false;
    if (sd->flags & AV_PIX_FMT_FLAG_ALPHA) return false;
    if (sd->nb_components != cd->nb_components - 1) return false;
    if (sd->log2_chroma_w != cd->log2_chroma_w) return false;
    if (sd->log2_chroma_h != cd->log2_chroma_h) return false;
    if (numPlanes(src_fmt) != numPlanes(canvas_fmt) - 1) return false;
    return true;
}

// Packed 8-bit RGB with 3 or 4 bytes per pixel (rgb0, bgr0, rgba, bgra, rgb24, ...): the compositor
// converts such sources onto an NV12 canvas on the GPU, so browser pages and video mix freely.
static bool isPackedRgb8(AVPixelFormat fmt, int &step, int &r_off, int &g_off, int &b_off) {
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

static bool isRgbToNv12Convertible(AVPixelFormat src_fmt, AVPixelFormat canvas_fmt) {
    int step, r, g, b;
    return canvas_fmt == AV_PIX_FMT_NV12 && isPackedRgb8(src_fmt, step, r, g, b);
}

// Returns the plane index of the alpha component for planar formats, or -1 if there is none /
// the format is packed (all components on plane 0).
static int alphaPlaneIndex(AVPixelFormat fmt) {
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(fmt);
    if (!d || !(d->flags & AV_PIX_FMT_FLAG_ALPHA)) return -1;
    int max_plane = -1;
    for (int i = 0; i < d->nb_components; ++i)
        max_plane = std::max(max_plane, d->comp[i].plane);
    return max_plane > 0 ? max_plane : -1;
}

// Set a rectangular region of one plane to a constant byte value.
static void fillPlaneRect(AVPixelFormat fmt, AVFrame *f, int plane,
                           int lx, int ly, int lw, int lh, uint8_t value) {
    if (!f->data[plane] || f->linesize[plane] <= 0) return;
    int bx, by, bw, bh;
    lumaRectToPlaneRegion(fmt, lx, ly, lw, lh, plane, bx, by, bw, bh);
    if (bw <= 0 || bh <= 0) return;
    const size_t pitch = (size_t)f->linesize[plane];
    CUdeviceptr base = (CUdeviceptr)(uintptr_t)f->data[plane];
    CHECK_CU(cuMemsetD2D8(base + (CUdeviceptr)((size_t)by * pitch + (size_t)bx),
                          (unsigned int)pitch, value, (size_t)bw, (size_t)bh));
}

static uint8_t blackLumaValue(const AVFrame *color_src) {
    return color_src && color_src->color_range == AVCOL_RANGE_JPEG ? 0 : 16;
}

static bool planeClearValue(AVPixelFormat fmt, const AVFrame *color_src, int plane, uint8_t &value) {
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(fmt);
    if (!d)
        return false;

    const int alpha_p = alphaPlaneIndex(fmt);
    if (plane == alpha_p) {
        value = 255;
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

    if (fmt == AV_PIX_FMT_NV12 || fmt == AV_PIX_FMT_NV21) {
        value = plane == 0 ? blackLumaValue(color_src) : 128;
        return true;
    }

    const int planes = numPlanes(fmt);
    if (planes == 1) {
        if (d->nb_components == 1) {
            value = blackLumaValue(color_src);
            return true;
        }
        // Packed YUV (e.g. yuyv422) requires a repeating Y/Cb/Y/Cr pattern.
        return false;
    }

    value = (plane == 1 || plane == 2) ? 128 : blackLumaValue(color_src);
    return true;
}

static bool fillFrameBlack(AVPixelFormat fmt, AVFrame *f, const AVFrame *color_src) {
    const int planes = numPlanes(fmt);
    for (int p = 0; p < planes && p < AV_NUM_DATA_POINTERS; ++p) {
        if (!f->data[p])
            continue;
        uint8_t value = 0;
        if (!planeClearValue(fmt, color_src, p, value))
            return false;
        fillPlaneRect(fmt, f, p, 0, 0, f->width, f->height, value);
    }
    return true;
}

} // namespace

class CudaRectOverlay : public NodeMultiInput<av::VideoFrame>,
                        public NodeSingleOutput<av::VideoFrame>,
                        public IVideoFormatSource,
                        public IFrameRateSource,
                        public IInputReset,
                        public TimelineReader,
                        public IInputsObjects {
    std::shared_ptr<HWAccelDevice> hwaccel_;
    AVBufferRef *out_frames_ref_ = nullptr;

    int canvas_w_ = 0;
    int canvas_h_ = 0;
    AVPixelFormat sw_fmt_ = AV_PIX_FMT_NONE;

    std::vector<LayerSpec> default_layers_;
    mutable std::mutex layers_mutex_;
    std::string metadata_key_;
    int debug_log_every_n_ = 0;
    uint64_t frame_counter_ = 0;

    AVCUDADeviceContext *cuda_dev_ = nullptr;
    CUmodule scale_module_ = nullptr;
    CUfunction scale_kernel_ = nullptr;
    CUfunction rgb_kernel_ = nullptr;
    std::string last_ops_desc_;
    bool sent_eof_ = false;

    std::vector<bool> input_eof_;
    std::vector<av::VideoFrame> held_;
    std::vector<bool> held_valid_;
    std::atomic<uint32_t> active_inputs_{~0u};

    // An explicit fps opts live, monotonic-PTS inputs into shared playout.
    // Unclocked callers retain the established timestamp-driven behavior.
    std::unique_ptr<avp::mixer::Playout<av::VideoFrame>> playout_;
    av::Rational frame_rate_{0, 1};
    std::atomic<uint64_t> input_generation_{0};
    std::atomic<int64_t> input_valid_from_ns_{0};
    uint64_t applied_generation_ = 0;
    uint32_t applied_active_mask_ = 0;


    // Bound how long we will wait for every active layer to produce a fresh
    // post-activation frame before emitting with whatever we have. <=0 disables
    // the bound (the historical "wait forever" behavior). The mixer can stall
    // indefinitely if one source is starved (e.g. a freshly switched router
    // output that has not yet been wired) and no timeout is set.
    int64_t warmup_timeout_ms_ = 0;
    int64_t warmup_started_pts_ = -1;

    void freeHwContexts() {
        if (scale_module_) {
            cuCtxSetCurrent(cuda_dev_->cuda_ctx);
            CHECK_CU(cuModuleUnload(scale_module_));
            scale_module_ = nullptr;
        }
        av_buffer_unref(&out_frames_ref_);
    }

    void ensureCudaDevice() {
        if (!hwaccel_ || !hwaccel_->deviceContext() || !hwaccel_->deviceContext()->data)
            throw Error("cuda_rect_overlay: invalid hwaccel device");
        AVHWDeviceContext *devctx = (AVHWDeviceContext *)hwaccel_->deviceContext()->data;
        cuda_dev_ = (AVCUDADeviceContext *)devctx->hwctx;
        if (!cuda_dev_ || !cuda_dev_->cuda_ctx)
            throw Error("cuda_rect_overlay: CUDA hwctx missing");
        if (CHECK_CU(cuCtxSetCurrent(cuda_dev_->cuda_ctx)))
            throw Error("cuda_rect_overlay: cuCtxSetCurrent failed");
    }

    void ensureScaleKernel() {
#ifdef HAVE_CUDA_RECT_SCALE
        if (scale_kernel_) return;
        ensureCudaDevice();
        const std::string image(avpl_rect_scale_ptx, avpl_rect_scale_ptx + avpl_rect_scale_ptx_len);
        if (CHECK_CU(cuModuleLoadDataEx(&scale_module_, image.c_str(), 0, nullptr, nullptr)) ||
            CHECK_CU(cuModuleGetFunction(&scale_kernel_, scale_module_, "scale_plane")) ||
            CHECK_CU(cuModuleGetFunction(&rgb_kernel_, scale_module_, "rgb_to_nv12")))
            throw Error("cuda_rect_overlay: cannot load scaling kernel");
#endif
    }

    /// Draw a packed RGB source onto the NV12 canvas: one fused scale + colour conversion pass.
    void convertRgbLayer(CUstream stream, const AVFrame *src, AVFrame *dst, const LayerSpec &layer,
                         int step, int r_off, int g_off, int b_off) {
        ensureScaleKernel();
        if (!rgb_kernel_) throw Error("cuda_rect_overlay: RGB conversion kernel unavailable");
        int sx = layer.crop_x, sy = layer.crop_y, sw = layer.crop_w, sh = layer.crop_h;
        int dx = layer.dst_x, dy = layer.dst_y;
        int dw = layer.dst_w > 0 ? layer.dst_w : layer.crop_w, dh = layer.dst_w > 0 ? layer.dst_h : layer.crop_h;
        int cw = canvas_w_, ch = canvas_h_;
        CUdeviceptr source = (CUdeviceptr)src->data[0], luma = (CUdeviceptr)dst->data[0],
                    chroma = (CUdeviceptr)dst->data[1];
        int source_pitch = src->linesize[0], luma_pitch = dst->linesize[0], chroma_pitch = dst->linesize[1];
        void *args[] = {&source, &source_pitch, &sx, &sy, &sw, &sh, &step, &r_off, &g_off, &b_off,
                        &luma, &luma_pitch, &chroma, &chroma_pitch, &dx, &dy, &dw, &dh, &cw, &ch};
        const int blocks_x = (dw + 1) / 2, blocks_y = (dh + 1) / 2;
        if (CHECK_CU(cuLaunchKernel(rgb_kernel_, (blocks_x + 31) / 32, (blocks_y + 7) / 8, 1,
                                    32, 8, 1, 0, stream, args, nullptr)))
            throw Error("cuda_rect_overlay: RGB conversion launch failed");
    }

    void scaleLayer(CUstream stream, const AVFrame *src, AVFrame *dst, const LayerSpec &layer) {
        const auto *desc = av_pix_fmt_desc_get(sw_fmt_);
        for (int c = 0; c < desc->nb_components; ++c)
            if (desc->comp[c].depth != 8)
                throw Error("cuda_rect_overlay: scaling currently requires 8-bit components");
        ensureScaleKernel();
        if (!scale_kernel_) throw Error("cuda_rect_overlay: scaling kernel unavailable");
        for (int p = 0; p < numPlanes(sw_fmt_); ++p) {
            if (!src->data[p]) continue; // Opaque input on an alpha canvas.
            int lanes = 1;
            for (int c = 0; c < desc->nb_components; ++c)
                if (desc->comp[c].plane == p) lanes = std::max(lanes, desc->comp[c].step);
            int sx, sy, sw, sh, dx, dy, dw, dh, cx, cy, cw, ch;
            lumaRectToPlaneRegion(sw_fmt_, layer.crop_x, layer.crop_y, layer.crop_w, layer.crop_h,
                                  p, sx, sy, sw, sh);
            lumaRectToPlaneRegion(sw_fmt_, layer.dst_x, layer.dst_y, layer.dst_w, layer.dst_h,
                                  p, dx, dy, dw, dh);
            lumaRectToPlaneRegion(sw_fmt_, 0, 0, canvas_w_, canvas_h_, p, cx, cy, cw, ch);
            sx /= lanes; sw /= lanes; dx /= lanes; dw /= lanes; cw /= lanes;
            CUdeviceptr source = (CUdeviceptr)src->data[p], destination = (CUdeviceptr)dst->data[p];
            int source_pitch = src->linesize[p], destination_pitch = dst->linesize[p];
            void *args[] = {&source, &source_pitch, &sx, &sy, &sw, &sh,
                            &destination, &destination_pitch, &dx, &dy, &dw, &dh, &cw, &ch, &lanes};
            if (CHECK_CU(cuLaunchKernel(scale_kernel_, (dw + 31) / 32, (dh + 7) / 8, 1,
                                        32, 8, 1, 0, stream, args, nullptr)))
                throw Error("cuda_rect_overlay: scaling launch failed");
        }
    }

    std::vector<LayerSpec> mergeLayersForTick(const av::VideoFrame *metadata_source) {
        std::vector<LayerSpec> layers;
        {
            std::lock_guard<std::mutex> lock(layers_mutex_);
            layers = default_layers_;
        }
        if (!metadata_source || !metadata_source->raw() || !metadata_source->raw()->metadata)
            return layers;
        AVDictionaryEntry *e = av_dict_get(metadata_source->raw()->metadata, metadata_key_.c_str(), nullptr, 0);
        if (!e || !e->value)
            return layers;
        try {
            Parameters md = Parameters::parse(e->value);
            if (md.contains("layers") && md["layers"].is_array()) {
                const auto &arr = md["layers"];
                for (size_t i = 0; i < arr.size() && i < layers.size(); ++i) {
                    if (!arr[i].is_object())
                        continue;
                    parseLayerFromJson(arr[i], layers[i]);
                }
            } else {
                for (size_t i = 0; i < layers.size(); ++i) {
                    const std::string k = std::to_string(i);
                    if (md.contains(k) && md[k].is_object())
                        parseLayerFromJson(md[k], layers[i]);
                }
            }
        } catch (const std::exception &e) {
            logstream << "cuda_rect_overlay: ignoring bad per-frame metadata: " << e.what();
        }
        return layers;
    }

    bool hwSwFormatMatch(const av::VideoFrame &f) const {
        if (!f.raw() || !f.raw()->hw_frames_ctx || !f.raw()->hw_frames_ctx->data)
            return false;
        AVHWFramesContext *ctx = (AVHWFramesContext *)f.raw()->hw_frames_ctx->data;
        if (!ctx) return false;
        return ctx->sw_format == sw_fmt_ || isAlphaCompatible(ctx->sw_format, sw_fmt_) ||
               isRgbToNv12Convertible(ctx->sw_format, sw_fmt_);
    }

    static AVPixelFormat frameSwFormat(const av::VideoFrame &f) {
        if (!f.raw() || !f.raw()->hw_frames_ctx || !f.raw()->hw_frames_ctx->data)
            return AV_PIX_FMT_NONE;
        AVHWFramesContext *ctx = (AVHWFramesContext *)f.raw()->hw_frames_ctx->data;
        return ctx ? ctx->sw_format : AV_PIX_FMT_NONE;
    }

    void clearCanvas(av::VideoFrame &outf, const AVFrame *color_src) {
        if (!fillFrameBlack(sw_fmt_, outf.raw(), color_src))
            throw Error("cuda_rect_overlay: unsupported sw_format for canvas clear");
    }

    std::vector<DrawOp> resolveDrawOps(const std::vector<const av::VideoFrame *> &sources,
                                       const std::vector<LayerSpec> &layers) const {
        std::vector<DrawOp> ops;
        ops.reserve(std::min(sources.size(), layers.size()));
        for (size_t i = 0; i < sources.size() && i < layers.size(); ++i) {
            const av::VideoFrame *srcp = sources[i];
            if (!srcp || !srcp->raw()) {
                ops.push_back({});
                continue;
            }
            LayerSpec L = layers[i];
            if (L.dst_w > 0) {
                const avp::compositor::Rect crop{L.crop_x, L.crop_y, L.crop_w, L.crop_h};
                const avp::compositor::Rect box{L.dst_x, L.dst_y, L.dst_w, L.dst_h};
                const int ax = chromaXAlign(sw_fmt_), ay = chromaYAlign(sw_fmt_);
                auto placement = L.source_canvas_w > 0
                    ? avp::compositor::placeInCanvas(srcp->width(), srcp->height(), crop, box,
                                                     L.source_canvas_w, L.source_canvas_h, ax, ay)
                    : avp::compositor::place(srcp->width(), srcp->height(), crop, box, L.fit, ax, ay);
                if (!placement || placement->destination.x >= canvas_w_ ||
                    placement->destination.y >= canvas_h_ ||
                    int64_t(placement->destination.x) + placement->destination.w <= 0 ||
                    int64_t(placement->destination.y) + placement->destination.h <= 0) {
                    DrawOp rejected;
                    rejected.src_w = -srcp->width(); rejected.src_h = srcp->height();   // marks "rejected" in the log
                    rejected.layer = L;
                    ops.push_back(rejected);
                    continue;
                }
                    ops.push_back({});
                    continue;
                }
                const auto &p = *placement;
                L.crop_x = p.source.x; L.crop_y = p.source.y;
                L.crop_w = p.source.w; L.crop_h = p.source.h;
                L.dst_x = p.destination.x; L.dst_y = p.destination.y;
                L.dst_w = p.destination.w; L.dst_h = p.destination.h;
                ops.push_back({srcp, srcp->width(), srcp->height(), L});
                continue;
            }
            // 0 means "remaining source extent from the crop origin".
            if (L.crop_w <= 0) L.crop_w = srcp->width()  - L.crop_x;
            if (L.crop_h <= 0) L.crop_h = srcp->height() - L.crop_y;
            if (!clipRect(L.crop_x, L.crop_y, L.crop_w, L.crop_h, srcp->width(), srcp->height()) ||
                !clipRect(L.dst_x, L.dst_y, L.crop_w, L.crop_h, canvas_w_, canvas_h_)) {
                ops.push_back({});
                continue;
            }
            const int ax = chromaXAlign(sw_fmt_);
            const int ay = chromaYAlign(sw_fmt_);
            L.crop_x = alignCoord(L.crop_x, ax);
            L.crop_y = alignCoord(L.crop_y, ay);
            L.dst_x = alignCoord(L.dst_x, ax);
            L.dst_y = alignCoord(L.dst_y, ay);
            if (!clipRect(L.crop_x, L.crop_y, L.crop_w, L.crop_h, srcp->width(), srcp->height()) ||
                !clipRect(L.dst_x, L.dst_y, L.crop_w, L.crop_h, canvas_w_, canvas_h_)) {
                ops.push_back({});
                continue;
            }
            ops.push_back({srcp, srcp->width(), srcp->height(), L});
        }
        // z decides who draws on top; equal z keeps source order (stable).
        std::stable_sort(ops.begin(), ops.end(),
                         [](const DrawOp &a, const DrawOp &b) { return a.layer.z < b.layer.z; });
        return ops;
    }

    void processComposite(av::Timestamp pts, const std::vector<const av::VideoFrame *> &sources,
                          const av::VideoFrame *metadata_src) {
        ensureCudaDevice();
        CUstream stream = (CUstream)cuda_dev_->stream;

        av::VideoFrame outf;
        int r = av_hwframe_get_buffer(out_frames_ref_, outf.raw(), 0);
        if (r < 0)
            throw Error(std::string("cuda_rect_overlay: av_hwframe_get_buffer failed: ") + av::error2string(r));
        av_buffer_unref(&outf.raw()->hw_frames_ctx);
        outf.raw()->hw_frames_ctx = av_buffer_ref(out_frames_ref_);
        outf.raw()->format = AV_PIX_FMT_CUDA;
        outf.raw()->width = canvas_w_;
        outf.raw()->height = canvas_h_;
        outf.setComplete(true);

        std::vector<LayerSpec> layers = mergeLayersForTick(metadata_src);
        std::vector<DrawOp> ops = resolveDrawOps(sources, layers);
        {
            // Log the resolved layer set whenever it changes (scene switches), not per tick.
            std::ostringstream desc;
            for (size_t i = 0; i < ops.size(); ++i) {
                const DrawOp &op = ops[i];
                const LayerSpec &L = op.layer;
                if (!op.src) {
                    if (op.src_w < 0)
                        desc << " [" << i << ":REJECTED src " << -op.src_w << "x" << op.src_h << " crop " << L.crop_x
                             << "," << L.crop_y << " " << L.crop_w << "x" << L.crop_h << " box " << L.dst_x << ","
                             << L.dst_y << " " << L.dst_w << "x" << L.dst_h << "]";
                    continue;
                }
                desc << " [" << i << ":" << op.src_w << "x" << op.src_h << " crop " << L.crop_x << "," << L.crop_y
                     << " " << L.crop_w << "x" << L.crop_h << " -> " << L.dst_x << "," << L.dst_y << " "
                     << L.dst_w << "x" << L.dst_h << " z" << L.z << "]";
            }
            if (desc.str() != last_ops_desc_) {
                last_ops_desc_ = desc.str();
                logstream << "cuda_rect_overlay layers:" << last_ops_desc_;
            }
        }
        clearCanvas(outf, metadata_src ? metadata_src->raw() : nullptr);

        for (const DrawOp &op : ops) {
            const av::VideoFrame *srcp = op.src;
            if (!srcp)
                continue;
            const LayerSpec &L = op.layer;

            const bool sized = L.dst_w > 0;
            const bool full_copy = !sized || (L.dst_w == L.crop_w && L.dst_h == L.crop_h &&
                L.dst_x >= 0 && L.dst_y >= 0 && L.dst_x + L.dst_w <= canvas_w_ && L.dst_y + L.dst_h <= canvas_h_);
            const AVPixelFormat src_sw_fmt = frameSwFormat(*srcp);
            int rgb_step, r_off, g_off, b_off;
            if (src_sw_fmt != sw_fmt_ && sw_fmt_ == AV_PIX_FMT_NV12 &&
                isPackedRgb8(src_sw_fmt, rgb_step, r_off, g_off, b_off)) {
                convertRgbLayer(stream, srcp->raw(), outf.raw(), L, rgb_step, r_off, g_off, b_off);
            } else if (!full_copy) {
                scaleLayer(stream, srcp->raw(), outf.raw(), L);
            } else if (!blitLayerPlanes(stream, sw_fmt_, srcp->raw(), outf.raw(), L.crop_x, L.crop_y, L.crop_w,
                                 L.crop_h, L.dst_x, L.dst_y))
                throw Error("cuda_rect_overlay: GPU blit failed");

            // When a non-alpha source is drawn onto an alpha canvas, fill the destination
            // alpha rect with 255 (fully opaque) so the output alpha is well-defined.
            if (src_sw_fmt != AV_PIX_FMT_NONE && src_sw_fmt != sw_fmt_) {
                const int alpha_p = alphaPlaneIndex(sw_fmt_);
                int x = L.dst_x, y = L.dst_y;
                int w = sized ? L.dst_w : L.crop_w, h = sized ? L.dst_h : L.crop_h;
                if (alpha_p >= 0 && clipRect(x, y, w, h, canvas_w_, canvas_h_))
                    fillPlaneRect(sw_fmt_, outf.raw(), alpha_p, x, y, w, h, 255);
            }
        }

        if (metadata_src && metadata_src->raw()) {
            const int cpy = av_frame_copy_props(outf.raw(), metadata_src->raw());
            if (cpy < 0)
                throw Error(std::string("cuda_rect_overlay: av_frame_copy_props failed: ") + av::error2string(cpy));
        }
        outf.setPts(pts);

        if (CHECK_CU(cuStreamSynchronize(stream)))
            throw Error("cuda_rect_overlay: cuStreamSynchronize failed");

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0)
            logstream << "cuda_rect_overlay: out frame=" << frame_counter_;

        ++frame_counter_;
        this->sink_->put(std::move(outf));
    }

public:
    using NodeSingleOutput<av::VideoFrame>::NodeSingleOutput;

    CudaRectOverlay(std::unique_ptr<Sink<av::VideoFrame>> &&sink, std::shared_ptr<HWAccelDevice> hw, int cw, int ch,
                    AVPixelFormat sw_fmt, std::vector<LayerSpec> layers, std::string metadata_key, int dbg_n)
        : NodeSingleOutput<av::VideoFrame>(std::move(sink)),
          hwaccel_(std::move(hw)),
          canvas_w_(cw),
          canvas_h_(ch),
          sw_fmt_(sw_fmt),
          default_layers_(std::move(layers)),
          metadata_key_(std::move(metadata_key)),
          debug_log_every_n_(dbg_n) {
        out_frames_ref_ = av_hwframe_ctx_alloc(hwaccel_->deviceContext());
        if (!out_frames_ref_)
            throw Error("cuda_rect_overlay: av_hwframe_ctx_alloc failed");
        AVHWFramesContext *fc = (AVHWFramesContext *)out_frames_ref_->data;
        fc->format = AV_PIX_FMT_CUDA;
        fc->sw_format = sw_fmt_;
        fc->width = canvas_w_;
        fc->height = canvas_h_;
        int err = av_hwframe_ctx_init(out_frames_ref_);
        if (err < 0)
            throw Error(std::string("cuda_rect_overlay: av_hwframe_ctx_init (output) failed: ") +
                        av::error2string(err));

        input_eof_.resize(default_layers_.size());
        held_.resize(default_layers_.size());
        held_valid_.resize(default_layers_.size());
        this->auto_eof_ = false;
    }

    ~CudaRectOverlay() override { freeHwContexts(); }

    void init(EdgeManager &edges, const Parameters &params) override {
        if (params.value("scale", false)) ensureScaleKernel();
        (void)edges;
        (void)params;
        NodeSingleOutput<av::VideoFrame>::init(edges, params);
    }

    std::weak_ptr<Node> sourceNode() override {
        if (source_edges_.empty())
            return {};
        return source_edges_[0]->producer();
    }

    std::shared_ptr<EdgeBase> sourceEdge() override {
        if (source_edges_.empty())
            return {};
        return source_edges_[0];
    }

    void processClocked() {
        const int64_t now = avp::mixer::monotonicNs();
        uint32_t active = active_inputs_.load(std::memory_order_acquire);
        if (hasTimeline()) {
            const auto deadline = playout_->nextDeadline();
            const int64_t content_time = std::max(now - playout_->latencyNs(),
                deadline ? *deadline - playout_->latencyNs() : now);
            const auto value = tlGetRaw("active_inputs", av::Timestamp(content_time, {1, 1000000000}));
            if (value) active = parseBitmask(*value);
        }
        const auto generation = input_generation_.load(std::memory_order_acquire);
        if (generation != applied_generation_ || active != applied_active_mask_) {
            for (size_t i = 0; i < source_edges_.size(); ++i) {
                playout_->setActive(i, (active & (1u << i)) != 0);
                if (generation != applied_generation_) {
                    const auto from = input_valid_from_ns_.load(std::memory_order_acquire);
                    playout_->resetInput(i, from ? std::optional<int64_t>(from) : std::nullopt);
                }
            }
            applied_generation_ = generation;
            applied_active_mask_ = active;
            warmup_started_pts_ = wallclock.pts();
            sent_eof_ = false;
        }
        if (sent_eof_) return;
        if (!active) {
            this->waitForInput();
            return;
        }
        for (size_t i = 0; i < source_edges_.size(); ++i) {
            if (!(active & (1u << i))) continue;
            // Bound ingestion as well as storage; an unpaced producer must not
            // monopolize the output thread before it reaches its deadline.
            for (size_t received = 0; received < 8; ++received) {
                auto *frame = source_edges_[i]->peek();
                if (!frame) break;
                if (isEofMarker(*frame)) {
                    playout_->endInput(i);
                    source_edges_[i]->pop();
                    break;
                }
                if (frameUsable(*frame)) {
                    if (frame->raw()->format != AV_PIX_FMT_CUDA)
                        throw Error("cuda_rect_overlay: input must be AV_PIX_FMT_CUDA");
                    if (!hwSwFormatMatch(*frame))
                        throw Error("cuda_rect_overlay: input hw sw_format mismatch node sw_format");
                    playout_->push(i, *frame, rescaleTS(frame->pts(), {1, 1000000000}).timestamp());
                }
                source_edges_[i]->pop();
            }
        }
        if (playout_->finished()) {
            av::VideoFrame eof;
            eof.setPts(NOTS);
            this->sink_->put(eof);
            sent_eof_ = true;
            return;
        }
        const bool require_all = warmup_timeout_ms_ <= 0 ||
            wallclock.pts() - warmup_started_pts_ < warmup_timeout_ms_;
        const auto *decision = playout_->prepare(avp::mixer::monotonicNs(), require_all);
        if (!decision) {
            this->findSourceWithData(avp::mixer::waitMilliseconds(
                playout_->nextDeadline(), avp::mixer::monotonicNs()));
            return;
        }
        std::vector<const av::VideoFrame *> sources;
        const av::VideoFrame *metadata = nullptr;
        for (const auto &frame : decision->frames) {
            sources.push_back(frame ? &*frame : nullptr);
            if (frame) metadata = &*frame;
        }
        processComposite(av::Timestamp(decision->index,
            {frame_rate_.getDenominator(), frame_rate_.getNumerator()}), sources, metadata);
        playout_->commit();
        if (debug_log_every_n_ > 0 && frame_counter_ % debug_log_every_n_ == 0) {
            std::ostringstream stats;
            stats << "cuda_rect_overlay: frames=" << frame_counter_
                  << " missed_deadlines=" << playout_->missedDeadlines();
            for (size_t i = 0; i < sources.size(); ++i) {
                if (!(active & (1u << i))) continue;
                stats << " input" << i << "_reuse=" << playout_->stats(i).repeats
                      << " input" << i << "_discarded=" << playout_->stats(i).discarded;
            }
            logstream << stats.str();
        }
    }

    void process() override {
        if (playout_) {
            processClocked();
            return;
        }
        if (sent_eof_)
            return;

        const size_t n = this->source_edges_.size();
        if (n == 0)
            return;

        // Read active_inputs bitmask: get a representative PTS from any peeked frame
        uint32_t active_mask = active_inputs_.load(std::memory_order_relaxed);
        if (hasTimeline()) {
            for (size_t i = 0; i < n; ++i) {
                auto* p = this->source_edges_[i]->peek();
                if (p && !isEofMarker(*p) && frameUsable(*p)) {
                    auto opt = tlGetRaw("active_inputs", p->pts());
                    if (opt) active_mask = parseBitmask(*opt);
                    break;
                }
            }
        }
        auto isActive = [active_mask](size_t i) { return (active_mask & (1u << i)) != 0; };

        // "All active inputs exhausted" == EOF only if there is at least one active
        // input. With active_mask == 0 the compositor is idle (e.g. unused PVW slot):
        // falling into the EOF branch here would vacuously set sent_eof_ on the very
        // first process() call and kill the node forever.
        if (input_eof_.size() == n) {
            bool any_active = false;
            bool all_exhausted = true;
            for (size_t i = 0; i < n; ++i) {
                if (!isActive(i)) continue;
                any_active = true;
                if (!input_eof_[i]) {
                    all_exhausted = false;
                    break;
                }
            }
            if (any_active && all_exhausted) {
                av::VideoFrame eof_out;
                eof_out.setPts(NOTS);
                sent_eof_ = true;
                this->sink_->put(std::move(eof_out));
                return;
            }
        }

        // No active inputs: nothing to wait for and nothing to produce. Park the
        // thread on any source edge so setObject("active_inputs", >0) can wake it
        // once frames start flowing again.
        {
            bool any_active = false;
            for (size_t i = 0; i < n; ++i) {
                if (isActive(i)) { any_active = true; break; }
            }
            if (!any_active) {
                this->waitForInput();
                return;
            }
        }

        bool any_data = false;
        for (size_t i = 0; i < n; ++i) {
            if (!isActive(i)) continue;
            if (this->source_edges_[i]->peek() != nullptr) {
                any_data = true;
                break;
            }
        }
        if (!any_data) {
            this->waitForInput();
            return;
        }

        bool all_peek_eof = true;
        for (size_t i = 0; i < n; ++i) {
            if (!isActive(i)) continue;
            av::VideoFrame *p = this->source_edges_[i]->peek();
            if (p == nullptr || !isEofMarker(*p))
                all_peek_eof = false;
        }
        if (all_peek_eof) {
            for (size_t i = 0; i < n; ++i) {
                if (!isActive(i)) continue;
                this->source_edges_[i]->pop();
                if (i < input_eof_.size())
                    input_eof_[i] = true;
            }
            av::VideoFrame eof_out;
            eof_out.setPts(NOTS);
            sent_eof_ = true;
            this->sink_->put(std::move(eof_out));
            return;
        }

        av::Timestamp min_ts = NOTS;
        for (size_t i = 0; i < n; ++i) {
            if (!isActive(i)) continue;
            if (input_eof_[i])
                continue;
            av::VideoFrame *p = this->source_edges_[i]->peek();
            if (!p || isEofMarker(*p))
                continue;
            if (!frameUsable(*p))
                continue;
            if (min_ts.isNoPts() || p->pts() < min_ts)
                min_ts = p->pts();
        }
        if (min_ts.isNoPts()) {
            this->waitForInput();
            return;
        }

        for (size_t i = 0; i < n; ++i) {
            if (!isActive(i)) continue;
            if (input_eof_[i])
                continue;
            while (true) {
                av::VideoFrame *p = this->source_edges_[i]->peek();
                if (!p)
                    break;
                if (isEofMarker(*p)) {
                    this->source_edges_[i]->pop();
                    input_eof_[i] = true;
                    break;
                }
                if (!frameUsable(*p))
                    break;
                if (p->pts() < min_ts)
                    this->source_edges_[i]->pop();
                else
                    break;
            }
        }

        std::vector<const av::VideoFrame *> src_for_layer(n, nullptr);
        const av::VideoFrame *meta_src = nullptr;
        bool need_wait = false;

        for (size_t i = 0; i < n; ++i) {
            if (!isActive(i)) continue;
            if (input_eof_[i]) {
                if (held_valid_[i] && !held_[i].isNull())
                    src_for_layer[i] = &held_[i];
                continue;
            }
            av::VideoFrame *p = this->source_edges_[i]->peek();
            if (!p) {
                need_wait = true;
                break;
            }
            if (isEofMarker(*p))
                continue;
            if (!frameUsable(*p)) {
                need_wait = true;
                break;
            }
            if (p->pts() == min_ts)
                src_for_layer[i] = p;
            else if (p->pts() > min_ts) {
                if (held_valid_[i] && !held_[i].isNull())
                    src_for_layer[i] = &held_[i];
            } else
                need_wait = true;
        }
        if (need_wait) {
            this->waitForInput();
            return;
        }

        for (size_t i = 0; i < n; ++i) {
            if (!isActive(i)) continue;
            av::VideoFrame *p = this->source_edges_[i]->peek();
            if (p && !input_eof_[i] && frameUsable(*p) && p->pts() == min_ts)
                meta_src = p;
        }
        if (!meta_src) {
            for (size_t i = 0; i < n; ++i) {
                if (src_for_layer[i]) {
                    meta_src = src_for_layer[i];
                    break;
                }
            }
        }

        for (size_t i = 0; i < n; ++i) {
            if (!isActive(i)) continue;
            av::VideoFrame *p = this->source_edges_[i]->peek();
            if (!p || input_eof_[i] || !frameUsable(*p) || p->pts() != min_ts)
                continue;
            if (p->raw()->format != AV_PIX_FMT_CUDA)
                throw Error("cuda_rect_overlay: input must be AV_PIX_FMT_CUDA");
            if (!hwSwFormatMatch(*p))
                throw Error("cuda_rect_overlay: input hw sw_format mismatch node sw_format");
            av::VideoFrame consumed = *p;
            this->source_edges_[i]->pop();
            held_[i] = std::move(consumed);
            held_valid_[i] = true;
            src_for_layer[i] = &held_[i];
        }

        // During slot warmup, do not emit a partial black composite just because
        // one input has produced an earlier PTS than the others.  Consume fresh
        // frames into held_ until every active layer has a post-activation frame.
        // Once warmup_timeout_ms_ elapses without all layers caught up, fall through
        // to render with whatever we have (missing layers stay black) so a stuck
        // input can never park the compositor indefinitely.
        bool any_missing = false;
        for (size_t i = 0; i < n; ++i) {
            if (!isActive(i) || input_eof_[i])
                continue;
            if (!src_for_layer[i]) {
                any_missing = true;
                break;
            }
        }
        if (any_missing) {
            const int64_t now_ms = wallclock.pts();
            if (warmup_timeout_ms_ <= 0) {
                this->waitForInput();
                return;
            }
            if (warmup_started_pts_ < 0) {
                warmup_started_pts_ = now_ms;
                this->waitForInput();
                return;
            }
            if (now_ms - warmup_started_pts_ < warmup_timeout_ms_) {
                this->waitForInput();
                return;
            }
            logstream << "cuda_rect_overlay: warmup timeout after "
                      << (now_ms - warmup_started_pts_) << "ms; emitting partial composite";
        }
        warmup_started_pts_ = -1;

        processComposite(min_ts, src_for_layer, meta_src);
    }

    void resetInput() override {
        if (!playout_) return;
        input_valid_from_ns_.store(avp::mixer::monotonicNs(), std::memory_order_release);
        input_generation_.fetch_add(1, std::memory_order_release);
        for (auto &edge : source_edges_) edge->producedEvent().signal();
    }

    void setObject(const std::string key, const Parameters& value) override {
        if (key == "active_inputs") {
            const uint32_t new_mask = parseBitmask(value);
            active_inputs_.store(new_mask, std::memory_order_relaxed);
            if (!playout_) {
                sent_eof_ = false;
                std::fill(input_eof_.begin(), input_eof_.end(), false);
                std::fill(held_valid_.begin(), held_valid_.end(), false);
                warmup_started_pts_ = -1;
            }
            for (auto& edge : this->source_edges_) {
                edge->producedEvent().signal();
            }
        } else if (key == "layers") {
            auto new_layers = parseLayersArray(value);
            std::lock_guard<std::mutex> lock(layers_mutex_);
            default_layers_ = std::move(new_layers);
        }
    }

    av::Rational frameRate() override {
        if (playout_) return frame_rate_;
        if (!source_edges_.empty()) {
            auto source = source_edges_.front()->findNodeUp<IFrameRateSource>();
            if (source) return source->frameRate();
        }
        return {0, 1};
    }

    int width() override { return canvas_w_; }
    int height() override { return canvas_h_; }
    av::PixelFormat pixelFormat() override { return av::PixelFormat(AV_PIX_FMT_CUDA); }
    av::PixelFormat realPixelFormat() override { return av::PixelFormat(sw_fmt_); }

    static std::shared_ptr<CudaRectOverlay> create(NodeCreationInfo &nci);
};

std::shared_ptr<CudaRectOverlay> CudaRectOverlay::create(NodeCreationInfo &nci) {
    EdgeManager &edges = nci.edges;
    const Parameters &params = nci.params;
    auto src_names = jsonToStringList(params["src"]);
    if (src_names.empty())
        throw Error("cuda_rect_overlay: at least one input required in src");
    std::vector<LayerSpec> layers = parseLayersParam(params);
    if (layers.size() != src_names.size())
        throw Error("cuda_rect_overlay: layers array length must match src count");

    if (!params.contains("hwaccel"))
        throw Error("cuda_rect_overlay: hwaccel parameter required");
    auto hw = InstanceSharedObjects<HWAccelDevice>::get(nci.instance, params["hwaccel"]);
    if (!hw)
        throw Error("cuda_rect_overlay: failed to resolve hwaccel");

    const int cw = params.at("width").get<int>();
    const int ch = params.at("height").get<int>();
    if (cw <= 0 || ch <= 0)
        throw Error("cuda_rect_overlay: width and height must be positive");
    const std::string sw_name = params.value("sw_format", std::string("nv12"));
    const AVPixelFormat sw_fmt = av_get_pix_fmt(sw_name.c_str());
    if (sw_fmt == AV_PIX_FMT_NONE)
        throw Error("cuda_rect_overlay: unknown sw_format");

    auto out_edge = edges.find<av::VideoFrame>(params["dst"]);

    const std::string mdkey = params.value("metadata_key", std::string("rect_overlay_v1"));
    const int dbg = params.value("debug_log_every_n", 0);

    auto node = std::make_shared<CudaRectOverlay>(
        make_unique<EdgeSink<av::VideoFrame>>(out_edge), std::move(hw), cw, ch, sw_fmt, std::move(layers), mdkey,
        dbg);
    node->createSourcesFromParameters(edges, params);
    out_edge->setProducer(node);
    node->initTimeline(nci);
    if (params.count("active_inputs"))
        node->active_inputs_.store(parseBitmask(params["active_inputs"]), std::memory_order_relaxed);
    node->warmup_timeout_ms_ = params.value("warmup_timeout_ms", (int64_t)0);
    if (params.contains("fps")) {
        node->frame_rate_ = parseRatio(params.at("fps"));
        std::optional<double> latency_ms;
        if (params.contains("latency_ms")) latency_ms = params.at("latency_ms").get<double>();
        node->playout_ = std::make_unique<avp::mixer::Playout<av::VideoFrame>>(
            src_names.size(), avp::mixer::FrameRate(node->frame_rate_.getNumerator(),
                node->frame_rate_.getDenominator()), latency_ms, avp::mixer::TimestampMode::Presentation);
        node->input_generation_.store(1);
        logstream << "cuda_rect_overlay: latency_ms=" << node->playout_->latencyNs() / 1000000.0;
    }

    return node;
}

DECLNODE(cuda_rect_overlay, CudaRectOverlay);
