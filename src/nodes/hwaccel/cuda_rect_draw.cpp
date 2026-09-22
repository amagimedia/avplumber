#include "cuda_rect_draw.hpp"
#include "cuda_rect_texture.h"
#include "../../mixer/primitives/compositor_color.hpp"
#ifdef HAVE_CUDA_RECT_SCALE
#include "../../../objs/src/nodes/hwaccel/cuda_rect_scale.ptx.h"
#endif

extern "C" {
#include <libavutil/pixdesc.h>
}

#include <string>

namespace avp::mixer {

int checkCu(CUresult err, const char *func) {
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

namespace {

/// Transfer id the RGB conversion takes: 0 = HLG, 1 = PQ, 2 = SDR (BT.709).
int kernelTransfer(AVColorTransferCharacteristic trc) {
    return trc == AVCOL_TRC_ARIB_STD_B67 ? 0 : trc == AVCOL_TRC_SMPTE2084 ? 1 : 2;
}

/// Samples per lane group of plane 0: 1 for luma, the packed step for RGB canvases.
int plane0Lanes(AVPixelFormat fmt) {
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(fmt);
    int lanes = 1;
    for (int c = 0; d && c < d->nb_components; ++c)
        if (d->comp[c].plane == 0) lanes = std::max(lanes, d->comp[c].step / sampleBytes(fmt));
    return lanes;
}

} // namespace

AVPixelFormat CudaRectDraw::frameSwFormat(const av::VideoFrame &f) {
    if (!f.raw() || !f.raw()->hw_frames_ctx || !f.raw()->hw_frames_ctx->data)
        return AV_PIX_FMT_NONE;
    AVHWFramesContext *ctx = (AVHWFramesContext *)f.raw()->hw_frames_ctx->data;
    return ctx ? ctx->sw_format : AV_PIX_FMT_NONE;
}

bool CudaRectDraw::canvasSupported(AVPixelFormat sw_fmt) {
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(sw_fmt);
    if (!d || (d->flags & AV_PIX_FMT_FLAG_BITSTREAM))
        return false;
    const int planes = av_pix_fmt_count_planes(sw_fmt);
    if (d->flags & AV_PIX_FMT_FLAG_RGB)
        return planes == 1 && sampleBytes(sw_fmt) == 1 && plane0Lanes(sw_fmt) == 4;
    return planes == 2 && d->nb_components == 3 && !(d->flags & AV_PIX_FMT_FLAG_ALPHA);
}

void CudaRectDraw::ensureDevice() {
    if (!hwaccel_ || !hwaccel_->deviceContext() || !hwaccel_->deviceContext()->data)
        throw Error("cuda_rect_overlay: invalid hwaccel device");
    AVHWDeviceContext *devctx = (AVHWDeviceContext *)hwaccel_->deviceContext()->data;
    cuda_dev_ = (AVCUDADeviceContext *)devctx->hwctx;
    if (!cuda_dev_ || !cuda_dev_->cuda_ctx)
        throw Error("cuda_rect_overlay: CUDA hwctx missing");
    if (AVP_CHECK_CU(cuCtxSetCurrent(cuda_dev_->cuda_ctx)))
        throw Error("cuda_rect_overlay: cuCtxSetCurrent failed");
}

void CudaRectDraw::ensureKernels() {
#ifdef HAVE_CUDA_RECT_SCALE
    if (composite_kernel_) return;
    ensureDevice();
    const std::string image(avpl_rect_scale_ptx, avpl_rect_scale_ptx + avpl_rect_scale_ptx_len);
    if (AVP_CHECK_CU(cuModuleLoadDataEx(&module_, image.c_str(), 0, nullptr, nullptr)) ||
        AVP_CHECK_CU(cuModuleGetFunction(&composite_kernel_, module_, "composite_planes")) ||
        AVP_CHECK_CU(cuModuleGetFunction(&composite_yuv_kernel_, module_, "composite_planes_yuv")) ||
        AVP_CHECK_CU(cuModuleGetFunction(&composite_packed_kernel_, module_, "composite_planes_packed4")))
        throw Error("cuda_rect_overlay: cannot load the composite kernel");
    const size_t bytes = sizeof(AvpRectLayer) * AVP_RECT_MAX_LAYERS;
    if (AVP_CHECK_CU(cuMemHostAlloc((void **)&table_host_, bytes, 0)) ||
        AVP_CHECK_CU(cuMemAlloc(&table_device_, bytes)))
        throw Error("cuda_rect_overlay: cannot allocate the rect table");
#else
    throw Error("cuda_rect_overlay: compositing requires a build with HAVE_NVCC=1");
#endif
}

void CudaRectDraw::unload() {
    if (module_) {
        cuCtxSetCurrent(cuda_dev_->cuda_ctx);
        if (table_device_) { AVP_CHECK_CU(cuMemFree(table_device_)); table_device_ = 0; }
        if (table_host_) { AVP_CHECK_CU(cuMemFreeHost(table_host_)); table_host_ = nullptr; }
        AVP_CHECK_CU(cuModuleUnload(module_));
        module_ = nullptr;
        composite_kernel_ = composite_yuv_kernel_ = composite_packed_kernel_ = nullptr;
    }
}

void CudaRectDraw::validateSourceColor(const av::VideoFrame &src, bool packed_rgb, const AVFrame *canvas) const {
    if (canvas_.transfer == AVCOL_TRC_UNSPECIFIED)
        return;
    const AVFrame *frame = src.raw();
    const bool valid = packed_rgb
        ? isSdrGraphicColor(*frame)
        : frame->color_trc == canvas->color_trc &&
          frame->color_primaries == canvas->color_primaries &&
          frame->colorspace == canvas->colorspace && frame->color_range == AVCOL_RANGE_MPEG;
    if (!valid)
        throw Error("cuda_rect_overlay: missing or mismatched source color metadata; "
                    "declare source color and normalize to the canvas before compositing");
}

// One table entry from a resolved op. Per-plane geometry goes through lumaRectToPlaneRegion in
// bytes and then lane groups.
void CudaRectDraw::fillTableEntry(const DrawOp &op, const AVFrame *canvas, AvpRectLayer &out) const {
    const AVPixelFormat sw_fmt = canvas_.sw_fmt;
    const LayerSpec &L = op.layer;
    const AVFrame *src = op.src->raw();
    const AVPixelFormat src_sw_fmt = frameSwFormat(*op.src);
    int rgb_step, r_off, g_off, b_off;
    const bool packed_rgb = isPackedRgb8(src_sw_fmt, rgb_step, r_off, g_off, b_off);
    validateSourceColor(*op.src, packed_rgb, canvas);
    const bool sized = L.dst_w > 0;
    const int dstw = sized ? L.dst_w : L.crop_w, dsth = sized ? L.dst_h : L.crop_h;
    const AVPixFmtDescriptor *dd = av_pix_fmt_desc_get(sw_fmt);
    const int dst_bytes = sampleBytes(sw_fmt);
    const int planes = av_pix_fmt_count_planes(sw_fmt);
    out = AvpRectLayer{};

    if (packed_rgb && src_sw_fmt != sw_fmt && isRgbToYuvConvertible(src_sw_fmt, sw_fmt)) {
        const int a_off = packedAlphaOffset(src_sw_fmt);
        const bool blend = L.blend && a_off >= 0;
        out.step = rgb_step; out.r_off = r_off; out.g_off = g_off; out.b_off = b_off; out.a_off = a_off;
#if LIBAVUTIL_VERSION_MAJOR >= 60
        out.premultiplied = src->alpha_mode == AVALPHA_MODE_PREMULTIPLIED;
#endif
        if (const TextureFrameDesc *tex = textureFrameDesc(src)) {
            // Zero-copy DMA-BUF import: sample the mapped array through its texture object.
            if (rgb_step != 4 || tex->width != src->width || tex->height != src->height)
                throw Error("cuda_rect_overlay: texture-backed source must be 4-byte packed RGB at frame size");
            out.kind = blend ? AVP_RECT_KIND_RGBA_TEX : AVP_RECT_KIND_RGB_TEX;
            out.src[0] = tex->tex;
        } else {
            out.kind = blend ? AVP_RECT_KIND_RGBA : AVP_RECT_KIND_RGB;
            out.src[0] = (unsigned long long)(uintptr_t)src->data[0];
            out.src_pitch[0] = src->linesize[0];
        }
        out.sx[0] = L.crop_x; out.sy[0] = L.crop_y; out.sw[0] = L.crop_w; out.sh[0] = L.crop_h;
        out.dx[0] = L.dst_x; out.dy[0] = L.dst_y; out.dw[0] = dstw; out.dh[0] = dsth;
        // Chroma sites owned by this layer: the blocks whose (aligned) luma origin lies in the rect.
        const int bw = 1 << dd->log2_chroma_w, bh = 1 << dd->log2_chroma_h;
        out.dx[1] = L.dst_x >> dd->log2_chroma_w;
        out.dy[1] = L.dst_y >> dd->log2_chroma_h;
        out.dw[1] = (L.dst_x + dstw + bw - 1) / bw - out.dx[1];
        out.dh[1] = (L.dst_y + dsth + bh - 1) / bh - out.dy[1];
        if (L.dst_x + dstw <= 0) out.dw[1] = 0;
        if (L.dst_y + dsth <= 0) out.dh[1] = 0;
        return;
    }

    const bool promote = src_sw_fmt != sw_fmt && isYuvPromoteConvertible(src_sw_fmt, sw_fmt);
    if (!promote && src_sw_fmt != sw_fmt)
        throw Error("cuda_rect_overlay: source format " + std::string(av_get_pix_fmt_name(src_sw_fmt)) +
                    " cannot be drawn onto a " + av_get_pix_fmt_name(sw_fmt) + " canvas");
    const AVPixelFormat geometry_fmt = promote ? src_sw_fmt : sw_fmt;
    out.kind = promote ? AVP_RECT_KIND_PROMOTE : AVP_RECT_KIND_YUV;
    out.src_bytes = sampleBytes(geometry_fmt);
    out.src_shift = storageShift(geometry_fmt);
    const AVPixFmtDescriptor *sd = av_pix_fmt_desc_get(geometry_fmt);
    out.mul = promote ? float(1 << (dd->comp[0].depth - sd->comp[0].depth)) : 1.f;
    for (int p = 0; p < planes; ++p) {
        if (!src->data[p])
            throw Error("cuda_rect_overlay: source frame lacks plane " + std::to_string(p));
        const int lanes = p ? 2 : plane0Lanes(sw_fmt);
        int sx, sy, sw, sh, dx, dy, dw, dh;
        lumaRectToPlaneRegion(geometry_fmt, L.crop_x, L.crop_y, L.crop_w, L.crop_h, p, sx, sy, sw, sh);
        lumaRectToPlaneRegion(sw_fmt, L.dst_x, L.dst_y, dstw, dsth, p, dx, dy, dw, dh);
        sx /= lanes * out.src_bytes; sw /= lanes * out.src_bytes;
        dx /= lanes * dst_bytes; dw /= lanes * dst_bytes;
        out.src[p] = (unsigned long long)(uintptr_t)src->data[p];
        out.src_pitch[p] = src->linesize[p];
        out.sx[p] = sx; out.sy[p] = sy; out.sw[p] = sw; out.sh[p] = sh;
        out.dx[p] = dx; out.dy[p] = dy; out.dw[p] = dw; out.dh[p] = dh;
    }
}

void CudaRectDraw::draw(CUstream stream, const std::vector<DrawOp> &ops, AVFrame *canvas, const AVFrame *color_src) {
    ensureKernels();
    const AVPixelFormat sw_fmt = canvas_.sw_fmt;
    int n = 0;
    bool any_rgb = false;
    for (const DrawOp &op : ops) {
        if (!op.src || !op.src->raw()) continue;
        if (n >= AVP_RECT_MAX_LAYERS)
            throw Error("cuda_rect_overlay: more than " + std::to_string(AVP_RECT_MAX_LAYERS) + " layers in one frame");
        fillTableEntry(op, canvas, table_host_[n]);
        any_rgb = any_rgb || table_host_[n].kind >= AVP_RECT_KIND_RGB;
        ++n;
    }
    const int planes = av_pix_fmt_count_planes(sw_fmt);
    uint16_t clear0 = 0, clear1 = 0;
    if (!planeClearValue(sw_fmt, color_src, 0, clear0) || (planes > 1 && !planeClearValue(sw_fmt, color_src, 1, clear1)))
        throw Error("cuda_rect_overlay: unsupported sw_format for canvas clear");
    if (n > 0 && AVP_CHECK_CU(cuMemcpyHtoDAsync(table_device_, table_host_, sizeof(AvpRectLayer) * n, stream)))
        throw Error("cuda_rect_overlay: rect table upload failed");

    const AVPixFmtDescriptor *cd = av_pix_fmt_desc_get(sw_fmt);
    int dst_sb = sampleBytes(sw_fmt), dst_shift = storageShift(sw_fmt);
    int canvas_w = canvas_.width, canvas_h = canvas_.height;   // plane 0 in lane groups
    int chroma_w = 0, chroma_h = 0;
    if (planes > 1) {
        int cx, cy;
        lumaRectToPlaneRegion(sw_fmt, 0, 0, canvas_.width, canvas_.height, 1, cx, cy, chroma_w, chroma_h);
        chroma_w /= 2 * dst_sb;
    }
    int dst_scale = 1 << (cd->comp[0].depth - 8), sub_x = cd->log2_chroma_w, sub_y = cd->log2_chroma_h;
    int clear0_i = clear0, clear1_i = clear1;
    int transfer = kernelTransfer(canvas_.transfer);
    float sdr_white = canvas_.sdr_white, hdr_peak = canvas_.hdr_peak;
    CUdeviceptr table = table_device_, plane0 = (CUdeviceptr)canvas->data[0];
    CUdeviceptr plane1 = planes > 1 ? (CUdeviceptr)canvas->data[1] : 0;
    int pitch0 = canvas->linesize[0], pitch1 = planes > 1 ? canvas->linesize[1] : 0;
    void *args[] = {&table, &n, &plane0, &pitch0, &plane1, &pitch1,
                    &canvas_w, &canvas_h, &chroma_w, &chroma_h,
                    &dst_sb, &dst_shift, &dst_scale, &sub_x, &sub_y,
                    &clear0_i, &clear1_i, &transfer, &sdr_white, &hdr_peak};
    const CUfunction kernel = planes == 1 ? composite_packed_kernel_ : any_rgb ? composite_kernel_ : composite_yuv_kernel_;
    // 128x8 luma tiles per block; gridDim.z spans the planes.
    if (AVP_CHECK_CU(cuLaunchKernel(kernel,
                                    (canvas_w + 32 * AVP_RECT_PX - 1) / (32 * AVP_RECT_PX), (canvas_h + 7) / 8,
                                    planes, 32, 8, 1, 0, stream, args, nullptr)))
        throw Error("cuda_rect_overlay: composite launch failed");
}

}
