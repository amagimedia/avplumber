#include "cuda_rect_draw.hpp"
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

/// Transfer id the rgb_to_yuv kernels take: 0 = HLG, 1 = PQ, 2 = SDR (BT.709).
int kernelTransfer(AVColorTransferCharacteristic trc) {
    return trc == AVCOL_TRC_ARIB_STD_B67 ? 0 : trc == AVCOL_TRC_SMPTE2084 ? 1 : 2;
}

bool memcpy2d_async(CUstream stream, CUdeviceptr dst, size_t dst_pitch, size_t dst_x_off_bytes,
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
    return AVP_CHECK_CU(cuMemcpy2DAsync(&cpy, stream)) == 0;
}

bool blitLayerPlanes(CUstream stream, AVPixelFormat sw_fmt, const AVFrame *src, const AVFrame *dst,
                     int src_luma_x, int src_luma_y, int lw, int lh, int dst_luma_x, int dst_luma_y) {
    const int planes = av_pix_fmt_count_planes(sw_fmt);
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

// Set a rectangular region of one plane to a constant logical sample value.
// Word-stored formats take the value below the storage shift (64, not 64 << 6)
// and get a 16-bit memset; padding bits stay zero.
void fillPlaneRect(AVPixelFormat fmt, AVFrame *f, int plane,
                   int lx, int ly, int lw, int lh, uint16_t value) {
    if (!f->data[plane] || f->linesize[plane] <= 0) return;
    int bx, by, bw, bh;
    lumaRectToPlaneRegion(fmt, lx, ly, lw, lh, plane, bx, by, bw, bh);
    if (bw <= 0 || bh <= 0) return;
    const size_t pitch = (size_t)f->linesize[plane];
    CUdeviceptr base = (CUdeviceptr)(uintptr_t)f->data[plane] + (CUdeviceptr)((size_t)by * pitch + (size_t)bx);
    if (sampleBytes(fmt) == 2)
        AVP_CHECK_CU(cuMemsetD2D16(base, (unsigned int)pitch,
                                   (unsigned short)(value << storageShift(fmt)), (size_t)bw / 2, (size_t)bh));
    else
        AVP_CHECK_CU(cuMemsetD2D8(base, (unsigned int)pitch, (unsigned char)value, (size_t)bw, (size_t)bh));
}

bool fillFrameBlack(AVPixelFormat fmt, AVFrame *f, const AVFrame *color_src) {
    const int planes = av_pix_fmt_count_planes(fmt);
    for (int p = 0; p < planes && p < AV_NUM_DATA_POINTERS; ++p) {
        if (!f->data[p])
            continue;
        uint16_t value = 0;
        if (!planeClearValue(fmt, color_src, p, value))
            return false;
        fillPlaneRect(fmt, f, p, 0, 0, f->width, f->height, value);
    }
    return true;
}

} // namespace

AVPixelFormat CudaRectDraw::frameSwFormat(const av::VideoFrame &f) {
    if (!f.raw() || !f.raw()->hw_frames_ctx || !f.raw()->hw_frames_ctx->data)
        return AV_PIX_FMT_NONE;
    AVHWFramesContext *ctx = (AVHWFramesContext *)f.raw()->hw_frames_ctx->data;
    return ctx ? ctx->sw_format : AV_PIX_FMT_NONE;
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
    if (scale_kernel_) return;
    ensureDevice();
    const std::string image(avpl_rect_scale_ptx, avpl_rect_scale_ptx + avpl_rect_scale_ptx_len);
    if (AVP_CHECK_CU(cuModuleLoadDataEx(&scale_module_, image.c_str(), 0, nullptr, nullptr)) ||
        AVP_CHECK_CU(cuModuleGetFunction(&scale_kernel_, scale_module_, "scale_plane")) ||
        AVP_CHECK_CU(cuModuleGetFunction(&convert_kernel_, scale_module_, "convert_scale_plane")) ||
        AVP_CHECK_CU(cuModuleGetFunction(&rgb_kernel_, scale_module_, "rgb_to_yuv")) ||
        AVP_CHECK_CU(cuModuleGetFunction(&rgba_kernel_, scale_module_, "rgba_over_yuv")))
        throw Error("cuda_rect_overlay: cannot load scaling kernel");
#endif
}

void CudaRectDraw::unload() {
    if (scale_module_) {
        cuCtxSetCurrent(cuda_dev_->cuda_ctx);
        AVP_CHECK_CU(cuModuleUnload(scale_module_));
        scale_module_ = nullptr;
    }
}

void CudaRectDraw::clearCanvas(AVFrame *canvas, const AVFrame *color_src) {
    if (!fillFrameBlack(canvas_.sw_fmt, canvas, color_src))
        throw Error("cuda_rect_overlay: unsupported sw_format for canvas clear");
}

/// Draw a packed RGB source onto the semiplanar YUV canvas: one fused scale
/// + color conversion pass, alpha-blended over what is already there when
/// the layer asks for it and the source carries alpha.
void CudaRectDraw::convertRgbLayer(CUstream stream, const AVFrame *src, AVFrame *dst, const LayerSpec &layer,
                                   int step, int r_off, int g_off, int b_off, int a_off) {
    ensureKernels();
    const bool blend = layer.blend && a_off >= 0;
    if (!rgb_kernel_ || (blend && !rgba_kernel_))
        throw Error("cuda_rect_overlay: RGB conversion kernel unavailable");
    int sx = layer.crop_x, sy = layer.crop_y, sw = layer.crop_w, sh = layer.crop_h;
    int dx = layer.dst_x, dy = layer.dst_y;
    int dw = layer.dst_w > 0 ? layer.dst_w : layer.crop_w, dh = layer.dst_w > 0 ? layer.dst_h : layer.crop_h;
    int cw = canvas_.width, ch = canvas_.height;
    const AVPixFmtDescriptor *cd = av_pix_fmt_desc_get(canvas_.sw_fmt);
    int dst_sb = sampleBytes(canvas_.sw_fmt), dst_shift = storageShift(canvas_.sw_fmt);
    int dst_scale = 1 << (cd->comp[0].depth - 8), sub_x = cd->log2_chroma_w, sub_y = cd->log2_chroma_h;
    CUdeviceptr source = (CUdeviceptr)src->data[0], luma = (CUdeviceptr)dst->data[0],
                chroma = (CUdeviceptr)dst->data[1];
    int source_pitch = src->linesize[0], luma_pitch = dst->linesize[0], chroma_pitch = dst->linesize[1];
    int transfer = kernelTransfer(canvas_.transfer);
    int premultiplied = 0;
#if LIBAVUTIL_VERSION_MAJOR >= 60
    premultiplied = src->alpha_mode == AVALPHA_MODE_PREMULTIPLIED;
#endif
    float sdr_white = canvas_.sdr_white, hdr_peak = canvas_.hdr_peak;
    void *opaque_args[] = {&source, &source_pitch, &sx, &sy, &sw, &sh, &step, &r_off, &g_off, &b_off,
                           &luma, &luma_pitch, &chroma, &chroma_pitch, &dx, &dy, &dw, &dh, &cw, &ch,
                           &dst_sb, &dst_shift, &dst_scale, &sub_x, &sub_y, &transfer, &sdr_white, &hdr_peak};
    void *blend_args[] = {&source, &source_pitch, &sx, &sy, &sw, &sh, &step, &r_off, &g_off, &b_off, &a_off,
                          &luma, &luma_pitch, &chroma, &chroma_pitch, &dx, &dy, &dw, &dh, &cw, &ch,
                          &dst_sb, &dst_shift, &dst_scale, &sub_x, &sub_y, &transfer, &sdr_white, &hdr_peak,
                          &premultiplied};
    const int bw = 1 << sub_x, bh = 1 << sub_y;
    const int blocks_x = (dw + bw - 1) / bw, blocks_y = (dh + bh - 1) / bh;
    if (AVP_CHECK_CU(cuLaunchKernel(blend ? rgba_kernel_ : rgb_kernel_, (blocks_x + 31) / 32, (blocks_y + 7) / 8, 1,
                                    32, 8, 1, 0, stream, blend ? blend_args : opaque_args, nullptr)))
        throw Error("cuda_rect_overlay: RGB conversion launch failed");
}

void CudaRectDraw::scaleLayer(CUstream stream, const AVFrame *src, AVFrame *dst, const LayerSpec &layer) {
    const AVPixelFormat sw_fmt = canvas_.sw_fmt;
    const auto *desc = av_pix_fmt_desc_get(sw_fmt);
    int sample_bytes = sampleBytes(sw_fmt), shift = storageShift(sw_fmt);
    ensureKernels();
    if (!scale_kernel_) throw Error("cuda_rect_overlay: scaling kernel unavailable");
    for (int p = 0; p < av_pix_fmt_count_planes(sw_fmt); ++p) {
        if (!src->data[p]) continue; // Opaque input on an alpha canvas.
        int lanes = 1;
        for (int c = 0; c < desc->nb_components; ++c)
            if (desc->comp[c].plane == p) lanes = std::max(lanes, desc->comp[c].step / sample_bytes);
        int sx, sy, sw, sh, dx, dy, dw, dh, cx, cy, cw, ch;
        lumaRectToPlaneRegion(sw_fmt, layer.crop_x, layer.crop_y, layer.crop_w, layer.crop_h,
                              p, sx, sy, sw, sh);
        lumaRectToPlaneRegion(sw_fmt, layer.dst_x, layer.dst_y, layer.dst_w, layer.dst_h,
                              p, dx, dy, dw, dh);
        lumaRectToPlaneRegion(sw_fmt, 0, 0, canvas_.width, canvas_.height, p, cx, cy, cw, ch);
        // Region byte extents -> lane-group coordinates for the kernel.
        const int group = lanes * sample_bytes;
        sx /= group; sw /= group; dx /= group; dw /= group; cw /= group;
        CUdeviceptr source = (CUdeviceptr)src->data[p], destination = (CUdeviceptr)dst->data[p];
        int source_pitch = src->linesize[p], destination_pitch = dst->linesize[p];
        void *args[] = {&source, &source_pitch, &sx, &sy, &sw, &sh,
                        &destination, &destination_pitch, &dx, &dy, &dw, &dh, &cw, &ch, &lanes,
                        &sample_bytes, &shift};
        if (AVP_CHECK_CU(cuLaunchKernel(scale_kernel_, (dw + 31) / 32, (dh + 7) / 8, 1,
                                        32, 8, 1, 0, stream, args, nullptr)))
            throw Error("cuda_rect_overlay: scaling launch failed");
    }
}

// Draw a lower-depth semiplanar source onto the deeper canvas: bilinear
// scale each plane from the source's geometry to the canvas geometry while
// promoting codes by 2^(dst_depth-src_depth). Reads the source's own plane
// layout, writes the canvas layout, so NV12(4:2:0)->P210(4:2:2) works.
void CudaRectDraw::convertLayer(CUstream stream, AVPixelFormat src_fmt, const AVFrame *src, AVFrame *dst,
                                const LayerSpec &layer) {
    const AVPixelFormat sw_fmt = canvas_.sw_fmt;
    ensureKernels();
    if (!convert_kernel_) throw Error("cuda_rect_overlay: convert kernel unavailable");
    const AVPixFmtDescriptor *sd = av_pix_fmt_desc_get(src_fmt);
    const AVPixFmtDescriptor *dd = av_pix_fmt_desc_get(sw_fmt);
    int src_bytes = sampleBytes(src_fmt), src_shift = storageShift(src_fmt);
    int dst_bytes = sampleBytes(sw_fmt), dst_shift = storageShift(sw_fmt);
    float mul = float(1 << (dd->comp[0].depth - sd->comp[0].depth));
    int dstw = layer.dst_w > 0 ? layer.dst_w : layer.crop_w;
    int dsth = layer.dst_w > 0 ? layer.dst_h : layer.crop_h;
    for (int p = 0; p < av_pix_fmt_count_planes(sw_fmt); ++p) {
        if (!src->data[p]) continue;
        int lanes = 1;
        for (int c = 0; c < dd->nb_components; ++c)
            if (dd->comp[c].plane == p) lanes = std::max(lanes, dd->comp[c].step / dst_bytes);
        int sx, sy, sw, sh, dx, dy, dw, dh, cx, cy, cw, ch;
        lumaRectToPlaneRegion(src_fmt, layer.crop_x, layer.crop_y, layer.crop_w, layer.crop_h,
                              p, sx, sy, sw, sh);
        lumaRectToPlaneRegion(sw_fmt, layer.dst_x, layer.dst_y, dstw, dsth, p, dx, dy, dw, dh);
        lumaRectToPlaneRegion(sw_fmt, 0, 0, canvas_.width, canvas_.height, p, cx, cy, cw, ch);
        sx /= lanes * src_bytes; sw /= lanes * src_bytes;
        dx /= lanes * dst_bytes; dw /= lanes * dst_bytes; cw /= lanes * dst_bytes;
        CUdeviceptr source = (CUdeviceptr)src->data[p], destination = (CUdeviceptr)dst->data[p];
        int source_pitch = src->linesize[p], destination_pitch = dst->linesize[p];
        void *args[] = {&source, &source_pitch, &sx, &sy, &sw, &sh, &src_bytes, &src_shift,
                        &destination, &destination_pitch, &dx, &dy, &dw, &dh, &cw, &ch, &lanes,
                        &dst_bytes, &dst_shift, &mul};
        if (AVP_CHECK_CU(cuLaunchKernel(convert_kernel_, (dw + 31) / 32, (dh + 7) / 8, 1,
                                        32, 8, 1, 0, stream, args, nullptr)))
            throw Error("cuda_rect_overlay: convert launch failed");
    }
}

void CudaRectDraw::drawLayer(CUstream stream, const av::VideoFrame &src, AVFrame *canvas, const LayerSpec &L) {
    const AVPixelFormat sw_fmt = canvas_.sw_fmt;
    const int canvas_w = canvas_.width, canvas_h = canvas_.height;
    const bool sized = L.dst_w > 0;
    const bool full_copy = !sized || (L.dst_w == L.crop_w && L.dst_h == L.crop_h &&
        L.dst_x >= 0 && L.dst_y >= 0 && L.dst_x + L.dst_w <= canvas_w && L.dst_y + L.dst_h <= canvas_h);
    const AVPixelFormat src_sw_fmt = frameSwFormat(src);
    int rgb_step, r_off, g_off, b_off;
    if (canvas_.transfer != AVCOL_TRC_UNSPECIFIED) {
        const AVFrame *frame = src.raw();
        const bool rgb = isPackedRgb8(src_sw_fmt, rgb_step, r_off, g_off, b_off);
        const bool valid = rgb
            ? isSdrGraphicColor(*frame)
            : frame->color_trc == canvas->color_trc &&
              frame->color_primaries == canvas->color_primaries &&
              frame->colorspace == canvas->colorspace && frame->color_range == AVCOL_RANGE_MPEG;
        if (!valid)
            throw Error("cuda_rect_overlay: missing or mismatched source color metadata; "
                        "declare source color and normalize to the canvas before compositing");
    }
    if (src_sw_fmt != sw_fmt && isRgbToYuvConvertible(src_sw_fmt, sw_fmt) &&
        isPackedRgb8(src_sw_fmt, rgb_step, r_off, g_off, b_off)) {
        convertRgbLayer(stream, src.raw(), canvas, L, rgb_step, r_off, g_off, b_off,
                        packedAlphaOffset(src_sw_fmt));
    } else if (src_sw_fmt != sw_fmt && isYuvPromoteConvertible(src_sw_fmt, sw_fmt)) {
        convertLayer(stream, src_sw_fmt, src.raw(), canvas, L);
    } else if (!full_copy) {
        scaleLayer(stream, src.raw(), canvas, L);
    } else if (!blitLayerPlanes(stream, sw_fmt, src.raw(), canvas, L.crop_x, L.crop_y, L.crop_w,
                                L.crop_h, L.dst_x, L.dst_y))
        throw Error("cuda_rect_overlay: GPU blit failed");

    // When a non-alpha source is drawn onto an alpha canvas, fill the destination
    // alpha rect with the depth's opaque maximum so the output alpha is well-defined.
    if (src_sw_fmt != AV_PIX_FMT_NONE && src_sw_fmt != sw_fmt) {
        const int alpha_p = alphaPlaneIndex(sw_fmt);
        int x = L.dst_x, y = L.dst_y;
        int w = sized ? L.dst_w : L.crop_w, h = sized ? L.dst_h : L.crop_h;
        uint16_t opaque = 255;
        if (alpha_p >= 0 && planeClearValue(sw_fmt, nullptr, alpha_p, opaque) &&
            clipRect(x, y, w, h, canvas_w, canvas_h))
            fillPlaneRect(sw_fmt, canvas, alpha_p, x, y, w, h, opaque);
    }
}

}
