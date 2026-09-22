#pragma once
// GPU side of the CUDA compositor: kernel module lifetime, canvas clearing and
// drawing one resolved layer (blit, scale, 8->10-bit promote, packed RGB(A)
// conversion). Owns no scheduling state; the node decides what to draw when.
#include "../../hwaccel.hpp"
#include "../../mixer/primitives/compositor_layers.hpp"
#include "cuda_rect_table.h"
#include <cuda_loader/cuda_drvapi_dynlink_cuda.h>

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

#include <memory>

namespace avp::mixer {

int checkCu(CUresult err, const char *func);
#define AVP_CHECK_CU(x) ::avp::mixer::checkCu((x), #x)

class CudaRectDraw {
public:
    struct Canvas {
        int width = 0;
        int height = 0;
        AVPixelFormat sw_fmt = AV_PIX_FMT_NONE;
        // Canvas transfer; unspecified skips color validation and treats graphics as SDR.
        AVColorTransferCharacteristic transfer = AVCOL_TRC_UNSPECIFIED;
        float sdr_white = 203.f;
        float hdr_peak = 1000.f;
    };

    CudaRectDraw(std::shared_ptr<HWAccelDevice> hw, Canvas canvas)
        : hwaccel_(std::move(hw)), canvas_(canvas) {}
    ~CudaRectDraw() { unload(); }
    CudaRectDraw(const CudaRectDraw &) = delete;
    CudaRectDraw &operator=(const CudaRectDraw &) = delete;

    const Canvas &canvas() const { return canvas_; }
    void setColor(AVColorTransferCharacteristic transfer, float sdr_white, float hdr_peak) {
        canvas_.transfer = transfer;
        canvas_.sdr_white = sdr_white;
        canvas_.hdr_peak = hdr_peak;
    }

    /// Make the device's CUDA context current on this thread; throws when the hwaccel is unusable.
    void ensureDevice();
    /// Load the scaling/convert kernels once (no-op without HAVE_CUDA_RECT_SCALE).
    void ensureKernels();
    void unload();
    CUstream stream() const { return (CUstream)cuda_dev_->stream; }

    /// Clear the whole canvas to opaque black in the canvas format.
    void clearCanvas(AVFrame *canvas, const AVFrame *color_src);

    /// Validate the source's color tags against the canvas contract (when one is set), then draw
    /// the resolved layer: RGB conversion, YUV promote, scale, or plain blit; finally make the
    /// destination alpha opaque on alpha canvases fed by non-alpha sources.
    void drawLayer(CUstream stream, const av::VideoFrame &src, AVFrame *canvas, const LayerSpec &layer);

    /// Draw every resolved op and the background in ONE kernel launch (semiplanar canvases
    /// without alpha only). Returns false, having touched nothing, when the canvas format or an
    /// op is outside what the batched kernel covers; the caller then clears and draws per layer.
    /// Output is bit-identical to clearCanvas + drawLayer for each op in order.
    bool drawBatched(CUstream stream, const std::vector<DrawOp> &ops, AVFrame *canvas, const AVFrame *color_src);
    /// Whether the canvas format can take the batched path at all.
    bool batchedSupported() const;

    /// sw_format of a hardware frame, AV_PIX_FMT_NONE when it has no frames context.
    static AVPixelFormat frameSwFormat(const av::VideoFrame &f);

private:
    std::shared_ptr<HWAccelDevice> hwaccel_;
    Canvas canvas_;
    AVCUDADeviceContext *cuda_dev_ = nullptr;
    CUmodule scale_module_ = nullptr;
    CUfunction scale_kernel_ = nullptr;
    CUfunction convert_kernel_ = nullptr;
    CUfunction rgb_kernel_ = nullptr;
    CUfunction rgba_kernel_ = nullptr;
    CUfunction composite_kernel_ = nullptr;
    CUfunction composite_yuv_kernel_ = nullptr;
    // Rect table for the batched kernel: pinned host staging + device copy, one entry per op.
    AvpRectLayer *table_host_ = nullptr;
    CUdeviceptr table_device_ = 0;

    bool fillTableEntry(const DrawOp &op, const AVFrame *canvas, AvpRectLayer &out);
    void validateSourceColor(const av::VideoFrame &src, AVPixelFormat src_sw_fmt, const AVFrame *canvas) const;

    void convertRgbLayer(CUstream stream, const AVFrame *src, AVFrame *dst, const LayerSpec &layer,
                         int step, int r_off, int g_off, int b_off, int a_off);
    void scaleLayer(CUstream stream, const AVFrame *src, AVFrame *dst, const LayerSpec &layer);
    void convertLayer(CUstream stream, AVPixelFormat src_fmt, const AVFrame *src, AVFrame *dst,
                      const LayerSpec &layer);
};

}
