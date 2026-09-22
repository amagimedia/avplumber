#pragma once
// GPU side of the CUDA compositor: kernel module lifetime and one draw call per
// frame that composes every resolved layer plus the background in a single
// kernel launch from a rect table. Owns no scheduling state; the node decides
// what to draw when.
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

    /// Canvas formats the composite kernel writes: semiplanar YUV (NV12, P010, P210) and
    /// packed 8-bit RGB without a separate alpha plane (rgb0, bgr0, rgba, bgra).
    static bool canvasSupported(AVPixelFormat sw_fmt);

    /// Make the device's CUDA context current on this thread; throws when the hwaccel is unusable.
    void ensureDevice();
    /// Load the composite kernels and allocate the rect table once; throws without HAVE_NVCC=1.
    void ensureKernels();
    void unload();
    CUstream stream() const { return (CUstream)cuda_dev_->stream; }

    /// Compose the frame: background where nothing draws, then every resolved op in order, in one
    /// kernel launch. Validates each source's color tags against the canvas contract first.
    /// `color_src` decides the clear level (JPEG-range sources clear to 0).
    void draw(CUstream stream, const std::vector<DrawOp> &ops, AVFrame *canvas, const AVFrame *color_src);

    /// sw_format of a hardware frame, AV_PIX_FMT_NONE when it has no frames context.
    static AVPixelFormat frameSwFormat(const av::VideoFrame &f);

private:
    std::shared_ptr<HWAccelDevice> hwaccel_;
    Canvas canvas_;
    AVCUDADeviceContext *cuda_dev_ = nullptr;
    CUmodule module_ = nullptr;
    CUfunction composite_kernel_ = nullptr;       // full: YUV, promote and RGB(A) layers
    CUfunction composite_yuv_kernel_ = nullptr;   // lean: no packed-RGB layers in the table
    CUfunction composite_packed_kernel_ = nullptr; // packed 4-byte RGB canvases
    // Rect table: pinned host staging + device copy, one entry per op, reused every frame
    // (the node synchronizes the stream after each frame).
    AvpRectLayer *table_host_ = nullptr;
    CUdeviceptr table_device_ = 0;

    void fillTableEntry(const DrawOp &op, const AVFrame *canvas, AvpRectLayer &out) const;
    void validateSourceColor(const av::VideoFrame &src, bool packed_rgb, const AVFrame *canvas) const;
};

}
