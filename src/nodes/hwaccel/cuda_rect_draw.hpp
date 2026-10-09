#pragma once
// GPU side of the CUDA compositor: kernel module lifetime and one draw call per
// frame that composes every resolved layer plus the background in a single
// kernel launch from a rect table. Owns no scheduling state; the node decides
// what to draw when.
#include "../../hwaccel.hpp"
#include "../../mixer/primitives/compositor_layers.hpp"
#include "cuda_rect_table.h"
#include "cuda_rect_viewport.h"
#include "graphic_color.h"
#include <cuda_loader/cuda_drvapi_dynlink_cuda.h>

extern "C" {
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

#include <memory>
#include <vector>

namespace avp::mixer {

int checkCu(CUresult err, const char *func);
#define AVP_CHECK_CU(x) ::avp::mixer::checkCu((x), #x)

class RectArrayTextures;

class CudaRectDraw {
public:
    struct Canvas {
        int width = 0;
        int height = 0;
        AVPixelFormat sw_fmt = AV_PIX_FMT_NONE;
        // Canvas transfer; unspecified skips color validation and treats graphics as SDR.
        AVColorTransferCharacteristic transfer = AVCOL_TRC_UNSPECIFIED;
        float sdr_white = kGraphicSdrWhite;
        float hdr_peak = kGraphicHdrPeak;
        // Draw a PQ/HLG source that is deeper than an 8-bit canvas anyway, keeping its tags.
        // Off, such a layer is an error (hdrOnEightBitCanvas): it needs tone mapping first.
        bool allow_hdr_depth_reduction = false;
    };

    CudaRectDraw(std::shared_ptr<HWAccelDevice> hw, Canvas canvas, int max_layers = 256);
    ~CudaRectDraw();
    CudaRectDraw(const CudaRectDraw &) = delete;
    CudaRectDraw &operator=(const CudaRectDraw &) = delete;

    const Canvas &canvas() const { return canvas_; }
    int maxLayers() const { return max_layers_; }

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
    /// `color_src` decides the clear level (JPEG-range sources clear to 0). A layer's opacity
    /// below 1 weights a blended RGBA source's alpha; other kinds cannot fade and draw opaque.
    /// A layer's filter applies to YUV sources on YUV canvases; packed RGB draws bilinear.
    /// A YUV source deeper than the canvas is reduced to its depth (rounded, not dithered, not
    /// tone mapped); a PQ/HLG one on an 8-bit canvas throws unless the canvas allows it.
    void draw(CUstream stream, const std::vector<DrawOp> &ops, AVFrame *canvas, const AVFrame *color_src);

    /// As draw(), for a canvas whose bottom layer is `base` unchanged: a frame of the canvas's
    /// format and size drawn whole, unscaled, over the whole canvas. Copies `base` to the canvas
    /// and composes only the rectangles the other layers cover, so a key on a third of the
    /// picture costs a third of the kernel's work. The result is the one draw() gives.
    /// Returns false, having drawn nothing, when the layers are not of that shape or cover so
    /// much that one launch over the canvas is as cheap; the caller then calls draw().
    bool drawOver(CUstream stream, const std::vector<DrawOp> &ops, AVFrame *canvas, const AVFrame *color_src,
                  const AVFrame *base);
    /// drawOver() gives up beyond this many separate rectangles or this share of the canvas.
    static constexpr size_t kMaxViewports = 4;
    static constexpr int kMaxViewportPercent = 60;

    /// sw_format of a hardware frame, AV_PIX_FMT_NONE when it has no frames context.
    static AVPixelFormat frameSwFormat(const av::VideoFrame &f);
    static bool frameSupported(AVPixelFormat format);

private:
    std::shared_ptr<HWAccelDevice> hwaccel_;
    Canvas canvas_;
    const int max_layers_;
    AVCUDADeviceContext *cuda_dev_ = nullptr;
    CUmodule module_ = nullptr;
    CUfunction composite_kernel_ = nullptr;       // full: YUV, promote and RGB(A) layers
    CUfunction composite_yuv_kernel_ = nullptr;   // lean: no packed-RGB layers in the table
    CUfunction composite_packed_kernel_ = nullptr; // packed 4-byte RGB canvases
    CUfunction composite_opacity_kernel_ = nullptr; // full, plus faded RGBA layers (LayerSpec::opacity < 1)
    CUfunction composite_array_kernel_ = nullptr;
    CUfunction composite_yuv_array_kernel_ = nullptr;
    CUfunction composite_opacity_array_kernel_ = nullptr;
    // Tables with a layer whose filter is not bilinear or whose source is deeper than the canvas;
    // no other table reaches these.
    // The first two hold what the scaler's defaults use (demotion, the cubic at A = 0, 4 samples).
    CUfunction composite_yuv_filter_kernel_ = nullptr;        // no packed-RGB layers, linear sources
    CUfunction composite_yuv_array_filter_kernel_ = nullptr;  // no packed-RGB layers, some arrays
    CUfunction composite_filter_kernel_ = nullptr;            // any layer kind, fade, storage and filter
    bool opacity_warned_ = false;   // an op with opacity < 1 on a kind that cannot blend, logged once
    bool filter_warned_ = false;    // an op naming a filter on a kind drawn bilinear, logged once
    // Rect table: pinned host staging + device copy, one entry per op, reused every frame
    // (the node synchronizes the stream after each frame).
    AvpRectLayer *table_host_ = nullptr;
    CUdeviceptr table_device_ = 0;
    std::unique_ptr<RectArrayTextures> array_textures_;
    CUevent producer_ready_ = nullptr;
    std::vector<CUstream> waited_streams_;

    // Which kernel entry a table needs.
    struct TableKinds {
        bool rgb = false, fade = false, array = false, filter = false, wide = false;
    };
    std::vector<RectViewport> over_views_;   // drawOver's last rectangles, logged when they change

    int fillTable(CUstream stream, const std::vector<DrawOp> &ops, const AVFrame *canvas, TableKinds &kinds);
    void launch(CUstream stream, const TableKinds &kinds, CUdeviceptr table, int n, AVFrame *canvas,
                const AVFrame *color_src, const RectViewport &view);
    bool baseCoversCanvas(const DrawOp &op, const AVFrame *base) const;
    void fillTableEntry(const DrawOp &op, const AVFrame *canvas, AvpRectLayer &out);
    void waitForProducer(const AVFrame *src, CUstream stream);
    void validateSourceColor(const av::VideoFrame &src, bool packed_rgb, const AVFrame *canvas) const;
};

}
