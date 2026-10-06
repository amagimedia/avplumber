#pragma once
// What a rect node does around CudaRectDraw to own an output canvas: the canvas parameters, the
// linear CUDA frame pool of that canvas, a blank frame from it and the source's properties on it.
// Shared by the compositor nodes (many inputs, one canvas) and cuda_transform (one input, many).
#include "../node_common.hpp"
#include "../../avbuffer.hpp"
#include "../../hwaccel.hpp"
#include "cuda_rect_draw.hpp"
#include "cuda_rect_texture.h"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}

#include <string>

namespace avp::mixer {

/// `width`, `height`, `sw_format` and the optional `color` contract of a canvas. `prefix` starts
/// every error.
inline CudaRectDraw::Canvas parseCanvas(const Parameters &params, const std::string &prefix) {
    CudaRectDraw::Canvas canvas;
    canvas.width = params.at("width").get<int>();
    canvas.height = params.at("height").get<int>();
    if (canvas.width <= 0 || canvas.height <= 0)
        throw Error(prefix + "width and height must be positive");
    const std::string sw_name = params.value("sw_format", std::string("nv12"));
    canvas.sw_fmt = av_get_pix_fmt(sw_name.c_str());
    if (canvas.sw_fmt == AV_PIX_FMT_NONE)
        throw Error(prefix + "unknown sw_format");
    if (!CudaRectDraw::canvasSupported(canvas.sw_fmt))
        throw Error(prefix + "sw_format must be semiplanar YUV (nv12, p010le, p210le) or packed 8-bit RGB");
    if (params.contains("color")) {
        const auto color = params.at("color").get<std::string>();
        canvas.transfer = graphicTransfer(color);
        if (canvas.transfer == AVCOL_TRC_UNSPECIFIED)
            throw Error(prefix + "color must be sdr, hlg or pq");
        if (canvas.transfer != AVCOL_TRC_BT709 && av_pix_fmt_desc_get(canvas.sw_fmt)->comp[0].depth < 10)
            throw Error(prefix + "HDR canvas requires 10-bit storage");
        canvas.sdr_white = params.value("sdr_white", kGraphicSdrWhite);
        canvas.hdr_peak = params.value("hdr_peak", kGraphicHdrPeak);
        if (!(canvas.sdr_white >= 1.f && canvas.sdr_white <= canvas.hdr_peak &&
              canvas.hdr_peak >= 100.f && canvas.hdr_peak <= 10000.f))
            throw Error(prefix + "invalid display white/peak");
    }
    return canvas;
}

/// The pool of linear CUDA frames a canvas is drawn into.
inline avp::AvBufferRef allocCanvasFrames(HWAccelDevice &hw, const CudaRectDraw::Canvas &canvas,
                                          const std::string &type) {
    avp::AvBufferRef frames(av_hwframe_ctx_alloc(hw.deviceContext()));
    if (!frames)
        throw Error(type + ": av_hwframe_ctx_alloc failed");
    AVHWFramesContext *fc = (AVHWFramesContext *)frames->data;
    fc->format = AV_PIX_FMT_CUDA;
    fc->sw_format = canvas.sw_fmt;
    fc->width = canvas.width;
    fc->height = canvas.height;
    int err = av_hwframe_ctx_init(frames.get());
    if (err < 0)
        throw Error(type + ": av_hwframe_ctx_init (output) failed: " + av::error2string(err));
    return frames;
}

/// A frame from the canvas pool, not drawn yet.
inline av::VideoFrame canvasFrame(AVBufferRef *frames, const std::string &type) {
    av::VideoFrame outf;
    int r = av_hwframe_get_buffer(frames, outf.raw(), 0);
    if (r < 0)
        throw Error(type + ": av_hwframe_get_buffer failed: " + av::error2string(r));
    outf.setComplete(true);   // av_hwframe_get_buffer set hw_frames_ctx, format and size
    return outf;
}

/// Gives a drawn canvas frame the properties (timing, metadata, side data) of `src`.
inline void copySourceProps(av::VideoFrame &outf, const av::VideoFrame &src, const std::string &type) {
    const int cpy = av_frame_copy_props(outf.raw(), src.raw());
    if (cpy < 0)
        throw Error(type + ": av_frame_copy_props failed: " + av::error2string(cpy));
    // The output owns a new canvas; an imported source texture describes
    // only that source's storage, not these rendered pixels.
    if (textureFrameDesc(outf.raw()))
        av_buffer_unref(&outf.raw()->opaque_ref);
}

}
