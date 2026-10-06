#pragma once
#include "../../util.hpp"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/parseutils.h>
#include <libavutil/pixdesc.h>
}

#include <array>
#include <cstdint>
#include <string>

namespace avp::mixer {

// Untagged RGB(A) graphics use the compositor's SDR BT.709 interpretation.
// Resolve each missing tag independently; explicit unsupported tags still fail.
// Input frames may be shared, so this does not stamp or modify their metadata.
inline bool isSdrGraphicColor(const AVFrame &frame) {
    return (frame.color_trc == AVCOL_TRC_UNSPECIFIED || frame.color_trc == AVCOL_TRC_BT709) &&
           (frame.color_primaries == AVCOL_PRI_UNSPECIFIED || frame.color_primaries == AVCOL_PRI_BT709);
}

/// The name of an HDR transfer as settings and error texts spell it; nullptr for any other.
inline const char *hdrTransferName(AVColorTransferCharacteristic trc) {
    return trc == AVCOL_TRC_SMPTE2084 ? "pq" : trc == AVCOL_TRC_ARIB_STD_B67 ? "hlg" : nullptr;
}

/// The HDR transfer of a frame tagged `trc` when a canvas of `canvas_fmt` stores 8 bits; nullptr
/// when the frame may be drawn or passed on to it. PQ or HLG samples in an 8-bit frame are
/// HDR-coded pixels in what every consumer reads as SDR, whether the depth is reduced here or
/// was reduced upstream by a filter that kept the tags; getting there properly is tone mapping.
/// SDR and untagged frames, and canvases deeper than 8 bits, raise no objection.
inline const char *hdrOnEightBitCanvas(AVColorTransferCharacteristic trc, AVPixelFormat canvas_fmt) {
    const AVPixFmtDescriptor *d = av_pix_fmt_desc_get(canvas_fmt);
    return d && d->comp[0].depth <= 8 ? hdrTransferName(trc) : nullptr;
}

/// The error for that frame at `node`. `hdr_source_transfer=<pq|hlg>` is there for callers to
/// parse out of the exception text.
inline std::string hdrOnEightBitError(const std::string &node, const char *transfer) {
    return node + ": the source is HDR (hdr_source_transfer=" + transfer + ") and the output is 8-bit; "
           "reducing its depth would leave HDR-coded pixels in an SDR frame, tone map it before this node";
}

/// A dip colour in libavutil's colour syntax ("#RRGGBB", "0xRRGGBB", "black", ...). Alpha
/// is rejected: the colour fills the frame at a dip's midpoint.
inline std::array<uint8_t, 3> parseDipColor(const std::string &text) {
    uint8_t rgba[4];
    if (av_parse_color(rgba, text.c_str(), -1, nullptr) < 0)
        throw Error("invalid dip colour '" + text + "' (expected #RRGGBB)");
    if (rgba[3] != 255)
        throw Error("dip colour '" + text + "' must be opaque");
    return {rgba[0], rgba[1], rgba[2]};
}

}
