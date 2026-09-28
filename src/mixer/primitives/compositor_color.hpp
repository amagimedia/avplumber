#pragma once
#include "../../nodes/hwaccel/graphic_color.h"
#include "../../util.hpp"

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/parseutils.h>
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

/// An opaque SDR RGB colour on a canvas of `transfer`, as the compositor draws RGB graphics:
/// Y, Cb, Cr at 8-bit limited-range scale (multiply by 1 << (depth - 8)), unrounded.
inline std::array<float, 3> canvasCodes(const std::array<uint8_t, 3> &rgb, AVColorTransferCharacteristic transfer) {
    const int id = kernelTransfer(transfer);
    float c[3] = {float(rgb[0]), float(rgb[1]), float(rgb[2])};
    convert_graphic_rgb(c, id, kGraphicSdrWhite, kGraphicHdrPeak);
    std::array<float, 3> codes{graphic_luma(c, id), 0.f, 0.f};
    graphic_chroma(c[0], c[1], c[2], id, codes[1], codes[2]);
    return codes;
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
