#pragma once
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
