#pragma once

extern "C" {
#include <libavutil/frame.h>
}

namespace avp::mixer {

// Untagged RGB(A) graphics use the compositor's SDR BT.709 interpretation.
// Resolve each missing tag independently; explicit unsupported tags still fail.
// Input frames may be shared, so this does not stamp or modify their metadata.
inline bool isSdrGraphicColor(const AVFrame &frame) {
    return (frame.color_trc == AVCOL_TRC_UNSPECIFIED || frame.color_trc == AVCOL_TRC_BT709) &&
           (frame.color_primaries == AVCOL_PRI_UNSPECIFIED || frame.color_primaries == AVCOL_PRI_BT709);
}

}
