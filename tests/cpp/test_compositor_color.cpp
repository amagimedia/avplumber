#include "mixer/primitives/compositor_color.hpp"
#include <cassert>
#include <initializer_list>

int main() {
    AVFrame frame{};
    for (auto transfer : {AVCOL_TRC_UNSPECIFIED, AVCOL_TRC_BT709}) {
        for (auto primaries : {AVCOL_PRI_UNSPECIFIED, AVCOL_PRI_BT709}) {
            frame.color_trc = transfer;
            frame.color_primaries = primaries;
            assert(avp::mixer::isSdrGraphicColor(frame));
            assert(frame.color_trc == transfer && frame.color_primaries == primaries);
        }
    }
    // A missing companion tag must not turn an explicit HDR/wide-gamut
    // declaration into SDR. Fully tagged unsupported combinations also fail.
    for (auto transfer : {AVCOL_TRC_ARIB_STD_B67, AVCOL_TRC_SMPTE2084, AVCOL_TRC_GAMMA22}) {
        for (auto primaries : {AVCOL_PRI_UNSPECIFIED, AVCOL_PRI_BT709, AVCOL_PRI_BT2020}) {
            frame.color_trc = transfer;
            frame.color_primaries = primaries;
            assert(!avp::mixer::isSdrGraphicColor(frame));
        }
    }
    for (auto transfer : {AVCOL_TRC_UNSPECIFIED, AVCOL_TRC_BT709}) {
        frame.color_trc = transfer;
        frame.color_primaries = AVCOL_PRI_BT2020;
        assert(!avp::mixer::isSdrGraphicColor(frame));
    }
}
