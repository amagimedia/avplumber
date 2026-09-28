#include "mixer/primitives/compositor_color.hpp"
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <string>

void print_stack_trace() {}

namespace {
// A code as a canvas of `depth` bits stores it.
int stored(float code, int depth) { return int(code * (1 << (depth - 8)) + 0.5f); }

bool rejected(const std::string &text) {
    try { avp::mixer::parseDipColor(text); }
    catch (const std::exception &) { return true; }
    return false;
}
}

int main() {
    using avp::mixer::canvasCodes;
    using Rgb = std::array<uint8_t, 3>;
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

    // Dip colours land on the codes the compositor draws RGB graphics with, never raw RGB:
    // limited-range BT.709 on SDR canvases, graphics white (203 nits) on HDR ones.
    for (auto transfer : {AVCOL_TRC_BT709, AVCOL_TRC_ARIB_STD_B67, AVCOL_TRC_SMPTE2084}) {
        const auto black = canvasCodes(Rgb{0, 0, 0}, transfer);
        for (int depth : {8, 10}) {
            assert(stored(black[0], depth) == 16 << (depth - 8));
            assert(stored(black[1], depth) == 128 << (depth - 8) && stored(black[2], depth) == 128 << (depth - 8));
        }
        const auto white = canvasCodes(Rgb{255, 255, 255}, transfer);
        assert(stored(white[1], 10) == 512 && stored(white[2], 10) == 512);
    }
    const auto white = canvasCodes(Rgb{255, 255, 255}, AVCOL_TRC_BT709);
    assert(stored(white[0], 8) == 235 && stored(white[0], 10) == 940);
    // HLG reference white is 75% signal (Y 721 of 10 bits), not peak; PQ puts 203 nits near 58%.
    assert(stored(canvasCodes(Rgb{255, 255, 255}, AVCOL_TRC_ARIB_STD_B67)[0], 10) == 721);
    const int pq_white = stored(canvasCodes(Rgb{255, 255, 255}, AVCOL_TRC_SMPTE2084)[0], 10);
    assert(pq_white > 560 && pq_white < 590);
    // Untagged canvases treat graphics as SDR, like the compositor.
    assert(canvasCodes(Rgb{255, 255, 255}, AVCOL_TRC_UNSPECIFIED) == white);
    const auto red = canvasCodes(Rgb{255, 0, 0}, AVCOL_TRC_BT709);
    assert(stored(red[0], 8) == 63 && stored(red[1], 8) == 102 && stored(red[2], 8) == 240);

    using avp::mixer::parseDipColor;
    assert(parseDipColor("#000000") == (Rgb{0, 0, 0}));
    assert(parseDipColor("#FFFFFF") == (Rgb{255, 255, 255}));
    assert(parseDipColor("#ff8000") == (Rgb{255, 128, 0}));
    assert(parseDipColor("0x102030") == (Rgb{16, 32, 48}));
    assert(parseDipColor("white") == (Rgb{255, 255, 255}));
    assert(parseDipColor("#000000ff") == (Rgb{0, 0, 0}));
    for (const char *bad : {"", "#12345", "#1234567", "#gg0000", "nope", "#00000080", "black@0.5"})
        assert(rejected(bad));
}
