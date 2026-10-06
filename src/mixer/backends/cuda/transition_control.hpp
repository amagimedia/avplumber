#pragma once
#include "../../transition_control.hpp"
#include "../../../nodes/hwaccel/graphic_color.h"

namespace avp::mixer::cuda {
std::vector<TransitionCommand> fadeCommand(const FadeRequest& request);

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
}
