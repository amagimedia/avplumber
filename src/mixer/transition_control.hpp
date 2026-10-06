#pragma once
#include "../util.hpp"
#include "primitives/fade_curve.hpp"
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

extern "C" {
#include <libavutil/pixfmt.h>
}

namespace avp::mixer {

struct FadeRequest {
    int64_t start_ms;
    double duration_sec;
    bool destination_is_a;
    FadeCurve curve = FadeCurve::Linear;  // shapes progress along the transition, either direction
    // set: dip through this opaque SDR RGB colour instead of mixing; the curve shapes each half
    std::optional<std::array<uint8_t, 3>> dip = std::nullopt;
    double dip_hold = 0;  // the dip colour's hold alone, as a share of the duration (dipHoldEnd)
    AVColorTransferCharacteristic transfer = AVCOL_TRC_BT709;  // the canvas the dip colour is drawn on
};

struct TransitionCommand {
    std::string key;
    Parameters value;
};

// Constructed per transition, never dispatched per frame; applied in order.
// Routing and the presentation deadline remain the orchestrator's responsibility.
using TransitionControl = std::vector<TransitionCommand> (*)(const FadeRequest&);
TransitionControl transitionControl(const std::string& backend);

} // namespace avp::mixer
