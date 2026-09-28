#pragma once
#include "../util.hpp"
#include "primitives/fade_curve.hpp"
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace avp::mixer {

/// A dip colour as canvas codes: Y, Cb, Cr at 8-bit limited-range scale (canvasCodes).
using DipCodes = std::optional<std::array<float, 3>>;

struct FadeRequest {
    int64_t start_ms;
    double duration_sec;
    bool destination_is_a;
    FadeCurve curve = FadeCurve::Linear;  // shapes progress along the transition, either direction
    DipCodes dip = std::nullopt;  // set: dip through this colour instead of mixing; the curve shapes each half
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
