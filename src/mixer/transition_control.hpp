#pragma once
#include "../util.hpp"
#include <cstdint>

namespace avp::mixer {

struct FadeRequest {
    int64_t start_ms;
    double duration_sec;
    bool destination_is_a;
};

struct TransitionCommand {
    std::string key;
    Parameters value;
};

// Constructed per transition, never dispatched per frame. Routing and the
// presentation deadline remain the orchestrator's responsibility.
using TransitionControl = TransitionCommand (*)(const FadeRequest&);
TransitionControl transitionControl(const std::string& backend);

} // namespace avp::mixer
