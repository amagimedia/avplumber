#pragma once
// Shared includes and name imports for the MixerOrchestrator translation units
// (src/mixer/orchestrator/*.cpp). Not part of the public interface.
#include "MixerOrchestrator.hpp"
#include "../graph_ops.hpp"
#include "../routing.hpp"
#include "../primitives/TransitionGuard.hpp"
#include "../../avutils.hpp"
#include "../../graph_interfaces.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <sstream>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace avp::mixer {
using namespace avp::mixer::graph;

/// A schedule as the clip_cache node's "scrub" object takes it.
inline Parameters scrubSchedule(const RampNs& ramp) {
    return {{"start_ns", ramp.start_ns}, {"duration_ns", ramp.duration_ns},
            {"floor_ns", ramp.floor_ns}, {"ceiling_ns", ramp.ceiling_ns}};
}
}
