#pragma once
// The progress of the background transition on air, published by the orchestrator for the keys
// that go with it (the downstream keyer's Next Transition keys). It is the mix's own clock:
// progress(t) = curve(clamp((min(max(t, floor), ceiling) - start) / duration)) at program pts t,
// with the floor and ceiling a pause or the lever sets. Each take on air gets a new generation;
// its end is Completed (keys take their new state from the program frame at end_ns on) or
// Cancelled (they return to the state they had). The previous take's end travels along, so a
// reader that missed it still applies it. Pure; tests/cpp/test_lever.cpp.
#include "fade_curve.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace avp::mixer {

struct TakeClock {
    enum class End : uint8_t { Running, Completed, Cancelled };
    struct Ending {
        uint64_t generation = 0;
        End end = End::Running;
        int64_t end_ns = 0;
    };
    uint64_t generation = 0;   // 0: no take yet
    int64_t start_ns = 0;
    int64_t duration_ns = 0;   // 0: a cut, whole from start_ns; negative: the lever moving back
    int64_t floor_ns = std::numeric_limits<int64_t>::min();
    int64_t ceiling_ns = std::numeric_limits<int64_t>::max();
    FadeCurve curve = FadeCurve::Linear;
    End end = End::Running;
    int64_t end_ns = 0;
    Ending previous;

    double progress(int64_t pts_ns) const {
        if (duration_ns == 0) return pts_ns >= start_ns ? 1.0 : 0.0;
        const int64_t t = std::min(std::max(pts_ns, floor_ns), ceiling_ns);
        return fadeCurveAt(curve, std::clamp(double(t - start_ns) / double(duration_ns), 0.0, 1.0));
    }
    /// The clock of the next take: this one's end, Cancelled if it had none, goes along.
    TakeClock next(int64_t start, int64_t duration, FadeCurve shape) const {
        TakeClock clock;
        clock.generation = generation + 1;
        clock.start_ns = start;
        clock.duration_ns = duration;
        clock.curve = shape;
        clock.previous = {generation, end == End::Running ? End::Cancelled : end, end_ns};
        return clock;
    }
};

} // namespace avp::mixer
