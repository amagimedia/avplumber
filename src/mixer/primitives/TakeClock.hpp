#pragma once
// The progress of the background transition on air, published by the orchestrator for the keys
// that go with it (the downstream keyer's Next Transition keys). It is the mix's own clock:
// progress(t) = curve(clamp((min(max(t, floor), ceiling) - start) / duration)) at program pts t,
// with the floor and ceiling a pause or the lever sets; a set `step` switches the keys whole where
// that linear part reaches it, as a lever wipe switches its background; the schedule is a
// RampHistory, so a lever move never changes a tick the keys may not have drawn yet. Each take on
// air gets a new generation; its end is Completed (keys take their new state from the program frame
// at end_ns on) or Cancelled (they return to the state they had). The previous take's end travels
// along, so a reader that missed it still applies it. Pure; tests/cpp/test_lever.cpp.
#include "fade_curve.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace avp::mixer {

/// A take's schedule on the program timeline (ns): linear from start_ns over duration_ns (negative:
/// moving back; 0: whole from start_ns), the time clamped to [floor_ns, ceiling_ns].
struct RampNs {
    int64_t start_ns = 0;
    int64_t duration_ns = 0;
    int64_t floor_ns = std::numeric_limits<int64_t>::min();
    int64_t ceiling_ns = std::numeric_limits<int64_t>::max();
};

/// The schedule's progress in [0, 1] at program pts `t_ns`, before any curve.
inline double rampAt(const RampNs& ramp, int64_t t_ns) {
    if (ramp.duration_ns == 0) return t_ns >= ramp.start_ns ? 1.0 : 0.0;
    const int64_t t = std::min(std::max(t_ns, ramp.floor_ns), ramp.ceiling_ns);
    return std::clamp(double(t - ramp.start_ns) / double(ramp.duration_ns), 0.0, 1.0);
}

/// A take's schedule as the lever, a pause or AUTO change it while ticks of it are in flight. A
/// change agrees with the schedule it changes from its floor up to the tick it is anchored on
/// (rampTo and runOn put the floor there, a pause only sets a ceiling past it), so it applies from
/// its floor on and every earlier tick keeps the value it had: the clip player, the program
/// selector and the keys, each reading a tick at its own time, read one schedule for it. The
/// newest kSize changes are kept: a lever streaming a position per tick changes it at most once a
/// tick, and readers trail the newest change by far fewer ticks; a tick older than all of them
/// reads the oldest kept.
class RampHistory {
public:
    static constexpr size_t kSize = 16;
    RampHistory() = default;
    /// One schedule for the whole take.
    explicit RampHistory(const RampNs& ramp) : ramps_{ramp} {}
    /// `ramp` from its floor on; it replaces the changes from that floor or later.
    void change(const RampNs& ramp) {
        while (count_ > 1 && ramps_[count_ - 1].floor_ns >= ramp.floor_ns) --count_;
        if (count_ == kSize) {
            std::move(ramps_.begin() + 1, ramps_.end(), ramps_.begin());
            --count_;
        }
        ramps_[count_++] = ramp;
    }
    /// The schedule tick `t_ns` follows.
    const RampNs& at(int64_t t_ns) const {
        size_t i = count_ - 1;
        while (i > 0 && ramps_[i].floor_ns > t_ns) --i;
        return ramps_[i];
    }

private:
    std::array<RampNs, kSize> ramps_{};
    size_t count_ = 1;
};

inline double rampAt(const RampHistory& schedule, int64_t t_ns) { return rampAt(schedule.at(t_ns), t_ns); }

/// Where a wipe clip switches the background (and the keys going with it) from Program to Preview.
constexpr double kWipeSwitchAt = 0.5;

struct TakeClock {
    enum class End : uint8_t { Running, Completed, Cancelled };
    struct Ending {
        uint64_t generation = 0;
        End end = End::Running;
        int64_t end_ns = 0;
    };
    uint64_t generation = 0;   // 0: no take yet
    RampHistory schedule;      // a duration 0: a cut, whole from its start
    FadeCurve curve = FadeCurve::Linear;
    double step = -1;          // >= 0: whole (1) from where the linear progress reaches it, else 0
    int64_t on_air_ns = 0;     // the take's first program frame; the clamps never move it
    End end = End::Running;
    int64_t end_ns = 0;
    Ending previous;

    double progress(int64_t pts_ns) const {
        const RampNs& ramp = schedule.at(pts_ns);
        const double x = rampAt(ramp, pts_ns);
        if (step >= 0) return x >= step ? 1.0 : 0.0;
        return ramp.duration_ns == 0 ? x : fadeCurveAt(curve, x);
    }
    /// The clock of the next take, on air from `start`: this one's end, Cancelled if it had none,
    /// goes along.
    TakeClock next(int64_t start, int64_t duration, FadeCurve shape, double switch_at = -1) const {
        TakeClock clock;
        clock.generation = generation + 1;
        clock.schedule = RampHistory({start, duration});
        clock.on_air_ns = start;
        clock.curve = shape;
        clock.step = switch_at;
        clock.previous = {generation, end == End::Running ? End::Cancelled : end, end_ns};
        return clock;
    }
};

} // namespace avp::mixer
