#pragma once
#include "../util.hpp"
#include "primitives/fade_curve.hpp"
#include "primitives/TakeClock.hpp"
#include <algorithm>
#include <array>
#include <cmath>
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
    // Clamp the transition's clock (ms on the program timeline): a paused transition stays at
    // the ceiling's progress while both sources keep moving; a resumed one continues from the
    // floor. Unset: unclamped.
    std::optional<int64_t> time_floor_ms, time_ceiling_ms;
};

/// The mix's progress before its curve at program time `t_ms`, as the alpha expression has it.
inline double requestProgress(const FadeRequest& request, int64_t t_ms) {
    if (request.time_floor_ms) t_ms = std::max(t_ms, *request.time_floor_ms);
    if (request.time_ceiling_ms) t_ms = std::min(t_ms, *request.time_ceiling_ms);
    const double x = double(t_ms - request.start_ms) / 1000.0 / request.duration_sec;
    return std::clamp(x, 0.0, 1.0);
}

/// `request` moving the mix in a straight line from where it is at `from_ms` to `to` at `to_ms`, then
/// holding there: the lever's position reaches the picture on every frame in between, not in steps
/// at the rate the positions arrive. The duration is negative when the lever moves back.
inline FadeRequest rampTo(FadeRequest request, double to, int64_t from_ms, int64_t to_ms) {
    const double from = requestProgress(request, from_ms);
    to_ms = std::max(to_ms, from_ms + 1);
    if (std::fabs(to - from) < 1e-6) {
        request.duration_sec = 1;
        request.start_ms = from_ms - int64_t(std::llround(to * 1000));
        request.time_floor_ms = request.time_ceiling_ms = from_ms;
        return request;
    }
    request.duration_sec = double(to_ms - from_ms) / 1000.0 / (to - from);
    request.start_ms = from_ms - int64_t(std::llround(from * request.duration_sec * 1000));
    request.time_floor_ms = from_ms;
    request.time_ceiling_ms = to_ms;
    return request;
}

/// `request`'s schedule in ns, as the alpha expression, the keys' clock and a scrubbed wipe clip
/// all evaluate it.
inline RampNs rampTimesNs(const FadeRequest& request) {
    RampNs ramp{request.start_ms * 1000000, int64_t(request.duration_sec * 1e9)};
    if (request.time_floor_ms) ramp.floor_ns = *request.time_floor_ms * 1000000;
    if (request.time_ceiling_ms) ramp.ceiling_ns = *request.time_ceiling_ms * 1000000;
    return ramp;
}

/// `request` running on from where it is at `at_ms` to its end at the rate of a whole take of
/// `duration_sec`, unclamped from there: AUTO taking over from the lever.
inline FadeRequest runOn(FadeRequest request, int64_t at_ms, double duration_sec) {
    const double x = requestProgress(request, at_ms);
    request.duration_sec = duration_sec;
    request.start_ms = at_ms - int64_t(std::llround(x * duration_sec * 1000));
    request.time_floor_ms = at_ms;
    request.time_ceiling_ms.reset();
    return request;
}

/// The time a lever position takes to reach the picture (ns): the controller's own spacing of its
/// positions when it sends that (`span_ms`, paced against network jitter), one frame to 200 ms, else
/// the time since the previous one arrived at `last_at_ns` (0: none): one frame when positions
/// stream in, at most 100 ms when they come seldom. `period_ns`: one output frame.
inline int64_t leverSpanNs(std::optional<double> span_ms, int64_t now_ns, int64_t last_at_ns, int64_t period_ns) {
    if (span_ms) return std::clamp<int64_t>(int64_t(*span_ms * 1e6), period_ns, 200000000);
    return last_at_ns ? std::clamp<int64_t>(now_ns - last_at_ns, period_ns, 100000000) : period_ns;
}

/// The lever's ramp to an end has reached the program frame at `pts_ns`. On time, never on the
/// progress: a ramp's start is whole milliseconds, so its progress may stop just short of 0 or 1.
inline bool leverEndLanded(const FadeRequest& request, int64_t pts_ns) {
    return request.time_ceiling_ms && pts_ns >= *request.time_ceiling_ms * 1000000;
}

struct TransitionCommand {
    std::string key;
    Parameters value;
};

// Constructed per transition, never dispatched per frame; applied in order.
// Routing and the presentation deadline remain the orchestrator's responsibility.
using TransitionControl = std::vector<TransitionCommand> (*)(const FadeRequest&);
TransitionControl transitionControl(const std::string& backend);

} // namespace avp::mixer
