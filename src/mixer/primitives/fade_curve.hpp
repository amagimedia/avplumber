#pragma once
// The easing curve of a fade, shared by the M/E fade (an FFmpeg expression the
// transition filter evaluates per frame) and the downstream keyer's key fades
// (KeyFade, evaluated by the keyer per program frame). Both run on the host,
// once per frame; the CUDA kernels only ever receive the resulting weight.
//
// A curve maps transition progress x = elapsed / duration to visual progress
// y(x). For every curve y(0) == 0 and y(1) == 1 exactly, so a finished fade lands
// on its target value. Linear is the default and is what fades did before curves.
#include "../../util.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>

namespace avp::mixer {

enum class FadeCurve : std::uint8_t { Linear, EaseIn, EaseOut, EaseInOut };

inline FadeCurve parseFadeCurve(const std::string& name) {
    if (name == "linear") return FadeCurve::Linear;
    if (name == "ease-in") return FadeCurve::EaseIn;
    if (name == "ease-out") return FadeCurve::EaseOut;
    if (name == "ease-in-out") return FadeCurve::EaseInOut;
    throw Error("unknown fade curve '" + name + "' (expected linear, ease-in, ease-out or ease-in-out)");
}

/// y(x) with x clamped to [0, 1]; NaN counts as 0. The arithmetic matches
/// fadeCurveExpression term for term, so both give bit-identical results.
inline double fadeCurveAt(FadeCurve curve, double x) {
    x = std::min(1.0, std::max(0.0, x));
    switch (curve) {
        case FadeCurve::EaseIn: return x * x;
        case FadeCurve::EaseOut: return x * (2 - x);               // 1-(1-x)^2
        case FadeCurve::EaseInOut: return x * x * (3 - 2 * x);     // smoothstep
        case FadeCurve::Linear: break;
    }
    return x;
}

/// The curve as an FFmpeg expression (libavutil/eval) of `progress`, which must
/// already be clamped to [0, 1] (the M/E fade passes clip((t-start)/dur,0,1)).
/// Linear returns `progress` unchanged, so the default fade command stays
/// exactly what it was. The other curves come back parenthesised, keeping
/// "1-" + result valid: progress is stored once in variable 0 and read back.
inline std::string fadeCurveExpression(FadeCurve curve, const std::string& progress) {
    const char* shape = nullptr;
    switch (curve) {
        case FadeCurve::EaseIn: shape = "ld(0)*ld(0)"; break;
        case FadeCurve::EaseOut: shape = "ld(0)*(2-ld(0))"; break;
        case FadeCurve::EaseInOut: shape = "ld(0)*ld(0)*(3-2*ld(0))"; break;
        case FadeCurve::Linear: return progress;
    }
    return "(st(0," + progress + ");" + shape + ")";
}

/// A dip (fade through a colour) runs the curve once per half around a hold:
/// y(x) = C(x/lo)/2 before it, exactly 1/2 over [lo, hi) and 1/2 + C((x-hi)/lo)/2
/// from hi, so y is 1/2 there in either direction (1-y too) and transition_cuda
/// shows the colour alone. `hold` is the hold's share of the dip, one frame
/// period / duration: a half-open window one period wide always contains a
/// frame of the 1/fps grid, whatever the start phase, whereas the bare midpoint
/// (hold 0) is seldom a frame. The dip keeps its duration. Clamped to [0, 1/2],
/// NaN counts as 0. Returns hi; lo = 1 - hi is exact, so y(1) == 1 exactly.
inline double dipHoldEnd(double hold) {
    return 0.5 + 0.5 * std::min(0.5, std::max(0.0, hold));
}

/// The dip's y(x) above as an FFmpeg expression of `progress` (clamped as for
/// fadeCurveExpression), term for term: %.17g round-trips, so the bounds are
/// dipHoldEnd's doubles. Progress lives in variable 1 because the curve uses
/// variable 0; the result is parenthesised for "1-" + result.
inline std::string dipCurveExpression(FadeCurve curve, const std::string& progress, double hold) {
    const double end = dipHoldEnd(hold);
    char lo[32], hi[32];
    std::snprintf(lo, sizeof lo, "%.17g", 1 - end);
    std::snprintf(hi, sizeof hi, "%.17g", end);
    return "(st(1," + progress + ");if(lt(ld(1)," + lo + "),0.5*(" +
           fadeCurveExpression(curve, "ld(1)/" + std::string(lo)) + "),if(lt(ld(1)," + hi + "),0.5,0.5+0.5*(" +
           fadeCurveExpression(curve, "(ld(1)-" + std::string(hi) + ")/" + lo) + "))))";
}

}  // namespace avp::mixer
