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
#include <string>

namespace avp::mixer {

enum class FadeCurve { Linear, EaseIn, EaseOut, EaseInOut };

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

/// A dip (fade through a colour) runs the curve once per half: y(x) = C(2x)/2
/// before the midpoint and 1/2 + C(2x-1)/2 from it, so y(1/2) == 1/2 exactly for
/// every curve, and 1-y too; transition_cuda shows the colour alone at 1/2.
inline double dipCurveAt(FadeCurve curve, double x) {
    x = std::min(1.0, std::max(0.0, x));
    return x < 0.5 ? 0.5 * fadeCurveAt(curve, 2 * x) : 0.5 + 0.5 * fadeCurveAt(curve, 2 * x - 1);
}

/// dipCurveAt as an FFmpeg expression of `progress` (clamped as for
/// fadeCurveExpression), term for term. Progress lives in variable 1 because
/// the curve uses variable 0; the result is parenthesised for "1-" + result.
inline std::string dipCurveExpression(FadeCurve curve, const std::string& progress) {
    return "(st(1," + progress + ");if(lt(ld(1),0.5),0.5*(" + fadeCurveExpression(curve, "2*ld(1)") +
           "),0.5+0.5*(" + fadeCurveExpression(curve, "2*ld(1)-1") + ")))";
}

}  // namespace avp::mixer
