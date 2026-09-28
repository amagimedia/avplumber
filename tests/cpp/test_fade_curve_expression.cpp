// The M/E fade hands the curve to FFmpeg as an expression; it must evaluate to
// what fadeCurveAt computes for the keyer.
#include "mixer/primitives/fade_curve.hpp"
extern "C" {
#include <libavutil/eval.h>
}
#include <cassert>
#include <cmath>
using namespace avp::mixer;

void print_stack_trace() {}

int main() {
    const char* names[] = {"t", nullptr};
    for (auto curve : {FadeCurve::Linear, FadeCurve::EaseIn, FadeCurve::EaseOut, FadeCurve::EaseInOut}) {
        const std::string progress = fadeCurveExpression(curve, "clip((t-12.345)/0.75,0,1)");
        for (double t = 12.0; t <= 13.5; t += 0.01) {
            const double x = (t - 12.345) / 0.75;
            for (bool to_a : {false, true}) {
                const std::string expr = to_a ? "1-" + progress : progress;
                double value = NAN;
                const double values[] = {t};
                assert(av_expr_parse_and_eval(&value, expr.c_str(), names, values,
                                              nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr) >= 0);
                const double expected = fadeCurveAt(curve, x);
                assert(std::fabs(value - (to_a ? 1 - expected : expected)) < 1e-12);
            }
        }
    }
}
