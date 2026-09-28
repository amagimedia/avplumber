// The M/E fade hands the curve to FFmpeg as an expression; it must evaluate to
// what fadeCurveAt computes for the keyer. A dip's expression must match
// dipCurveAt the same way and land exactly on 1/2, the colour, at its midpoint.
#include "mixer/primitives/fade_curve.hpp"
extern "C" {
#include <libavutil/eval.h>
}
#include <cassert>
#include <cmath>
#include <string>
using namespace avp::mixer;

void print_stack_trace() {}

namespace {
double evaluate(const std::string& expr, double t) {
    const char* names[] = {"t", nullptr};
    const double values[] = {t};
    double value = NAN;
    assert(av_expr_parse_and_eval(&value, expr.c_str(), names, values,
                                  nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr) >= 0);
    return value;
}
}

int main() {
    for (auto curve : {FadeCurve::Linear, FadeCurve::EaseIn, FadeCurve::EaseOut, FadeCurve::EaseInOut}) {
        const std::string progress = fadeCurveExpression(curve, "clip((t-12.345)/0.75,0,1)");
        const std::string dip = dipCurveExpression(curve, "clip((t-12.345)/0.75,0,1)");
        for (double t = 12.0; t <= 13.5; t += 0.01) {
            const double x = (t - 12.345) / 0.75;
            for (bool to_a : {false, true}) {
                const double expected = fadeCurveAt(curve, x);
                assert(std::fabs(evaluate(to_a ? "1-" + progress : progress, t) -
                                 (to_a ? 1 - expected : expected)) < 1e-12);
                const double dipped = dipCurveAt(curve, x);
                assert(std::fabs(evaluate(to_a ? "1-" + dip : dip, t) - (to_a ? 1 - dipped : dipped)) < 1e-12);
            }
        }
        // Each half runs the whole curve: 0 and 1 at the ends, exactly 1/2 at the midpoint
        // (t = 12.25 makes the progress exactly 0.5), in both directions.
        const std::string mid = dipCurveExpression(curve, "clip((t-12)/0.5,0,1)");
        for (bool to_a : {false, true}) {
            assert(evaluate(to_a ? "1-" + mid : mid, 12.25) == 0.5);
            assert(evaluate(to_a ? "1-" + mid : mid, 12.0) == (to_a ? 1.0 : 0.0));
            assert(evaluate(to_a ? "1-" + mid : mid, 12.5) == (to_a ? 0.0 : 1.0));
        }
        assert(dipCurveAt(curve, 0.5) == 0.5);
        assert(dipCurveAt(curve, 0.25) == 0.5 * fadeCurveAt(curve, 0.5));
        assert(dipCurveAt(curve, 0.75) == 0.5 + 0.5 * fadeCurveAt(curve, 0.5));
    }
}
