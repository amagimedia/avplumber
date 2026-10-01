// The M/E fade hands the curve to FFmpeg as an expression; it must evaluate to
// what fadeCurveAt computes for the keyer. A dip's expression must match
// dipCurveAt below the same way, and some frame of the 1/fps grid must get alpha
// 1/2 exactly, the colour alone, whatever the phase of the millisecond start.
#include "mixer/primitives/fade_curve.hpp"
extern "C" {
#include <libavutil/eval.h>
#include <libavutil/rational.h>
}
#include <cassert>
#include <cmath>
#include <cstdint>
#include <string>
using namespace avp::mixer;

void print_stack_trace() {}

namespace {
const char* const kNames[] = {"t", nullptr};

// The dip's y(x) as fade_curve.hpp defines it (dipHoldEnd), on the host.
double dipCurveAt(FadeCurve curve, double x, double hold) {
    x = std::min(1.0, std::max(0.0, x));
    const double hi = dipHoldEnd(hold), lo = 1 - hi;
    if (x < lo) return 0.5 * fadeCurveAt(curve, x / lo);
    return x < hi ? 0.5 : 0.5 + 0.5 * fadeCurveAt(curve, (x - hi) / lo);
}

double evaluate(const std::string& expr, double t) {
    const double values[] = {t};
    double value = NAN;
    assert(av_expr_parse_and_eval(&value, expr.c_str(), kNames, values,
                                  nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr) >= 0);
    return value;
}

// Frames within two periods of `mid` that transition_cuda shows as the colour
// alone: it computes t = pts * av_q2d(time_base), so frame k is at k * period,
// and takes alpha as a float, so a frame a rounding error off the hold edge
// still gets exactly 1/2.
int colourFrames(const std::string& expr, AVRational period, double mid) {
    AVExpr* parsed = nullptr;
    const int ret = av_expr_parse(&parsed, expr.c_str(), kNames, nullptr, nullptr, nullptr, nullptr, 0, nullptr);
    assert(ret >= 0);
    int count = 0;
    const int64_t centre = std::llround(mid / av_q2d(period));
    for (int64_t k = centre - 2; k <= centre + 2; ++k) {
        const double t = k * av_q2d(period);
        count += static_cast<float>(av_expr_eval(parsed, &t, nullptr)) == 0.5f;
    }
    av_expr_free(parsed);
    return count;
}
}

int main() {
    const double hold = 0.04 / 0.75;  // 25 fps, 0.75 s
    for (auto curve : {FadeCurve::Linear, FadeCurve::EaseIn, FadeCurve::EaseOut, FadeCurve::EaseInOut}) {
        const std::string progress = fadeCurveExpression(curve, "clip((t-12.345)/0.75,0,1)");
        const std::string dip = dipCurveExpression(curve, "clip((t-12.345)/0.75,0,1)", hold);
        for (double t = 12.0; t <= 13.5; t += 0.01) {
            const double x = (t - 12.345) / 0.75;
            for (bool to_a : {false, true}) {
                const double expected = fadeCurveAt(curve, x);
                assert(std::fabs(evaluate(to_a ? "1-" + progress : progress, t) -
                                 (to_a ? 1 - expected : expected)) < 1e-12);
                const double dipped = dipCurveAt(curve, x, hold);
                assert(std::fabs(evaluate(to_a ? "1-" + dip : dip, t) - (to_a ? 1 - dipped : dipped)) < 1e-12);
            }
        }
        // Each half runs the whole curve: 0 and 1 at the ends, exactly 1/2 over the
        // hold (t 12.23 to 12.27 at 25 fps), in both directions.
        const std::string mid = dipCurveExpression(curve, "clip((t-12)/0.5,0,1)", 0.04 / 0.5);
        for (bool to_a : {false, true}) {
            for (double t : {12.235, 12.25, 12.265})
                assert(evaluate(to_a ? "1-" + mid : mid, t) == 0.5);
            assert(evaluate(to_a ? "1-" + mid : mid, 12.0) == (to_a ? 1.0 : 0.0));
            assert(evaluate(to_a ? "1-" + mid : mid, 12.5) == (to_a ? 0.0 : 1.0));
        }
        // Hold 0 is the bare midpoint: each half is C over half the dip.
        assert(dipCurveAt(curve, 0.5, 0) == 0.5);
        assert(dipCurveAt(curve, 0.25, 0) == 0.5 * fadeCurveAt(curve, 0.5));
        assert(dipCurveAt(curve, 0.75, 0) == 0.5 + 0.5 * fadeCurveAt(curve, 0.5));
        assert(dipCurveAt(curve, 1, hold) == 1 && dipCurveAt(curve, 1, 2) == 1 && dipCurveAt(curve, 1, NAN) == 1);

        // A one-period hold shows the colour alone on a frame for every millisecond
        // start phase and rate; two only when frames sit on both edges.
        for (AVRational period : {AVRational{1, 25}, AVRational{1, 30}, AVRational{1, 50},
                                  AVRational{1, 60}, AVRational{1001, 60000}}) {
            for (double duration : {0.1, 0.3, 0.5, 1.0}) {
                for (int start_ms = 1000; start_ms < 1200; ++start_ms) {
                    const std::string clip = "clip((t-" + std::to_string(start_ms / 1000.0) + ")/" +
                                             std::to_string(duration) + ",0,1)";
                    const std::string grid = dipCurveExpression(curve, clip, av_q2d(period) / duration);
                    for (bool to_a : {false, true}) {
                        const int frames = colourFrames(to_a ? "1-" + grid : grid, period,
                                                        start_ms / 1000.0 + duration / 2);
                        assert(frames >= 1 && frames <= 2);
                    }
                }
            }
        }
    }
}
