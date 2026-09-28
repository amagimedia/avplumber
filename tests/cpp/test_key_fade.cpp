#include "mixer/primitives/key_fade.hpp"
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>
using namespace avp::mixer;

void print_stack_trace() {}

namespace {

constexpr FadeCurve kCurves[] = {FadeCurve::Linear, FadeCurve::EaseIn, FadeCurve::EaseOut, FadeCurve::EaseInOut};
constexpr double kPeriod = 0.04;   // 25 fps
constexpr double kT0 = 10.0;       // first program frame after the command

bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }
double frameTime(int k) { return kT0 + k * kPeriod; }

void curves() {
    for (auto curve : kCurves) {
        assert(parseFadeCurve(fadeCurveName(curve)) == curve);
        // Exact ends, clamped outside [0, 1], NaN treated as the start.
        assert(fadeCurveAt(curve, 0) == 0 && fadeCurveAt(curve, 1) == 1);
        assert(fadeCurveAt(curve, -3) == 0 && fadeCurveAt(curve, 7) == 1);
        assert(fadeCurveAt(curve, std::numeric_limits<double>::quiet_NaN()) == 0);
        double previous = 0;
        for (int i = 1; i <= 1000; ++i) {
            const double y = fadeCurveAt(curve, i / 1000.0);
            assert(y >= previous && y <= 1);
            previous = y;
        }
    }
    assert(fadeCurveAt(FadeCurve::Linear, 0.25) == 0.25);
    assert(fadeCurveAt(FadeCurve::EaseIn, 0.5) == 0.25);
    assert(fadeCurveAt(FadeCurve::EaseOut, 0.5) == 0.75);
    assert(fadeCurveAt(FadeCurve::EaseInOut, 0.5) == 0.5);
    assert(near(fadeCurveAt(FadeCurve::EaseInOut, 0.25), 0.15625));
    for (const char* bad : {"", "Linear", "smooth", "ease_in", "bezier"}) {
        bool rejected = false;
        try { parseFadeCurve(bad); } catch (const Error&) { rejected = true; }
        assert(rejected);
    }

    // Linear keeps the M/E fade command byte-identical; the others stay one operand.
    const std::string p = "clip((t-12.345000)/0.750000,0,1)";
    assert(fadeCurveExpression(FadeCurve::Linear, p) == p);
    assert(fadeCurveExpression(FadeCurve::EaseIn, p) == "(st(0," + p + ");ld(0)*ld(0))");
    assert(fadeCurveExpression(FadeCurve::EaseOut, p) == "(st(0," + p + ");ld(0)*(2-ld(0)))");
    assert(fadeCurveExpression(FadeCurve::EaseInOut, p) == "(st(0," + p + ");ld(0)*ld(0)*(3-2*ld(0)))");
}

void settledStates() {
    KeyFade off, on(true);
    assert(off.level(kT0) == 0 && !off.visible() && !off.fading() && !off.target());
    assert(on.level(kT0) == 1 && on.visible() && !on.fading() && on.target());
}

// A fade of N periods changes over exactly N frames, starting on the first frame after the command.
void rampEndpoints() {
    for (auto curve : kCurves) {
        KeyFade in;
        in.retarget(true, 5 * kPeriod, curve, frameTime(0), kPeriod);
        for (int k = 0; k < 4; ++k) {
            assert(in.fading() && in.visible());
            assert(near(in.level(frameTime(k)), fadeCurveAt(curve, (k + 1) / 5.0)));
        }
        assert(in.level(frameTime(4)) == 1 && !in.fading());

        KeyFade out(true);
        out.retarget(false, 5 * kPeriod, curve, frameTime(0), kPeriod);
        for (int k = 0; k < 4; ++k) {
            // Transition progress is curved the same way in both directions.
            assert(near(out.level(frameTime(k)), 1 - fadeCurveAt(curve, (k + 1) / 5.0)));
            assert(out.visible() && !out.target());
        }
        assert(out.level(frameTime(4)) == 0 && !out.fading() && !out.visible());
    }
}

void cuts() {
    KeyFade key;
    key.retarget(true, 0, FadeCurve::EaseIn, kT0, kPeriod);
    assert(!key.fading() && key.level(kT0) == 1);
    key.retarget(false, std::numeric_limits<double>::quiet_NaN(), FadeCurve::Linear, kT0, kPeriod);
    assert(key.level(kT0) == 0 && !key.visible());

    // A cut during a fade snaps; so does a cut to the target already being faded to.
    key.retarget(true, 1.0, FadeCurve::Linear, frameTime(0), kPeriod);
    assert(near(key.level(frameTime(0)), 0.04));
    key.retarget(true, 0, FadeCurve::Linear, frameTime(1), kPeriod);
    assert(key.level(frameTime(1)) == 1 && !key.fading());
    key.retarget(false, 1.0, FadeCurve::Linear, frameTime(2), kPeriod);
    assert(near(key.level(frameTime(2)), 0.96));
    key.retarget(true, 0, FadeCurve::Linear, frameTime(3), kPeriod);
    assert(key.level(frameTime(3)) == 1);
}

void reversal() {
    KeyFade key;
    key.retarget(true, 1.0, FadeCurve::Linear, frameTime(0), kPeriod);
    for (int k = 0; k < 10; ++k) key.level(frameTime(k));
    assert(near(key.current(), 0.4));

    // Re-sending the running target does not restart the fade.
    key.retarget(true, 3.0, FadeCurve::EaseIn, frameTime(10), kPeriod);
    assert(near(key.level(frameTime(10)), 0.44));

    // Reversed: continues from the shown level at the same pace, so 0.44 takes 11 frames.
    key.retarget(false, 1.0, FadeCurve::Linear, frameTime(11), kPeriod);
    double previous = 0.44;
    for (int k = 11; k < 21; ++k) {
        const double level = key.level(frameTime(k));
        assert(near(previous - level, 0.04) && key.visible());
        previous = level;
    }
    assert(key.level(frameTime(21)) == 0 && !key.visible());

    // A curved reversal is continuous too: its first step is one curve step from the shown level.
    KeyFade eased;
    eased.retarget(true, 1.0, FadeCurve::EaseIn, frameTime(0), kPeriod);
    for (int k = 0; k < 13; ++k) eased.level(frameTime(k));
    const double shown = eased.current();
    assert(near(shown, 0.52 * 0.52));
    eased.retarget(false, 1.0, FadeCurve::EaseIn, frameTime(13), kPeriod);
    const double span = shown;   // seconds left at the full-swing pace of 1 s
    assert(near(eased.level(frameTime(13)), shown * (1 - fadeCurveAt(FadeCurve::EaseIn, kPeriod / span))));
}

// The ramp is anchored at the command; the keyer's first drawable key frame does not move it.
void startsAtCommand() {
    KeyFade key;
    key.retarget(true, 0.2, FadeCurve::Linear, frameTime(0), kPeriod);
    // Program frames 0..2 had no key frame to draw; the level still ran on.
    assert(near(key.level(frameTime(3)), 0.8));
}

void timestampJumps() {
    KeyFade key;
    key.retarget(true, 1.0, FadeCurve::Linear, frameTime(0), kPeriod);
    for (int k = 0; k < 10; ++k) key.level(frameTime(k));
    // Backwards: holds the shown level, then ramps on from it at the same pace.
    const double back = frameTime(9) - 5;
    assert(near(key.level(back), 0.4));
    assert(near(key.level(back + kPeriod), 0.44));
    assert(near(key.level(back + 2 * kPeriod), 0.48));
    // Forwards past the end: lands on the target.
    assert(key.level(back + 100) == 1 && !key.fading());
}

// Timestamps rounded to milliseconds (60 fps in a 1/1000 time base) still end a fade on its Nth frame.
void roundedTimestamps() {
    const double period = 1 / 60.0;
    auto ms = [](double t) { return std::round(t * 1000) / 1000; };
    KeyFade key;
    key.retarget(true, 30 * period, FadeCurve::Linear, kT0, period);
    for (int k = 0; k < 29; ++k) assert(key.level(ms(kT0 + k * period)) < 1);
    assert(key.level(ms(kT0 + 29 * period)) == 1);
}

void independentKeys() {
    std::vector<KeyFade> keys(2);
    keys[0].retarget(true, 0.2, FadeCurve::Linear, frameTime(0), kPeriod);
    for (int k = 0; k < 3; ++k)
        for (auto& key : keys) key.level(frameTime(k));
    keys[1].retarget(true, 0.4, FadeCurve::EaseOut, frameTime(3), kPeriod);
    assert(near(keys[0].level(frameTime(3)), 0.8));
    assert(near(keys[1].level(frameTime(3)), fadeCurveAt(FadeCurve::EaseOut, 0.1)));
    assert(keys[0].level(frameTime(4)) == 1);
    assert(near(keys[1].level(frameTime(4)), fadeCurveAt(FadeCurve::EaseOut, 0.2)));
}

}  // namespace

int main() {
    curves();
    settledStates();
    rampEndpoints();
    cuts();
    reversal();
    startsAtCommand();
    timestampJumps();
    roundedTimestamps();
    independentKeys();
}
