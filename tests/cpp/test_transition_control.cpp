#include "mixer/transition_control.hpp"
#include "mixer/backends/cuda/transition_control.hpp"
#include <cassert>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

void print_stack_trace() {}

extern "C" {
#include <libavutil/eval.h>
}

namespace {
Parameters filterCommand(const std::string& command, const std::string& argument) {
    return {{"target", "transition_cuda"}, {"command", command}, {"argument", argument}};
}

// The filter's value of an alpha expression at frame time `t` (seconds).
double alphaAt(const std::string& expression, double t) {
    const char* names[] = {"t", nullptr};
    const double values[] = {t};
    double result = NAN;
    const int ret = av_expr_parse_and_eval(&result, expression.c_str(), names, values,
                                           nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr);
    assert(ret >= 0);
    return result;
}
}

// The lever's ramp: from where the mix is to the lever's new position, both ways, as the filter
// evaluates it, then held there.
void leverRamp() {
    using avp::mixer::rampTo;
    using avp::mixer::requestProgress;
    const auto control = avp::mixer::transitionControl("cuda");
    avp::mixer::FadeRequest held{1000, 1.0, false};
    held.time_floor_ms = held.time_ceiling_ms = 1300;   // held at 0.3
    // The ramp's start is whole milliseconds: progress may be off by 0.5 ms of the ramp, under 0.5 % here.
    const auto near = [](double a, double b) { return std::fabs(a - b) < 5e-3; };
    const auto up = rampTo(held, 0.5, 2000, 2033);
    assert(near(requestProgress(up, 1900), 0.3) && near(requestProgress(up, 2000), 0.3));
    assert(near(requestProgress(up, 2016), 0.4) && near(requestProgress(up, 2033), 0.5) && near(requestProgress(up, 9000), 0.5));
    const auto down = rampTo(up, 0.2, 3000, 3050);
    assert(down.duration_sec < 0);
    assert(near(requestProgress(down, 3000), 0.5) && near(requestProgress(down, 3025), 0.35) && near(requestProgress(down, 4000), 0.2));
    for (const double t : {2.9, 3.0, 3.025, 3.05, 4.0}) {
        const double alpha = alphaAt(control(down).back().value.at("argument"), t);
        assert(near(alpha, requestProgress(down, int64_t(std::llround(t * 1000)))));
    }
    const auto still = rampTo(down, 0.2, 5000, 5033);   // the same position again: held, no ramp
    assert(near(requestProgress(still, 5000), 0.2) && near(requestProgress(still, 6000), 0.2));
}

// The schedule every consumer of a lever take evaluates (alpha, keys, a scrubbed clip) in ns, its
// run-on when AUTO takes over, and where the lever's end lands.
void leverSchedule() {
    using avp::mixer::FadeRequest;
    using avp::mixer::rampAt;
    using avp::mixer::rampTimesNs;
    using avp::mixer::requestProgress;
    FadeRequest held{1000, 1.0, false};
    held.time_floor_ms = held.time_ceiling_ms = 1300;
    for (const auto& ramp : {avp::mixer::rampTo(held, 0.9, 2000, 2016), avp::mixer::rampTo(held, 0.1, 2000, 2016)})
        for (int64_t t_ms = 1990; t_ms <= 2030; ++t_ms)   // the duration in whole ns: 1e-6 of a ramp
            assert(std::fabs(rampAt(rampTimesNs(ramp), t_ms * 1000000) - requestProgress(ramp, t_ms)) < 1e-6);
    const FadeRequest unclamped{1000, 2.0, false};
    const auto ramp = rampTimesNs(unclamped);
    assert(ramp.start_ns == 1000000000 && ramp.duration_ns == 2000000000 &&
           ramp.floor_ns == std::numeric_limits<int64_t>::min() && ramp.ceiling_ns == std::numeric_limits<int64_t>::max());

    // AUTO runs the lever's take on from where it is, at a whole take's rate, without a ceiling.
    const auto lever = avp::mixer::rampTo(held, 0.6, 2000, 2016);
    const auto run = avp::mixer::runOn(lever, 2100, 0.5);
    assert(std::fabs(requestProgress(run, 2100) - requestProgress(lever, 2100)) < 1e-3);   // no jump
    assert(std::fabs(requestProgress(run, 2050) - requestProgress(lever, 2100)) < 1e-3);   // older frames hold
    assert(std::fabs(requestProgress(run, 2200) - (0.6 + 0.1 / 0.5)) < 1e-3);              // 1/duration
    assert(requestProgress(run, 2300) == 1.0 && !run.time_ceiling_ms && run.duration_sec == 0.5);

    // An end lands on the ramp's last frame by time, where its progress may stop short of the end.
    FadeRequest near_end{1000, 1.0, false};
    near_end.start_ms = 1000;
    near_end.time_floor_ms = 1000;
    near_end.time_ceiling_ms = 1980;   // progress 0.98 at the ceiling
    assert(!avp::mixer::leverEndLanded(near_end, 1979999999) && avp::mixer::leverEndLanded(near_end, 1980000000));
    near_end.duration_sec = -1;        // moving back: 0.02 at the ceiling
    near_end.start_ms = 2000;
    assert(std::fabs(requestProgress(near_end, 1980) - 0.02) < 1e-9 && avp::mixer::leverEndLanded(near_end, 1980000000));
    assert(!avp::mixer::leverEndLanded(unclamped, std::numeric_limits<int64_t>::max()));   // AUTO's: no lever end
}

// The time a lever position takes to reach the picture, at 60 fps.
void leverSpan() {
    using avp::mixer::leverSpanNs;
    const int64_t frame = 16666667, now = 10000000000;
    assert(leverSpanNs(std::nullopt, now, 0, frame) == frame);                       // the first position
    assert(leverSpanNs(std::nullopt, now, now - 20000000, frame) == 20000000);       // streamed
    assert(leverSpanNs(std::nullopt, now, now - 1000000, frame) == frame);           // never under a frame
    assert(leverSpanNs(std::nullopt, now, now - 2000000000, frame) == 100000000);    // seldom: 100 ms at most
    assert(leverSpanNs(25.0, now, now - 2000000000, frame) == 25000000);             // the controller's spacing
    assert(leverSpanNs(4.0, now, 0, frame) == frame);                                // one frame at least
    assert(leverSpanNs(500.0, now, 0, frame) == 200000000);                          // 200 ms at most
}

int main() {
    leverRamp();
    leverSchedule();
    leverSpan();
    const auto control = avp::mixer::transitionControl("cuda");
    const std::string clip = "clip((t-12.345000)/0.750000,0,1)";
    // Every crossfade names its mode first: the filter keeps a dip's mode until told otherwise.
    for (bool to_a : {false, true}) {
        const auto commands = control({12345, 0.75, to_a});
        assert(commands.size() == 2);
        for (const auto& command : commands) assert(command.key == "filter_command");
        assert(commands[0].value == filterCommand("mode", "fade"));
        assert(commands[1].value == filterCommand("alpha", to_a ? "1-" + clip : clip));
    }
    // Curves wrap the same clip() progress; "1-" applies to the curved value,
    // so a fade toward A mirrors the fade toward B. Evaluation of these
    // expressions is covered by test_fade_curve_expression.
    const std::pair<avp::mixer::FadeCurve, std::string> shaped[] = {
        {avp::mixer::FadeCurve::Linear, clip},
        {avp::mixer::FadeCurve::EaseIn, "(st(0," + clip + ");ld(0)*ld(0))"},
        {avp::mixer::FadeCurve::EaseOut, "(st(0," + clip + ");ld(0)*(2-ld(0)))"},
        {avp::mixer::FadeCurve::EaseInOut, "(st(0," + clip + ");ld(0)*ld(0)*(3-2*ld(0)))"},
    };
    for (const auto& [curve, progress] : shaped) {
        for (bool to_a : {false, true}) {
            const auto commands = control({12345, 0.75, to_a, curve});
            assert(commands.back().value.at("argument") == (to_a ? "1-" + progress : progress));
        }
    }
    // A dip sets its colour, as canvas codes (black: Y 16, Cb Cr 128 on any canvas), then the
    // mode, then alpha, which runs the curve once per half around the request's hold.
    const std::string linear_dip = avp::mixer::dipCurveExpression(avp::mixer::FadeCurve::Linear, clip, 0.04 / 0.75);
    for (bool to_a : {false, true}) {
        const auto commands = control({12345, 0.75, to_a, avp::mixer::FadeCurve::Linear,
                                       std::array<uint8_t, 3>{0, 0, 0}, 0.04 / 0.75});
        assert(commands.size() == 3);
        assert(commands[0].value == filterCommand("color", "16.000000:128.000000:128.000000"));
        assert(commands[1].value == filterCommand("mode", "dip"));
        assert(commands[2].value == filterCommand("alpha", to_a ? "1-" + linear_dip : linear_dip));
    }
    // The colour is converted for the request's canvas: white on HLG is graphics white.
    const auto white = avp::mixer::cuda::canvasCodes({255, 255, 255}, AVCOL_TRC_ARIB_STD_B67);
    const auto eased = control({12345, 0.75, false, avp::mixer::FadeCurve::EaseIn,
                                std::array<uint8_t, 3>{255, 255, 255}, 0, AVCOL_TRC_ARIB_STD_B67});
    assert(eased[0].value.at("argument") ==
           std::to_string(white[0]) + ":" + std::to_string(white[1]) + ":" + std::to_string(white[2]));
    assert(eased[2].value.at("argument") ==
           "(st(1," + clip + ");if(lt(ld(1),0.5),0.5*((st(0,ld(1)/0.5);ld(0)*ld(0))),"
           "if(lt(ld(1),0.5),0.5,0.5+0.5*((st(0,(ld(1)-0.5)/0.5);ld(0)*ld(0))))))");
    // Pause: the clock stops at the ceiling, so the mix holds there while frames keep coming.
    // Resume: a later start shifted by the pause, clamped from below at the resume time.
    {
        avp::mixer::FadeRequest held{12345, 0.75, false};
        held.time_ceiling_ms = 12600;
        const std::string paused = control(held).back().value.at("argument");
        assert(paused == "clip((min(t,12.600000)-12.345000)/0.750000,0,1)");
        assert(std::fabs(alphaAt(paused, 12.5) - 0.155 / 0.75) < 1e-9);    // before the hold: running
        assert(std::fabs(alphaAt(paused, 12.6) - 0.255 / 0.75) < 1e-9);
        assert(std::fabs(alphaAt(paused, 30.0) - 0.255 / 0.75) < 1e-9);    // long after: held
        // Resumed 2 s later: start shifted by 2 s, continuing from the held position.
        avp::mixer::FadeRequest resumed{14345, 0.75, false};
        resumed.time_floor_ms = 14600;
        const std::string running = control(resumed).back().value.at("argument");
        assert(running == "clip((max(t,14.600000)-14.345000)/0.750000,0,1)");
        assert(std::fabs(alphaAt(running, 14.6) - alphaAt(paused, 12.6)) < 1e-9);  // no jump
        assert(std::fabs(alphaAt(running, 14.0) - alphaAt(paused, 12.6)) < 1e-9);  // late frames too
        assert(alphaAt(running, 15.2) == 1.0);                                      // ends after the remainder
        // Toward slot A the same clock is mirrored.
        resumed.destination_is_a = true;
        assert(control(resumed).back().value.at("argument") == "1-" + running);
    }
    bool rejected = false;
    try { avp::mixer::transitionControl("vulkan"); }
    catch (const std::exception&) { rejected = true; }
    assert(rejected);
}
