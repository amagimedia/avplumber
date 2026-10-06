#include "mixer/transition_control.hpp"
#include "mixer/backends/cuda/transition_control.hpp"
#include <cassert>
#include <string>
#include <utility>
#include <vector>

void print_stack_trace() {}

namespace {
Parameters filterCommand(const std::string& command, const std::string& argument) {
    return {{"target", "transition_cuda"}, {"command", command}, {"argument", argument}};
}
}

int main() {
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
    bool rejected = false;
    try { avp::mixer::transitionControl("vulkan"); }
    catch (const std::exception&) { rejected = true; }
    assert(rejected);
}
