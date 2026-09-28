#include "mixer/transition_control.hpp"
#include <cassert>
#include <string>
#include <utility>

void print_stack_trace() {}

int main() {
    const auto control = avp::mixer::transitionControl("cuda");
    for (bool to_a : {false, true}) {
        const auto command = control({12345, 0.75, to_a});
        assert(command.key == "filter_command");
        assert(command.value == Parameters({
            {"target", "transition_cuda"}, {"command", "alpha"},
            {"argument", to_a ? "1-clip((t-12.345000)/0.750000,0,1)"
                              : "clip((t-12.345000)/0.750000,0,1)"},
        }));
    }
    // Curves wrap the same clip() progress; "1-" applies to the curved value,
    // so a fade toward A mirrors the fade toward B. Evaluation of these
    // expressions is covered by test_fade_curve_expression.
    const std::string clip = "clip((t-12.345000)/0.750000,0,1)";
    const std::pair<avp::mixer::FadeCurve, std::string> shaped[] = {
        {avp::mixer::FadeCurve::Linear, clip},
        {avp::mixer::FadeCurve::EaseIn, "(st(0," + clip + ");ld(0)*ld(0))"},
        {avp::mixer::FadeCurve::EaseOut, "(st(0," + clip + ");ld(0)*(2-ld(0)))"},
        {avp::mixer::FadeCurve::EaseInOut, "(st(0," + clip + ");ld(0)*ld(0)*(3-2*ld(0)))"},
    };
    for (const auto& [curve, progress] : shaped) {
        for (bool to_a : {false, true}) {
            const auto command = control({12345, 0.75, to_a, curve});
            assert(command.value.at("argument") == (to_a ? "1-" + progress : progress));
        }
    }
    bool rejected = false;
    try { avp::mixer::transitionControl("vulkan"); }
    catch (const std::exception&) { rejected = true; }
    assert(rejected);
}
