#include "mixer/transition_control.hpp"
#include <cassert>

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
    bool rejected = false;
    try { avp::mixer::transitionControl("vulkan"); }
    catch (const std::exception&) { rejected = true; }
    assert(rejected);
}
