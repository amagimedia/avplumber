#include "transition_control.hpp"

namespace avp::mixer::cuda {

TransitionCommand fadeCommand(const FadeRequest& request) {
    // Linear leaves the clip() term untouched, so the default command is the
    // same string it was before curves existed.
    const std::string progress = fadeCurveExpression(request.curve,
        "clip((t-" + std::to_string(request.start_ms / 1000.0) +
        ")/" + std::to_string(request.duration_sec) + ",0,1)");
    return {"filter_command", {
        {"target", "transition_cuda"},
        {"command", "alpha"},
        {"argument", request.destination_is_a ? "1-" + progress : progress},
    }};
}

} // namespace avp::mixer::cuda
