#include "transition_control.hpp"

namespace avp::mixer::cuda {

TransitionCommand fadeCommand(const FadeRequest& request) {
    const std::string progress = "clip((t-" + std::to_string(request.start_ms / 1000.0) +
        ")/" + std::to_string(request.duration_sec) + ",0,1)";
    return {"filter_command", {
        {"target", "transition_cuda"},
        {"command", "alpha"},
        {"argument", request.destination_is_a ? "1-" + progress : progress},
    }};
}

} // namespace avp::mixer::cuda
