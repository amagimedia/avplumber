#include "transition_control.hpp"

namespace avp::mixer::cuda {

// The filter keeps its mode between transitions, so every fade names its own:
// a crossfade after a dip must not dip.
std::vector<TransitionCommand> fadeCommand(const FadeRequest& request) {
    std::vector<TransitionCommand> commands;
    const auto send = [&](const char* command, const std::string& argument) {
        commands.push_back({"filter_command", {
            {"target", "transition_cuda"}, {"command", command}, {"argument", argument}}});
    };
    const std::string clip = "clip((t-" + std::to_string(request.start_ms / 1000.0) +
        ")/" + std::to_string(request.duration_sec) + ",0,1)";
    std::string progress;
    if (request.dip) {
        const auto c = canvasCodes(*request.dip, request.transfer);
        send("color", std::to_string(c[0]) + ":" + std::to_string(c[1]) + ":" + std::to_string(c[2]));
        send("mode", "dip");
        progress = dipCurveExpression(request.curve, clip, request.dip_hold);
    } else {
        send("mode", "fade");
        progress = fadeCurveExpression(request.curve, clip);
    }
    send("alpha", request.destination_is_a ? "1-" + progress : progress);
    return commands;
}

} // namespace avp::mixer::cuda
