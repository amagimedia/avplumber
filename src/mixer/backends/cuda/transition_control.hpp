#pragma once
#include "../../transition_control.hpp"

namespace avp::mixer::cuda {
std::vector<TransitionCommand> fadeCommand(const FadeRequest& request);
}
