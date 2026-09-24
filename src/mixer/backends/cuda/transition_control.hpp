#pragma once
#include "../../transition_control.hpp"

namespace avp::mixer::cuda {
TransitionCommand fadeCommand(const FadeRequest& request);
}
