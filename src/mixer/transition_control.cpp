#include "transition_control.hpp"
#include "backends/cuda/transition_control.hpp"

namespace avp::mixer {

TransitionControl transitionControl(const std::string& backend) {
    if (backend == "cuda") return cuda::fadeCommand;
    throw Error("mixer: unsupported backend: " + backend);
}

} // namespace avp::mixer
