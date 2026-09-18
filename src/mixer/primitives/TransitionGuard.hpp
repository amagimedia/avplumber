#pragma once
// Generation/mode check for transition workers and the abort-on-exception guard
// used while a transition is being prepared under the control mutex.
#include "MixerState.hpp"
#include <functional>
#include <memory>

namespace avp::mixer {

class TransitionGuard {
    std::function<void()> abort_;
    bool active_ = true;

public:
    explicit TransitionGuard(std::function<void()> abort)
        : abort_(std::move(abort)) {}

    ~TransitionGuard() {
        if (active_) abort_();
    }

    void release() {
        active_ = false;
    }
};

inline bool transitionIsCurrent(const std::shared_ptr<MixerState>& state,
                         uint64_t generation,
                         MixerState::TransitionMode mode) {
    return state->transition_generation.load(std::memory_order_acquire) == generation &&
           state->transition_mode.load(std::memory_order_acquire) == mode;
}

}
