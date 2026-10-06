#pragma once
// Abort-on-exception guard used while a transition is prepared under the control mutex.
#include <functional>
#include <utility>

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

}
