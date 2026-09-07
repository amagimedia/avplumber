#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>

namespace avp::mixer {

// Caller serializes this state with output publication, so an interrupt captures
// the last committed picture, including a partially rendered transition.
template<class Frame> class Snapshot {
    std::optional<Frame> displayed_;
    std::optional<Frame> frozen_;
    int slot_ = -1;
    uint64_t generation_ = 0;
    std::optional<int64_t> release_pts_;
    bool require_snapshot_ = true;
    bool holding_ = false;

public:
    void presented(const Frame& frame) { displayed_ = frame; }
    uint64_t capture(int slot) {
        if (!displayed_) throw std::runtime_error("mixer output has no picture yet");
        frozen_ = displayed_;
        slot_ = slot;
        holding_ = true;
        release_pts_.reset();
        require_snapshot_ = true;
        return ++generation_;
    }
    void arm(int64_t pts, bool require_snapshot = true) {
        release_pts_ = pts;
        require_snapshot_ = require_snapshot;
    }
    bool canRelease(int64_t pts, uint64_t frame_generation) const {
        return holding_ && release_pts_ && pts >= *release_pts_ &&
            (!require_snapshot_ || frame_generation == generation_);
    }
    void release() { holding_ = false; }
    void finish() { slot_ = -1; }
    bool holding() const { return holding_; }
    uint64_t generation() const { return generation_; }
    const std::optional<Frame>& frozen() const { return frozen_; }
    bool replaces(int slot) const { return slot_ == slot && frozen_.has_value(); }
};

} // namespace avp::mixer
