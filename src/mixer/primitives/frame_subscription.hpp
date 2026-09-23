#pragma once
#include "../../instance_shared.hpp"
#include "TickGrid.hpp"
#include <mutex>
#include <optional>

namespace avp::mixer {

// The consumer closes delivery before draining its edge. Producers never wait
// for that consumer, including when it is changing subscriptions or stopping.
class FrameSubscription : public InstanceShared<FrameSubscription> {
    std::mutex mutex_;
    bool enabled_ = false;
    bool closed_ = false;
    std::optional<TickGrid> rate_;
    std::optional<int64_t> last_tick_;
public:
    void configure(av::Rational fps) {
        std::lock_guard<std::mutex> lock(mutex_);
        rate_.emplace(fps);
        enabled_ = closed_ = false;
        last_tick_.reset();
    }
    void enable(bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (enabled_ != enabled) last_tick_.reset();
        enabled_ = enabled && !closed_;
    }
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        enabled_ = false;
        closed_ = true;
    }
    template<class Frame, class Publish> void publish(const Frame &frame, Publish send) {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock || !enabled_) return;
        if (rate_ && frame.pts().isValid()) {
            const auto tick = rate_->atOrBefore(frame.pts().timestamp({1, 1000000000}));
            if (last_tick_ && tick == *last_tick_) return;
            last_tick_ = tick;
        }
        send(); // only a nonblocking enqueue; no GPU work or waits under this lock
    }
};

}
