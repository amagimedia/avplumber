#pragma once
#include "../../instance_shared.hpp"
#include "TickGrid.hpp"
#include <atomic>
#include <mutex>
#include <optional>

namespace avp::mixer {

// The consumer closes delivery before draining its edge. Producers never wait
// for that consumer, including when it is changing subscriptions or stopping.
class FrameSubscription : public InstanceShared<FrameSubscription> {
    std::mutex mutex_;
    std::atomic<bool> enabled_{false};
    bool closed_ = false;
    std::optional<TickGrid> rate_;
    std::optional<int64_t> last_tick_;
public:
    bool enabled() const { return enabled_.load(std::memory_order_relaxed); }
    void configure(av::Rational fps, bool rate_limit = true) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (rate_limit) rate_.emplace(fps);
        else rate_.reset();
        enabled_.store(false, std::memory_order_relaxed);
        closed_ = false;
        last_tick_.reset();
    }
    void enable(bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (this->enabled() != enabled) last_tick_.reset();
        enabled_.store(enabled && !closed_, std::memory_order_relaxed);
    }
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        enabled_.store(false, std::memory_order_relaxed);
        closed_ = true;
    }
    template<class Frame, class Publish> void publish(const Frame &frame, Publish send) {
        // Most source/AUX pairs are inactive. Avoid touching their mutex; the
        // locked recheck still makes disable/close acknowledge all prior sends.
        if (!enabled()) return;
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock || !enabled()) return;
        if (rate_ && frame.pts().isValid()) {
            const auto tick = rate_->atOrBefore(frame.pts().timestamp({1, 1000000000}));
            if (last_tick_ && tick == *last_tick_) return;
            last_tick_ = tick;
        }
        send(); // only a nonblocking enqueue; no GPU work or waits under this lock
    }
};

}
