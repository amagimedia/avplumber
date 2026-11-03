#pragma once
// Destroys values on a thread of its own. The last reference to a source's frames can own its
// decoder and surface pool, and a compositor that dropped them would miss its next ticks. (A
// decoder that was parked at its close, see DecoderParking, costs nothing here: its pool and
// session stay alive, and what a compositor drops is a surface index going back to it.)
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include "../util.hpp"

namespace avp::mixer {

class Retirement {
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<std::function<void()>> jobs_;
    bool stopping_ = false;
    std::thread worker_;

    void run();

public:
    Retirement(): worker_([this] { run(); }) {}
    ~Retirement() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        changed_.notify_all();
        worker_.join();
    }
    Retirement(const Retirement &) = delete;
    Retirement &operator=(const Retirement &) = delete;

    /// Takes `value` and destroys it later, in the order retired.
    template<class T> void retire(T value) {
        auto held = std::make_shared<T>(std::move(value));   // std::function needs a copyable job
        {
            std::lock_guard<std::mutex> lock(mutex_);
            jobs_.push_back([held = std::move(held)]() mutable { held.reset(); });
        }
        changed_.notify_one();
    }
};

inline void Retirement::run() {
    set_thread_name("frame_retirement");
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        changed_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
        if (jobs_.empty()) return;
        auto job = std::move(jobs_.front());
        jobs_.pop_front();
        lock.unlock();
        job();
        job = nullptr;   // what it held goes here, outside the lock
        lock.lock();
    }
}

} // namespace avp::mixer
