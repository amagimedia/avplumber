#pragma once

#include <chrono>
#include <optional>

// Scoped per-thread context also handles commands executed through the Python
// API, nested command scripts, and concurrent TCP clients without sharing clocks.
class CommandTiming {
public:
    using Clock = std::chrono::steady_clock;
private:
    inline static thread_local std::optional<Clock::time_point> received_;
    std::optional<Clock::time_point> previous_;
public:
    explicit CommandTiming(Clock::time_point received) : previous_(received_) { received_ = received; }
    ~CommandTiming() { received_ = previous_; }
    CommandTiming(const CommandTiming&) = delete;
    CommandTiming& operator=(const CommandTiming&) = delete;
    static Clock::time_point received() { return received_.value_or(Clock::now()); }
};
