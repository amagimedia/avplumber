#pragma once

#include "FrameRate.hpp"
#include <algorithm>
#include <deque>
#include <optional>
#include <vector>

namespace avp::mixer {

// Arrival-stamped sources can paint sporadically while opening. Establish a
// phase from the first second, then keep it fixed during normal playout: using
// each new arrival as a fresh presentation time would reintroduce jitter.
class Cadence {
public:
    struct Position {
        int64_t index;
        int64_t phase_shift = 0;
        bool discontinuity = false;
    };

private:
    FrameRate rate_;
    int64_t tolerance_;
    std::optional<int64_t> next_;
    int64_t observations_ = 0;
    std::deque<int64_t> phase_errors_;

public:
    Cadence(FrameRate rate, int64_t latency_ns)
        : rate_(rate), tolerance_(std::max<int64_t>(2, rate.nearestIndex(latency_ns) + 1)) {}

    void advance(int64_t slots) {
        if (next_) *next_ += slots;
        for (auto &error : phase_errors_) error -= rate_.time(slots);
    }

    Position observe(int64_t timestamp_ns) {
        const auto stamped = rate_.nearestIndex(timestamp_ns);
        Position result{stamped};
        if (!next_) next_ = stamped;
        const auto error = stamped - *next_;
        if (error > tolerance_ || error < -tolerance_) {
            next_ = stamped;
            observations_ = 0;
            phase_errors_.clear();
            result.discontinuity = true;
        }
        const auto startup_frames = std::max<int64_t>(8, rate_.nearestIndex(1000000000));
        if (observations_ < startup_frames) {
            phase_errors_.push_back(timestamp_ns - rate_.time(*next_));
            if (phase_errors_.size() > 8) phase_errors_.pop_front();
            if (++observations_ == startup_frames) {
                std::vector<int64_t> sorted(phase_errors_.begin(), phase_errors_.end());
                std::sort(sorted.begin(), sorted.end());
                const auto median = sorted[sorted.size() / 2];
                // Leave sub-frame phase differences alone; changing them is
                // unnecessary and could turn normal delivery jitter into a skip.
                if (median >= rate_.time(1) || median <= -rate_.time(1)) {
                    result.phase_shift = rate_.nearestIndex(median);
                    *next_ += result.phase_shift;
                }
                phase_errors_.clear();
            }
        }
        result.index = (*next_)++;
        return result;
    }
};

} // namespace avp::mixer
