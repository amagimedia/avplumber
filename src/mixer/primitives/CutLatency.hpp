#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace avp::mixer {

// PTS identifies the picture; steady_clock measures elapsed real time. Never
// subtract a media timestamp from a command timestamp to measure a cut.
class CutLatency {
public:
    using Clock = std::chrono::steady_clock;
    struct Sample {
        uint64_t id = 0;
        std::string scene;
        bool previewed = false;
        std::string state = "waiting";
        std::optional<double> milliseconds;
        std::optional<int64_t> encoded_pts;
    };
    struct Snapshot {
        Sample direct, previewed;
        std::vector<Sample> direct_recent, previewed_recent;
    };

private:
    mutable std::mutex mutex_;
    Snapshot samples_;
    uint64_t next_id_ = 0;
    std::optional<Sample> pending_;
    Clock::time_point received_;
    int target_input_ = -1;
    bool armed_ = false;
    std::map<int64_t, uint64_t> input_pts_;
    std::optional<int64_t> last_encoder_input_pts_;

    void publish() {
        (pending_->previewed ? samples_.previewed : samples_.direct) = *pending_;
    }

    void cancelLocked(const char* reason) {
        if (pending_) {
            pending_->state = reason;
            publish();
            pending_.reset();
        }
        armed_ = false;
        input_pts_.clear();
    }

public:
    void begin(std::string scene, bool previewed, int target_input, Clock::time_point received) {
        std::lock_guard<std::mutex> lock(mutex_);
        cancelLocked("superseded");
        received_ = received;
        target_input_ = target_input;
        pending_ = Sample{++next_id_, std::move(scene), previewed, "pending", std::nullopt, std::nullopt};
        publish();
    }

    void arm() {
        std::lock_guard<std::mutex> lock(mutex_);
        armed_ = pending_.has_value();
    }

    void cancel(const char* reason = "interrupted") {
        std::lock_guard<std::mutex> lock(mutex_);
        cancelLocked(reason);
    }

    uint64_t tokenForInput(int input, bool fallback) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pending_ && armed_ && !fallback && input == target_input_ ? pending_->id : 0;
    }

    void encoderInput(uint64_t token, int64_t pts) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (last_encoder_input_pts_ && pts <= *last_encoder_input_pts_) {
            cancelLocked("unmatched");
            last_encoder_input_pts_ = pts;
            return;
        }
        last_encoder_input_pts_ = pts;
        if (!pending_ || token != pending_->id || !armed_) return;
        // Bounded even if a broken encoder never returns packets. Do not turn a
        // missing or ambiguous correspondence into an apparently valid latency.
        if (input_pts_.count(pts) || input_pts_.size() >= 256) {
            cancelLocked("unmatched");
            return;
        }
        input_pts_[pts] = token;
    }

    void encoderOutput(int64_t pts, Clock::time_point emitted) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto input = input_pts_.find(pts);
        if (!pending_ || input == input_pts_.end() || input->second != pending_->id) return;
        const double ms = std::chrono::duration<double, std::milli>(emitted - received_).count();
        if (ms < 0) {
            cancelLocked("invalid_clock");
            return;
        }
        pending_->milliseconds = ms;
        pending_->encoded_pts = pts;
        pending_->state = "measured";
        publish();
        auto& recent = pending_->previewed ? samples_.previewed_recent : samples_.direct_recent;
        if (recent.size() == 3) recent.erase(recent.begin());
        recent.push_back(*pending_);
        pending_.reset();
        armed_ = false;
        input_pts_.clear();
    }

    Snapshot snapshot(Clock::time_point now = Clock::now()) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (pending_ && now - received_ > std::chrono::seconds(30)) cancelLocked("timeout");
        return samples_;
    }
};

} // namespace avp::mixer
