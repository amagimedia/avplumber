#pragma once

#include "FrameRate.hpp"
#include "Cadence.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace avp::mixer {

enum class TimestampMode { Cadence, Presentation };

// Frame is a reference-owning value (e.g. VideoFrame or EglImageFrame).
// Rendering adapters retain the decision until their output is accepted.
template<class Frame> class Playout {
public:
    struct Stats {
        uint64_t repeats = 0;
        uint64_t discarded = 0;
        uint64_t overflow = 0;
        uint64_t discontinuities = 0;
        uint64_t phase_corrections = 0;
    };
    struct Decision {
        int64_t index;
        std::vector<std::optional<Frame>> frames;
    };

private:
    static constexpr size_t queue_capacity = 8;
    struct Entry { int64_t index; Frame frame; };
    struct Input {
        std::deque<Entry> queue;
        std::optional<Frame> held;
        std::optional<int64_t> held_index;
        std::optional<int64_t> next_index;
        std::optional<Cadence> cadence;
        std::optional<int64_t> valid_from_ns;
        bool ended = false;
        bool active = true;
        bool prewarm = false;
        Stats stats;
    };
    FrameRate rate_;
    TimestampMode timestamp_mode_;
    int64_t latency_ns_;
    std::vector<Input> inputs_;
    std::optional<int64_t> index_;
    std::optional<Decision> pending_;
    std::optional<int64_t> waiting_deadline_;
    std::vector<size_t> consume_;
    bool started_ = false;
    uint64_t missed_deadlines_ = 0;

public:
    Playout(size_t inputs, FrameRate rate, std::optional<double> latency_ms = {},
            TimestampMode timestamp_mode = TimestampMode::Cadence)
        : rate_(rate), timestamp_mode_(timestamp_mode), latency_ns_(rate.time(2)),
          inputs_(inputs), consume_(inputs) {
        if (!inputs) throw std::invalid_argument("mixer needs inputs");
        if (latency_ms) {
            if (!std::isfinite(*latency_ms) || *latency_ms < 0 ||
                *latency_ms >= static_cast<double>(std::numeric_limits<int64_t>::max()) / 1000000)
                throw std::invalid_argument("mixer latency_ms must be finite and nonnegative");
            latency_ns_ = static_cast<int64_t>(*latency_ms * 1000000);
        }
        // Reserve two queue entries for phase alignment and an arriving frame;
        // larger delays would silently evict frames before their deadlines.
        if (latency_ns_ > rate_.time(queue_capacity - 2))
            throw std::invalid_argument("mixer latency_ms exceeds the six-frame buffer budget");
    }

    void push(size_t input, Frame frame, int64_t timestamp_ns) {
        if (pending_) throw std::logic_error("commit mixer decision before pushing");
        auto &state = inputs_.at(input);
        if (!state.active && !state.prewarm) return;
        if (state.valid_from_ns && timestamp_ns < *state.valid_from_ns) {
            ++state.stats.discarded;
            return;
        }
        state.ended = false;
        int64_t slot;
        if (timestamp_mode_ == TimestampMode::Presentation) {
            slot = rate_.nearestIndex(timestamp_ns);
            if (state.next_index && slot < *state.next_index - 1) {
                state.stats.discarded += state.queue.size();
                state.queue.clear();
                ++state.stats.discontinuities;
            }
            state.next_index = slot + 1;
        } else {
            if (!state.cadence) state.cadence.emplace(rate_, latency_ns_);
            const auto position = state.cadence->observe(timestamp_ns);
            slot = position.index;
            if (position.discontinuity) {
                state.stats.discarded += state.queue.size();
                state.queue.clear();
                ++state.stats.discontinuities;
            } else if (position.phase_shift) {
                for (auto &entry : state.queue) entry.index += position.phase_shift;
                ++state.stats.phase_corrections;
            }
        }
        // Missing paints can leave the inferred cadence behind the output.
        // Recover only a fresh frame assigned to an already committed slot;
        // ordinary jitter inside the buffer and stale catch-up bursts retain
        // their original timing. Queued older frames must not be relabelled.
        if (state.cadence && started_ && slot < *index_ &&
            timestamp_ns >= rate_.time(*index_)) {
            state.cadence->advance(*index_ - slot);
            slot = *index_;
            ++state.stats.phase_corrections;
        }
        if (state.queue.size() == queue_capacity) {
            state.queue.pop_front();
            ++state.stats.overflow;
            ++state.stats.discarded;
        }
        state.queue.push_back({slot, std::move(frame)});
        if (!index_ || (!started_ && slot < *index_)) index_ = slot;
    }

    const Decision *prepare(int64_t now_ns, bool require_all = false) {
        if (pending_) return &*pending_;
        if (!index_ || now_ns < rate_.time(*index_) + latency_ns_) return nullptr;
        auto scheduled = std::max(*index_, rate_.atOrBefore(now_ns - latency_ns_));
        waiting_deadline_.reset();
        if (require_all) {
            for (const auto &input : inputs_) {
                if (!input.active || input.held || input.ended) continue;
                if (input.queue.empty()) return nullptr;
                scheduled = std::max(scheduled, input.queue.front().index);
            }
            waiting_deadline_ = rate_.time(scheduled) + latency_ns_;
            if (now_ns < *waiting_deadline_) return nullptr;
        }
        pending_ = Decision{scheduled, {}};
        for (size_t i = 0; i < inputs_.size(); ++i) {
            auto &input = inputs_[i];
            size_t count = 0;
            for (const auto &entry : input.queue) {
                if (entry.index > scheduled) break;
                ++count;
            }
            consume_[i] = count;
            pending_->frames.push_back(count ? std::optional<Frame>(input.queue[count-1].frame)
                                             : input.held);
        }
        return &*pending_;
    }

    void commit() {
        if (!pending_) throw std::logic_error("no mixer decision to commit");
        for (size_t i = 0; i < inputs_.size(); ++i) {
            auto &input = inputs_[i];
            if (consume_[i]) {
                input.stats.discarded += consume_[i] - 1;
                input.held = std::move(pending_->frames[i]);
                input.held_index = input.queue[consume_[i] - 1].index;
                for (size_t count = consume_[i]; count; --count) input.queue.pop_front();
            } else if (input.held) {
                ++input.stats.repeats;
            }
        }
        if (started_) missed_deadlines_ += pending_->index - *index_;
        index_ = pending_->index + 1;
        pending_.reset();
        waiting_deadline_.reset();
        started_ = true;
    }

    const Stats &stats(size_t input) const { return inputs_.at(input).stats; }
    void resetInput(size_t input, std::optional<int64_t> valid_from_ns = {}, bool preserve_warm = false) {
        if (pending_) throw std::logic_error("commit mixer decision before resetting");
        auto &state = inputs_.at(input);
        if (preserve_warm && state.prewarm && valid_from_ns) {
            // Scene geometry may change while source identity stays fixed.
            // Retain only frames in the current playout window, never an old
            // held picture from a source that stopped while the slot was idle.
            while (!state.queue.empty() && rate_.time(state.queue.front().index) < *valid_from_ns) {
                state.queue.pop_front();
                ++state.stats.discarded;
            }
            if (state.held_index && rate_.time(*state.held_index) < *valid_from_ns) {
                state.held.reset();
                state.held_index.reset();
            }
            state.valid_from_ns = valid_from_ns;
            waiting_deadline_.reset();
            return;
        }
        state.stats.discarded += state.queue.size();
        state.queue.clear();
        state.held.reset();
        state.held_index.reset();
        state.next_index.reset();
        state.cadence.reset();
        state.valid_from_ns = valid_from_ns;
        state.ended = false;
        waiting_deadline_.reset();
    }
    void setActive(size_t input, bool active) {
        auto &state = inputs_.at(input);
        if (state.active == active) return;
        if (!state.prewarm) resetInput(input);
        state.active = active;
    }
    void setPrewarm(size_t input, bool prewarm) {
        auto &state = inputs_.at(input);
        if (state.prewarm == prewarm) return;
        state.prewarm = prewarm;
        if (!prewarm && !state.active) resetInput(input);
    }
    void endInput(size_t input) { inputs_.at(input).ended = true; }
    bool finished() const {
        bool any_active = false;
        for (const auto &input : inputs_) {
            if (!input.active) continue;
            any_active = true;
            if (!input.ended || !input.queue.empty()) return false;
        }
        return any_active && !pending_;
    }
    size_t queued(size_t input) const { return inputs_.at(input).queue.size(); }
    uint64_t missedDeadlines() const { return missed_deadlines_; }
    int64_t latencyNs() const { return latency_ns_; }
    std::optional<int64_t> nextDeadline() const {
        if (waiting_deadline_) return waiting_deadline_;
        return index_ ? std::optional<int64_t>(rate_.time(*index_) + latency_ns_) : std::nullopt;
    }
};

} // namespace avp::mixer
