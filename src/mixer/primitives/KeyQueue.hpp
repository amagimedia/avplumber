#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <utility>

namespace avp::mixer {

/// One key's frames between their arrival at the keyer and the program tick they are
/// stamped for, and the frame the key shows meanwhile.
///
/// Arrival order is content order; a stamp only says which tick a frame is for. A source
/// that holds its last frame while it is idle (repeat_last_frame) numbers those copies on
/// from the last stamp, so a paint that was merely delivered late arrives after copies
/// stamped at or after its own tick. Taken first in, first out, such a copy stood at the
/// head of the queue until its own, later tick and kept every paint behind it from the
/// ticks they were stamped for: the key repeated its old picture for those ticks and then
/// skipped as many paints at once. So a frame supersedes whatever is queued with its stamp
/// or a later one: those frames are older pictures. The queue therefore stays in stamp
/// order, and frames that arrive in stamp order are queued exactly as they came.
///
/// The bound stops a key on an unrelated clock from retaining its producer's frames for
/// good: past it the oldest queued frame becomes the shown one ahead of its tick. One such
/// frame costs nothing (it is the next to show); a second one before the next tick skips
/// a frame, and the tick after repeats one. `Stats` counts these events.
template<class Frame, class Stamp> class KeyQueue {
public:
    struct Stats {
        uint64_t superseded = 0;   // queued frames dropped for a later arrival stamped before them
        uint64_t replaced = 0;     // queued frames dropped for a later arrival with their own stamp
        uint64_t early = 0;        // frames shown ahead of their tick because the queue was full
    };

private:
    struct Entry { Stamp stamp; Frame frame; };
    std::deque<Entry> queue_;
    std::optional<Frame> shown_;
    size_t bound_;
    Stats stats_;

public:
    explicit KeyQueue(size_t bound) : bound_(bound) {}

    /// A frame nothing supersedes until a queued one reaches its tick (the fade to black's).
    void show(Frame frame) { shown_ = std::move(frame); }

    void push(Frame frame, Stamp stamp) {
        while (!queue_.empty() && !(queue_.back().stamp < stamp)) {
            // Stamped past the arriving frame, it would have stood before it until its own tick.
            // With the same stamp the later arrival was shown anyway; now they share a place.
            ++(stamp < queue_.back().stamp ? stats_.superseded : stats_.replaced);
            queue_.pop_back();
        }
        queue_.push_back({std::move(stamp), std::move(frame)});
        if (queue_.size() > bound_) {
            shown_ = std::move(queue_.front().frame);
            queue_.pop_front();
            ++stats_.early;
        }
    }

    /// The frame to show on the tick that ends at `tick_end`: the newest one stamped before
    /// it, or the one shown last when none is; nullptr before the key's first frame.
    const Frame *at(const Stamp &tick_end) {
        while (!queue_.empty() && queue_.front().stamp < tick_end) {
            shown_ = std::move(queue_.front().frame);
            queue_.pop_front();
        }
        return shown();
    }

    const Frame *shown() const { return shown_ ? &*shown_ : nullptr; }
    size_t queued() const { return queue_.size(); }
    const Stats &stats() const { return stats_; }
};

} // namespace avp::mixer
