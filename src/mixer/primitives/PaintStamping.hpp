#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>

extern "C" {
#include <libavutil/mathematics.h>
#include <libavutil/rational.h>
}

namespace avp::mixer {

/// Slot numbering of a browser source's frames on the compositor's tick grid, by counting,
/// corrected against the playout deadline.
///
/// The compositor (TimestampMode::Presentation) draws slot k at time(k) + latency and shows the
/// newest frame numbered at or before k; a frame that arrives after that instant is repeated over
/// and then superseded (a repeat and a discard), and two frames numbered alike cost a discard.
/// Numbering by counting (each frame the previous slot plus one) keeps a browser's own clock
/// jitter out of the slots, but a count knows nothing of time: paints that reach the mixer slower
/// than the slot rate (a page at 28 fps, a stalled delivery) fall behind their deadlines one slot
/// per missing paint, and a numbering re-aligned from a mean drift reacts late and in jumps.
///
/// Rules, applied to every frame with its own stamp `in_ns` (the paint's CLOCK_MONOTONIC time or
/// a hold copy's emission) and the instant `now_ns` it is numbered, both on the clock time(k) is on:
///
///  * First frame, and the first after an input jump longer than `discontinuity_ns` (a page that
///    paused with no hold ahead of it): the first slot at or after its stamp (AV_ROUND_UP), never
///    behind a slot already given out. Otherwise: the previous slot plus one.
///  * Late: when the frame's margin to its slot's deadline, time(slot) + latency - now, is under
///    `guard_ns`, the numbering moves one slot later, once, and further only while the frame would
///    still miss its tick (margin below zero: a stall). The slot left empty is one repeat, which is
///    the truth of a paint that did not come; the frame is then drawn at a tick it can make.
///  * Early: when the frame's lead on its slot time, time(slot) - now, exceeds `early_ns` for
///    `early_frames` frames in a row, the numbering moves one slot back (one frame shares a slot
///    with the previous: one discard). The two thresholds are three slots apart at a two-slot
///    latency, so jitter crosses at most one of them: no oscillation. A late shift resets the run.
///
/// Cost: a window whose paints arrive within the guard of the deadline carries one extra slot of
/// latency, kept until its paints run more than `early_ns` ahead; nothing else changes latency.
/// With repeat_last_frame ahead of this numbering its copies are frames like any other: paints,
/// copies and both handoffs between them share one count, so a page that pauses and resumes
/// never puts two frames in one slot or none in the next.
class PaintStamping {
public:
    struct Config {
        AVRational fps{30, 1};
        int64_t latency_ns = 0;        // slot k is drawn at time(k) + latency
        int64_t guard_ns = 0;          // margin to that instant under which the numbering moves later
        int64_t early_ns = 0;          // lead on time(k) above which it may move back (0: never)
        int early_frames = 30;         // ... after this many such frames in a row
        int64_t discontinuity_ns = 0;  // input jump that re-aligns to the stamp (0: never)
    };
    struct Stats {
        uint64_t shifts_later = 0;     // slots skipped for late frames
        uint64_t shifts_back = 0;      // slots shared for early frames
        uint64_t realignments = 0;     // re-alignments to a stamp after an input jump
    };
    struct Result {
        int64_t slot = 0;
        int64_t shifted = 0;           // slots moved for this frame: > 0 later, -1 back
        bool realigned = false;
    };

    explicit PaintStamping(const Config &config) : cfg_(config), frame_(av_inv_q(config.fps)) {}

    const Config &config() const { return cfg_; }
    const Stats &stats() const { return stats_; }
    bool started() const { return started_; }
    int64_t nextSlot() const { return next_; }

    /// time(k): the slot's instant on the grid.
    int64_t timeOf(int64_t slot) const { return av_rescale_q(slot, frame_, kNs); }
    /// The first slot whose instant is at or after `ns`.
    int64_t slotAtOrAfter(int64_t ns) const { return av_rescale_q_rnd(ns, kNs, frame_, AV_ROUND_UP); }
    /// How long before the compositor draws `slot` a frame numbered for it at `now_ns` arrives.
    int64_t marginNs(int64_t slot, int64_t now_ns) const { return timeOf(slot) + cfg_.latency_ns - now_ns; }

    /// The slot for a frame stamped `in_ns` (when `in_valid`) that is being numbered at `now_ns`.
    Result stamp(bool in_valid, int64_t in_ns, int64_t now_ns) {
        Result r;
        int64_t slot;
        const bool jump = in_valid && last_in_valid_ && cfg_.discontinuity_ns > 0 &&
                          std::llabs(in_ns - last_in_ns_) > cfg_.discontinuity_ns;
        if (!started_ || jump) {
            slot = slotAtOrAfter(in_valid ? in_ns : now_ns);
            if (started_) {
                slot = std::max(slot, next_);
                ++stats_.realignments;
                r.realigned = true;
            }
            early_run_ = 0;
        } else {
            slot = next_;
        }
        if (marginNs(slot, now_ns) < cfg_.guard_ns) {
            const int64_t from = slot;
            ++slot;
            while (marginNs(slot, now_ns) < 0) ++slot;
            r.shifted = slot - from;
            stats_.shifts_later += r.shifted;
            early_run_ = 0;
        } else if (started_ && cfg_.early_ns > 0 && timeOf(slot) - now_ns > cfg_.early_ns) {
            if (++early_run_ >= cfg_.early_frames) {
                --slot;
                r.shifted = -1;
                ++stats_.shifts_back;
                early_run_ = 0;
            }
        } else {
            early_run_ = 0;
        }
        next_ = slot + 1;
        started_ = true;
        if (in_valid) {
            last_in_ns_ = in_ns;
            last_in_valid_ = true;
        }
        r.slot = slot;
        return r;
    }

    /// Forget the numbering (an input reset): the next frame is aligned to its stamp again.
    /// The counters stay; they describe the node's life.
    void reset() {
        started_ = false;
        last_in_valid_ = false;
        next_ = 0;
        early_run_ = 0;
    }

private:
    static constexpr AVRational kNs{1, 1000000000};
    Config cfg_;
    AVRational frame_;                 // one slot as a time base: 1/fps
    Stats stats_;
    bool started_ = false;
    bool last_in_valid_ = false;
    int64_t last_in_ns_ = 0;
    int64_t next_ = 0;
    int early_run_ = 0;
};

} // namespace avp::mixer
