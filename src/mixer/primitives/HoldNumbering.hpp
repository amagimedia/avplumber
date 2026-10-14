#pragma once

#include <algorithm>
#include <cstdint>

extern "C" {
#include <libavutil/mathematics.h>
#include <libavutil/rational.h>
}

namespace avp::mixer {

/// Schedule and stamps of the copies repeat_last_frame emits while its input is idle.
///
/// Copies are numbered, not timed: each one is stamped one frame after the last stamp
/// the node put on its output, counted from the last live frame's own stamp. The
/// emission instant never enters a stamp, so the node's wake-up slop cannot move a
/// copy between two slots. (Rounding the emission time to the nearest slot did exactly
/// that whenever the train's phase sat near the half slot: two copies in one slot and
/// none in the next, which a presentation-time compositor counts as a discard and a
/// repeat, in pairs, for minutes at a time.)
///
/// Schedule: the first copy 1.5 periods after a live frame, then one per period on a
/// train anchored to that arrival; a live frame always re-anchors both the train and
/// the numbering, so a real paint takes precedence over the copies around it.
///
/// Stall path, a one-slot guard band: a copy whose counted slot lies more than one
/// period before its emission (the node woken late, or a live frame that reached the
/// node long after its stamp) is re-aligned to the first slot at or after the emission
/// (AV_ROUND_UP, as smooth_timestamps aligns), never at or before a slot already
/// emitted, and the count continues from there. That fires once per excursion: the
/// re-aligned copy precedes its slot, so the next counted copy, one period later, is
/// again within the band. Hence no copy is ever emitted more than one period after
/// its slot time. A wake-up more than a period late also restarts the train from the
/// wake-up (no burst). The band assumes the stamps are on the clock `now_ns` reads:
/// the DMA-BUF chain stamps on the monotonic clock and smooth_timestamps keeps that
/// origin. A counter ahead of the clock is left alone; a copy is never numbered back.
class HoldNumbering {
    AVRational frame_;            // one frame as a time base: 1/fps
    int64_t period_ns_;
    bool armed_ = false;
    AVRational tb_{0, 1};         // the live frame's time base; copies are stamped in it
    int64_t anchor_pts_ = 0;      // the stamp the copies count from, in tb_
    int64_t copies_ = 0;          // copies emitted since the anchor
    int64_t next_repeat_ns_ = 0;
    uint64_t realignments_ = 0;

    static constexpr AVRational kNs{1, 1000000000};

    int64_t stamp(int64_t copies) const { return anchor_pts_ + av_rescale_q(copies, frame_, tb_); }

public:
    explicit HoldNumbering(AVRational fps)
        : frame_(av_inv_q(fps)), period_ns_(av_rescale_q(1, frame_, kNs)) {}

    int64_t periodNs() const { return period_ns_; }
    /// Whether a live frame has been seen, i.e. whether copies can be due at all.
    bool armed() const { return armed_; }
    /// When the next copy is due; meaningful once armed().
    int64_t nextRepeatNs() const { return next_repeat_ns_; }
    uint64_t realignments() const { return realignments_; }

    /// A live frame with stamp `pts` in `tb` passed through at `now_ns`.
    void live(int64_t now_ns, int64_t pts, AVRational tb) {
        tb_ = tb;
        anchor_pts_ = pts;
        copies_ = 0;
        armed_ = true;
        // Only step in after a missed period, never between live frames.
        next_repeat_ns_ = now_ns + period_ns_ + period_ns_ / 2;
    }

    bool due(int64_t now_ns) const { return armed_ && now_ns >= next_repeat_ns_; }

    /// The stamp, in the live frame's time base, of the copy emitted at `now_ns`
    /// (call when due()); advances the train.
    int64_t copy(int64_t now_ns) {
        int64_t pts = stamp(copies_ + 1);
        const int64_t late_ns = now_ns - next_repeat_ns_;                 // the wake-up against the train
        const int64_t behind_ns = now_ns - av_rescale_q(pts, tb_, kNs);   // the counted slot against the clock
        next_repeat_ns_ += period_ns_;
        if (late_ns > period_ns_) next_repeat_ns_ = now_ns + period_ns_;   // do not burst after a long stall
        if (late_ns <= period_ns_ && behind_ns <= period_ns_) {
            ++copies_;
            return pts;
        }
        // Outside the band: the first slot at or after the emission, but never one already emitted.
        const int64_t slot = av_rescale_q_rnd(now_ns, kNs, frame_, AV_ROUND_UP);
        pts = std::max(pts, av_rescale_q(slot, frame_, tb_));
        anchor_pts_ = pts;
        copies_ = 0;
        ++realignments_;
        return pts;
    }
};

} // namespace avp::mixer
