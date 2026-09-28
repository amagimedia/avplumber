#pragma once
// Opacity envelope of one downstream key, stepped by the keyer once per
// program frame on its render thread (not thread-safe; one instance per key).
//
// Times are program-frame timestamps in seconds (e.g. av::Timestamp::seconds()),
// the same clock the M/E fade expression reads as t.
//
// Timing rule: a fade starts at the command. The keyer calls retarget() on the
// first program frame after the command, with that frame's time, then level()
// for the same frame. The ramp is anchored one frame period earlier, so that
// frame already shows the first step and a fade of N periods changes over
// exactly N frames, as a cut (duration 0) changes on the first one. A key whose
// first frame arrives later simply joins the ramp where it is by then.
//
// The duration is for the full 0 <-> 1 swing. A fade reversed mid-way continues
// from the level the last frame showed and covers the remaining distance at the
// same pace, so it takes duration * |target - level|. The curve shapes each
// ramp as transition progress, as for the M/E fade: an ease-in fade-out starts
// slowly, like an ease-in fade-in.
#include "fade_curve.hpp"

#include <algorithm>
#include <cmath>

namespace avp::mixer {

class KeyFade {
    bool target_ = false;
    double from_ = 0;       // level at anchor_
    double last_ = 0;       // level returned for the latest program frame
    double anchor_ = 0;     // time at which the level was from_
    double last_t_ = 0;     // time of the latest level() call
    double span_ = 0;       // seconds from from_ to target_; 0 when settled
    double duration_ = 0;   // seconds for a full 0 <-> 1 swing
    double period_ = 0;     // program frame period, seconds
    FadeCurve curve_ = FadeCurve::Linear;

    double targetLevel() const { return target_ ? 1.0 : 0.0; }
    void settle() {
        last_ = from_ = targetLevel();
        span_ = 0;
    }
    // Ramps from the current level to the target, starting at `anchor`.
    void start(double anchor) {
        from_ = last_;
        anchor_ = last_t_ = anchor;
        span_ = duration_ * std::fabs(targetLevel() - from_);
        if (!(span_ > 0)) settle();
    }

public:
    /// Settled at the target: level 1 when on (a key configured on at start), else 0.
    explicit KeyFade(bool on = false) : target_(on) { settle(); }

    /// Starts a fade towards `on`, anchored at t - period (see above).
    /// A duration that is not positive (or NaN) is a cut: the level snaps to the
    /// target. Retargeting to the current target keeps the running fade, except
    /// that a cut still snaps it.
    void retarget(bool on, double duration_s, FadeCurve curve, double t, double period) {
        const bool cut = !(duration_s > 0);
        if (on == target_ && !cut) return;
        target_ = on;
        curve_ = curve;
        duration_ = cut ? 0 : duration_s;
        period_ = period;
        start(t - period);
    }

    /// Level for the program frame at time t, in [0, 1]. Call once per program
    /// frame with non-decreasing t. If t goes backwards (a PTS discontinuity), the
    /// fade restarts from the current level instead of jumping; a jump forward
    /// past the end of the ramp lands on the target.
    double level(double t) {
        if (span_ == 0) return last_;
        if (t < last_t_) start(t);
        last_t_ = t;
        // The ramp ends on the frame nearest its end: half a period absorbs timestamps
        // rounded to their time base (milliseconds at 60 fps), so the Nth frame of an
        // N-period fade lands exactly on the target.
        if (t - anchor_ >= span_ - period_ / 2) {
            settle();
            return last_;
        }
        const double x = (t - anchor_) / span_;
        last_ = std::min(1.0, std::max(0.0, from_ + (targetLevel() - from_) * fadeCurveAt(curve_, x)));
        return last_;
    }

    bool target() const { return target_; }
    /// The level returned by the latest level() call (or the settled level).
    double current() const { return last_; }
    bool fading() const { return span_ > 0; }
    /// The key needs its feed and a draw: it is on, or still fading out.
    bool visible() const { return target_ || last_ > 0; }
};

}  // namespace avp::mixer
