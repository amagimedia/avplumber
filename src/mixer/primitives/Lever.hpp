#pragma once
// A switcher's lever arm (T-bar), as a fader that reports its position in [0, 1]. A transition
// runs from the end the lever rests at (the origin) to the other end; reaching it completes the
// transition, and that end is the origin of the next one. A transition that ends without the
// lever (AUTO, CUT, a failed take) leaves the lever where it was: away from both ends it then
// does nothing until it is brought to one, which becomes the origin. Pure; tests/cpp/test_lever.cpp.
#include <algorithm>
#include <cmath>

namespace avp::mixer {

class Lever {
    static constexpr double kEnd = 0.002;   // this close to an end counts as there
    double origin_ = 0;
    double position_ = 0;
    bool engaged_ = true;

public:
    enum class Kind { Idle, Progress, Complete, Cancel };
    struct Move {
        Kind kind;
        double progress;   // from the origin, in [0, 1]
    };

    /// A new position. `lever_take`: the transition on air is the lever's own. Progress: that
    /// transition (or a new one) is at `progress`; Complete: the far end was reached and is the
    /// origin now; Cancel: the lever is back at the origin under its own transition.
    Move move(double position) { return move(position, true); }
    Move move(double position, bool lever_take) {
        position_ = std::clamp(position, 0.0, 1.0);
        if (!engaged_) {
            if (position_ > kEnd && position_ < 1 - kEnd) return {Kind::Idle, 0};
            origin_ = position_ <= kEnd ? 0 : 1;
            engaged_ = true;
            return {Kind::Idle, 0};
        }
        const double progress = std::fabs(position_ - origin_);
        if (progress >= 1 - kEnd) {
            origin_ = 1 - origin_;
            return {Kind::Complete, 1};
        }
        if (progress <= kEnd) return {lever_take ? Kind::Cancel : Kind::Idle, 0};
        return {Kind::Progress, progress};
    }
    /// Records a position while another transition (AUTO, CUT) owns the program.
    void track(double position) { position_ = std::clamp(position, 0.0, 1.0); }
    /// A transition ended other than at the lever's far end.
    void release() {
        if (position_ <= kEnd) origin_ = 0;
        else if (position_ >= 1 - kEnd) origin_ = 1;
        else engaged_ = false;
    }
    double origin() const { return origin_; }
    double position() const { return position_; }
    bool engaged() const { return engaged_; }
};

} // namespace avp::mixer
