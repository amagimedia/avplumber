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

/// How the lever's own take ends: at the far end (Complete) or back at its start (Cancel), each once
/// the picture has followed the lever there.
enum class LeverEnd { None, Complete, Cancel };

/// What one lever position does to the program. Ignore: nothing (another take owns the program, or
/// the lever's end is still landing, so only the position is kept); Drop: cancel at once, nothing
/// of it is on air; Start: begin the lever's take at `progress`; Move: the lever's take goes to
/// `progress`, and `end` says whether the lever reached one of its ends.
struct LeverStep {
    enum class Kind { Ignore, Drop, Start, Move };
    Kind kind;
    double progress = 0;
    LeverEnd end = LeverEnd::None;
};

/// `idle`: no take at all; `lever_take`: the take is the lever's; `ending`: its end is landing;
/// `on_air`: something of it is on air, so a cancel lands on the picture instead of dropping it.
inline LeverStep leverStep(Lever& lever, double position, bool idle, bool lever_take, bool ending, bool on_air) {
    using Kind = LeverStep::Kind;
    if ((!idle && !lever_take) || ending) {
        lever.track(position);
        return {Kind::Ignore};
    }
    const auto move = lever.move(position, lever_take);
    const Kind kind = lever_take ? Kind::Move : Kind::Start;
    switch (move.kind) {
        case Lever::Kind::Idle: return {Kind::Ignore};
        case Lever::Kind::Cancel: return on_air ? LeverStep{Kind::Move, 0, LeverEnd::Cancel} : LeverStep{Kind::Drop};
        case Lever::Kind::Complete: return {kind, 1, LeverEnd::Complete};
        case Lever::Kind::Progress: break;
    }
    return {kind, move.progress};
}

} // namespace avp::mixer
