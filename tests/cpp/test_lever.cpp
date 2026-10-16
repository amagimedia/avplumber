// The lever arm's direction and pickup, and the take clock the keys follow.
#include "mixer/primitives/Lever.hpp"
#include "mixer/primitives/TakeClock.hpp"
#include <cassert>
#include <cmath>

using avp::mixer::Lever;
using avp::mixer::TakeClock;

void print_stack_trace() {}   // util.hpp's Error, reached through fade_curve.hpp

static bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

static void testLever() {
    Lever lever;
    assert(lever.move(0.0, false).kind == Lever::Kind::Idle);   // at rest, nothing on air
    auto m = lever.move(0.25);
    assert(m.kind == Lever::Kind::Progress && near(m.progress, 0.25));
    assert(lever.move(0.0).kind == Lever::Kind::Cancel);          // back to the start: cancelled
    assert(lever.move(0.6).kind == Lever::Kind::Progress);
    assert(lever.move(1.0).kind == Lever::Kind::Complete && lever.origin() == 1);
    // The next transition runs from the far end back.
    m = lever.move(0.7);
    assert(m.kind == Lever::Kind::Progress && near(m.progress, 0.3));
    assert(lever.move(0.0).kind == Lever::Kind::Complete && lever.origin() == 0);

    // AUTO finished a transition the lever had taken half way: the lever does nothing until it
    // reaches an end, and that end is the next origin.
    lever.move(0.5);
    lever.release();
    assert(!lever.engaged());
    assert(lever.move(0.8).kind == Lever::Kind::Idle && lever.move(0.4).kind == Lever::Kind::Idle);
    assert(lever.move(1.0).kind == Lever::Kind::Idle && lever.engaged() && lever.origin() == 1);
    assert(lever.move(0.9).kind == Lever::Kind::Progress);

    // A position recorded under an AUTO is where release() finds the lever.
    Lever other;
    other.track(1.0);
    other.release();
    assert(other.engaged() && other.origin() == 1);
    other.track(0.0);
    other.release();
    assert(other.engaged() && other.origin() == 0);
}

static void testClock() {
    TakeClock none;
    TakeClock mix = none.next(1000, 1000, avp::mixer::FadeCurve::Linear);
    assert(mix.generation == 1 && mix.previous.generation == 0);
    assert(near(mix.progress(999), 0) && near(mix.progress(1500), 0.5) && near(mix.progress(2500), 1));
    mix.ceiling_ns = 1250;   // paused, or the lever held at a quarter
    assert(near(mix.progress(1900), 0.25));
    mix.floor_ns = mix.ceiling_ns = 1600;
    assert(near(mix.progress(100), 0.6) && near(mix.progress(5000), 0.6));
    TakeClock cut = mix.next(3000, 0, avp::mixer::FadeCurve::Linear);
    assert(cut.previous.generation == 1 && cut.previous.end == TakeClock::End::Cancelled);
    assert(near(cut.progress(2999), 0) && near(cut.progress(3000), 1));
    cut.end = TakeClock::End::Completed;
    cut.end_ns = 3000;
    const TakeClock after = cut.next(4000, 500, avp::mixer::FadeCurve::Linear);
    assert(after.previous.end == TakeClock::End::Completed && after.previous.end_ns == 3000);
}

int main() {
    testLever();
    testClock();
}
