// The lever arm's direction and pickup, and the take clock the keys follow.
#include "mixer/primitives/Lever.hpp"
#include "mixer/primitives/TakeClock.hpp"
#include <cassert>
#include <cmath>
#include <limits>

using avp::mixer::Lever;
using avp::mixer::RampHistory;
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
    constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
    mix.schedule = RampHistory({1000, 1000, kMin, 1250});   // paused, or the lever held at a quarter
    assert(near(mix.progress(1900), 0.25));
    mix.schedule = RampHistory({1000, 1000, 1600, 1600});
    assert(near(mix.progress(100), 0.6) && near(mix.progress(5000), 0.6));
    TakeClock cut = mix.next(3000, 0, avp::mixer::FadeCurve::Linear);
    assert(cut.previous.generation == 1 && cut.previous.end == TakeClock::End::Cancelled);
    assert(near(cut.progress(2999), 0) && near(cut.progress(3000), 1));
    cut.end = TakeClock::End::Completed;
    cut.end_ns = 3000;
    const TakeClock after = cut.next(4000, 500, avp::mixer::FadeCurve::Linear);
    assert(after.previous.end == TakeClock::End::Completed && after.previous.end_ns == 3000);
}

// The clock's linear part is rampAt, both ways and under its clamps; a step switches whole there.
static void testRampAndStep() {
    using avp::mixer::rampAt;
    TakeClock clock = TakeClock().next(1000, 1000, avp::mixer::FadeCurve::Linear);
    for (const int64_t t : {500, 1000, 1250, 1999, 2000, 9000})
        assert(near(clock.progress(t), rampAt(clock.schedule, t)));
    clock.schedule = RampHistory({3000, -2000, 1400, 1800});   // the lever moving back: 0.8 at 1400, 0.6 at 1800
    assert(near(clock.progress(0), 0.8) && near(clock.progress(1600), 0.7) && near(clock.progress(5000), 0.6));
    assert(near(rampAt({3000, -2000, 1400, 1800}, 1600), 0.7));
    assert(clock.on_air_ns == 1000);   // the clamps never move where the take went on air

    TakeClock wipe = TakeClock().next(1000, 1000, avp::mixer::FadeCurve::EaseIn, avp::mixer::kWipeSwitchAt);
    assert(wipe.progress(1499) == 0 && wipe.progress(1500) == 1 && wipe.progress(1900) == 1);
    wipe.schedule = RampHistory({3000, -2000, 1400, 2600});   // back: past the switch point until 2000, Program again from there
    assert(wipe.progress(1999) == 1 && wipe.progress(2000) == 1 && wipe.progress(2001) == 0 && wipe.progress(9000) == 0);
    wipe.schedule = RampHistory({3000, -2000, 1600, 1600});   // held past it
    assert(wipe.progress(0) == 1 && wipe.progress(9000) == 1);
    const TakeClock mix = TakeClock().next(0, 1000, avp::mixer::FadeCurve::EaseIn);
    assert(mix.step < 0 && near(mix.progress(500), 0.25));   // no step: the curve, as before
}

// A change of a take's schedule applies from its floor on; the ticks before it keep their value.
static void testRampHistory() {
    using avp::mixer::FadeCurve;
    using avp::mixer::kWipeSwitchAt;
    using avp::mixer::rampAt;
    using avp::mixer::RampNs;
    RampHistory history(RampNs{0, 1000, 0, 0});          // held at 0
    history.change({100, 200, 100, 200});               // to 0.5 over 100..200
    history.change({50, 400, 150, 250});                // from 150: 0.25 there, on to 0.5 at 250
    assert(rampAt(history, 120) == 0.1 && rampAt(history, 140) == 0.2 && rampAt(history, 180) == 0.325);
    assert(rampAt(history, 50) == 0 && rampAt(history, 400) == 0.5);
    history.change({150, 1000, 150, 150});              // the same tick again replaces that change
    assert(rampAt(history, 180) == 0 && rampAt(history, 140) == 0.2);
    history.change({20, 100, 120, 120});                // an earlier one replaces those after it
    assert(rampAt(history, 110) == 0.05 && rampAt(history, 180) == 1);
    // Only the newest kSize are kept: a tick older than all of them reads the oldest kept.
    for (int64_t k = 0; k < int64_t(2 * RampHistory::kSize); ++k) history.change({0, 1, 1000 + k, 1000 + k});
    assert(rampAt(history, 1000 + 2 * int64_t(RampHistory::kSize)) == 1 && rampAt(history, 0) == 1);
    // The keys' clock reads the same history.
    TakeClock clock = TakeClock().next(0, 1000, FadeCurve::Linear, kWipeSwitchAt);
    clock.schedule.change({1500, -1000, 600, 1500});   // back from 0.9 at 600
    assert(clock.progress(550) == 1 && clock.progress(1000) == 1 && clock.progress(1001) == 0);
}

// One lever position against the take: what leverStep says the orchestrator does with it.
static void testLeverStep() {
    using avp::mixer::LeverEnd;
    using avp::mixer::LeverStep;
    using avp::mixer::leverStep;
    using Kind = LeverStep::Kind;
    Lever lever;
    // From rest a position starts the lever's take; then it moves it, and its ends land.
    auto step = leverStep(lever, 0.3, true, false, false, false);
    assert(step.kind == Kind::Start && near(step.progress, 0.3) && step.end == LeverEnd::None);
    step = leverStep(lever, 0.6, false, true, false, true);
    assert(step.kind == Kind::Move && near(step.progress, 0.6));
    step = leverStep(lever, 1.0, false, true, false, true);
    assert(step.kind == Kind::Move && step.end == LeverEnd::Complete && lever.origin() == 1);
    // While that end lands, a late or repeated position only moves the lever: the origin stays
    // flipped, so 0.995 never reads as a cancel against it.
    step = leverStep(lever, 0.995, false, true, true, true);
    assert(step.kind == Kind::Ignore && lever.origin() == 1 && near(lever.position(), 0.995));
    // Landed, the lever stays engaged at its end (the take released nothing): moving it on starts
    // the next take, from that end.
    assert(lever.engaged());
    step = leverStep(lever, 0.8, true, false, false, false);
    assert(step.kind == Kind::Start && near(step.progress, 0.2));
    // Back at its start: on air the cancel lands on the picture, before air it drops at once.
    step = leverStep(lever, 1.0, false, true, false, true);
    assert(step.kind == Kind::Move && step.end == LeverEnd::Cancel && near(step.progress, 0) && lever.origin() == 1);
    leverStep(lever, 0.7, true, false, false, false);
    step = leverStep(lever, 1.0, false, true, false, false);
    assert(step.kind == Kind::Drop);
    // One message from rest to the far end starts the take already complete.
    Lever flick;
    step = leverStep(flick, 1.0, true, false, false, false);
    assert(step.kind == Kind::Start && step.end == LeverEnd::Complete && near(step.progress, 1));
    // Another take owns the program (AUTO, a cut on air): the lever is only tracked.
    Lever other;
    step = leverStep(other, 0.4, false, false, false, true);
    assert(step.kind == Kind::Ignore && near(other.position(), 0.4) && other.origin() == 0);
}

int main() {
    testLever();
    testClock();
    testRampAndStep();
    testRampHistory();
    testLeverStep();
}
