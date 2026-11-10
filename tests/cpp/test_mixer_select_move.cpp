// The program selector's move branch (MixerOrchestrator::selectMove) on a mixer with no nodes,
// against the whole library (tests/test_mixer_select_move.py): Program until the destination's
// first frame of its revision, which puts the move on air and starts the next take clock there;
// later destination frames go out directly at that clock's progress, never a tick already out.
#include "graph_mgmt.hpp"
#include "mixer/MixerGraph.hpp"
#include "mixer/orchestrator/MixerOrchestrator.hpp"
#include "mixer/primitives/MonotonicClock.hpp"
#include <cassert>
#include <cmath>

using namespace avp::mixer;

namespace {

av::VideoFrame frameAt(int64_t tick, const char *revision = nullptr) {
    av::VideoFrame frame(AV_PIX_FMT_YUV420P, 16, 16);
    frame.setTimeBase({1, 60});
    frame.setPts(av::Timestamp(tick, {1, 60}));
    if (revision) av_dict_set(&frame.raw()->metadata, "avp.mixer.scene", revision, 0);
    return frame;
}

int64_t ns(int64_t tick) { return av::Timestamp(tick, {1, 60}).timestamp({1, 1000000000}); }

}

int main() {
    const auto manager = std::make_shared<NodeManager>();   // MixerGraph holds it weakly
    auto state = std::make_shared<MixerState>();
    state->fps_num = 60;
    state->fps_den = 1;
    MixerOrchestrator control(std::make_shared<MixerGraph>(manager), state);
    std::lock_guard<std::mutex> lock(state->mutex);
    const int64_t T = 6000;

    // A move to slot B prepared (beginMove): its composition is revision 7.
    state->pgm_is_slot_a = true;
    state->slot_b.revision = "7";
    state->transition_mode = MixerState::TransitionMode::Move;
    state->take_phase = MixerState::TakePhase::Ready;
    state->take_duration_ns = 1000000000;
    state->take_curve = FadeCurve::Linear;
    state->take_deadline_ns = monotonicNs() + 5000000000;
    const uint64_t generation = state->clock()->generation;

    // The destination's frame of an older composition: Program goes out, nothing is on air.
    const av::VideoFrame program = frameAt(T), stale = frameAt(T);
    assert(control.selectFrame(false, {&program, &stale, nullptr}, 0, ns(T - 1)) == 0);
    assert(!state->move_on_air && state->clock()->generation == generation);

    // Its first frame of revision 7: on air there, the next clock starting at its pts.
    const av::VideoFrame first = frameAt(T + 1, "7"), program1 = frameAt(T + 1);
    assert(control.selectFrame(false, {&program1, &first, nullptr}, 0, ns(T)) == 1);
    assert(state->move_on_air && state->take_phase == MixerState::TakePhase::Fade);
    const auto clock = state->clock();
    assert(clock->generation == generation + 1 && clock->end == TakeClock::End::Running);
    assert(clock->on_air_ns == ns(T + 1) && std::fabs(clock->progress(ns(T + 31)) - 0.5) < 1e-9);

    // Half way: the destination's frame goes out directly, at the clock's progress.
    const av::VideoFrame half = frameAt(T + 31, "7"), program31 = frameAt(T + 31);
    assert(control.selectFrame(false, {&program31, &half, nullptr}, 1, ns(T + 30)) == 1);
    assert(std::fabs(state->take_progress - 0.5) < 1e-9 && state->move_on_air);

    // A tick already out goes nowhere; the take stays on air.
    assert(control.selectFrame(false, {&program31, &half, nullptr}, 1, ns(T + 31)) == -1);
    assert(state->move_on_air && state->transition_mode == MixerState::TransitionMode::Move);

    // A replaced preset's older revision keeps going out until its own arrives.
    state->slot_b.revision = "8";
    state->destination_ready = false;   // as replacePreset leaves it
    const av::VideoFrame older = frameAt(T + 32, "7"), program32 = frameAt(T + 32);
    assert(control.selectFrame(false, {&program32, &older, nullptr}, 1, ns(T + 31)) == 1);
}
