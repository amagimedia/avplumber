// The program selector's decision (MixerOrchestrator::selectFrame) on a mixer with no nodes, against
// the whole library (tests/test_mixer_select_program.py). A wipe armed after the lever's take ended
// on the frame before, as resumeLever arms one, puts the program slot's frame out: never the input
// the selector stored on that frame, which a lever wipe leaves at -1 (a tick consumed without a
// frame) and a lever mix at the blend.
#include "graph_mgmt.hpp"
#include "mixer/MixerGraph.hpp"
#include "mixer/orchestrator/MixerOrchestrator.hpp"
#include "mixer/primitives/MonotonicClock.hpp"
#include <cassert>

using namespace avp::mixer;

namespace {

av::VideoFrame frameAt(int64_t tick) {
    av::VideoFrame frame(AV_PIX_FMT_YUV420P, 16, 16);
    frame.setTimeBase({1, 60});
    frame.setPts(av::Timestamp(tick, {1, 60}));
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
    const int64_t T = 1000;

    // The lever's wipe on air, held at its far end, background Preview (slot B): the program
    // slot's frame of tick T reaches the selector first and goes nowhere.
    state->pgm_is_slot_a = true;
    state->transition_mode = MixerState::TransitionMode::Wipe;
    state->take_phase = MixerState::TakePhase::Wipe;
    state->wipe_on_air = state->wipe_switched = true;
    FadeRequest at_end{0, 1.0, false};
    at_end.time_floor_ms = at_end.time_ceiling_ms = ns(T) / 1000000;
    at_end.start_ms = *at_end.time_floor_ms - 1000;
    state->scheduleTake(at_end, true);
    state->take_deadline_ns = monotonicNs() + 5000000000;
    const av::VideoFrame program = frameAt(T);
    const int consumed = control.selectFrame(false, {&program, nullptr, nullptr}, 1, ns(T - 1));
    assert(consumed == -1);

    // The far end landed (completeTake: Program is slot B) and the lever, moved back meanwhile,
    // armed the wipe back (armWipe: nothing on air yet). Both slots' frames of tick T + 1 are in.
    state->pgm_is_slot_a = false;
    state->take_phase = MixerState::TakePhase::WipeReady;
    state->wipe_on_air = state->wipe_switched = false;
    const av::VideoFrame a = frameAt(T + 1), b = frameAt(T + 1);
    assert(control.selectFrame(false, {&a, &b, nullptr}, consumed, ns(T)) == 1);

    // The same after a lever mix ended in the program selector, which put the blend out last.
    state->pgm_is_slot_a = true;
    assert(control.selectFrame(false, {&a, &b, nullptr}, MixerState::transSourceSwitcherIndex(), ns(T)) == 0);
}
