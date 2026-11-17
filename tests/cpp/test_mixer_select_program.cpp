// The program selector's decision (MixerOrchestrator::selectFrame) on a mixer with no nodes, against
// the whole library (tests/test_mixer_select_program.py). A wipe armed after the lever's take ended
// on the frame before, as resumeLever arms one, puts the program slot's frame out: never the input
// the selector stored on that frame, which a lever wipe leaves at -1 (a tick consumed without a
// frame) and a lever mix at the blend.
//
// A take for a show tick (at_show_tick) lands on that tick whichever slot's frame of it reaches
// the selector first: program's waits (kWaitForDestination) for the destination's, half a frame at
// most; a wait that runs out is the last, the take lands late, and program never stops. A frame of
// the tick that is not ready ends the wait, and the next tick waits again. With no nodes here, a
// cut that lands fails in its routing (completeTake) and aborts after the selector decided; the
// keys' clock the cut branch set just before says where it landed: on the first frame of the new
// program, the tick's own exactly when the destination's frame of it goes out (the branch returns
// the destination then, -1 otherwise). A mix that lands fails the same way after its start tick
// was set. A move lands without nodes: on real edges (MixerSourceSwitcher::holdForInput, as the
// selector node runs it), program's frame of the tick never goes out and the destination's does.
#include "graph_mgmt.hpp"
#include "mixer/MixerGraph.hpp"
#include "mixer/orchestrator/MixerOrchestrator.hpp"
#include "mixer/primitives/MonotonicClock.hpp"
#include "mixer/SourceSwitcher.hpp"
#include <cassert>
#include <chrono>
#include <memory>
#include <thread>

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

constexpr int kWait = MixerOrchestrator::kWaitForDestination;
constexpr int64_t kLongNs = 5000000000;   // a wait no test may sit out

/// A take of `mode` to slot B (its composition revision 7) for tick `at`, nothing on air, as
/// beginTake leaves it after mixer.cut/fade/move {"at_show_tick"}. A cut's clock starts from a
/// previous take's ended one.
std::shared_ptr<MixerState> timedTake(MixerState::TransitionMode mode, int64_t at) {
    auto state = std::make_shared<MixerState>();
    state->fps_num = 60;
    state->fps_den = 1;
    state->pgm_is_slot_a = true;
    state->slot_b.revision = "7";
    state->transition_mode = mode;
    state->take_phase = MixerState::TakePhase::Ready;
    state->take_duration_ns = 1000000000;
    state->take_curve = FadeCurve::Linear;
    state->take_not_before_ns = ns(at);
    state->take_deadline_ns = monotonicNs() + kLongNs;
    TakeClock ended;
    ended.generation = 1;
    ended.end = TakeClock::End::Completed;
    state->publishClock(ended);
    return state;
}

/// Where a cut landed: its clock (generation 2) ended there, on the new program's first frame.
bool cutLandedAt(const MixerState& state, int64_t pts_ns) {
    const auto clock = state.clock();
    return clock->generation == 2 && clock->end == TakeClock::End::Completed &&
           clock->on_air_ns == pts_ns && clock->end_ns == pts_ns;
}

double secondsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

/// The program selector node on real edges: program (slot A), destination (slot B), blend.
class Selector : public MixerSourceSwitcher<av::VideoFrame> {
public:
    using MixerSourceSwitcher::MixerSourceSwitcher;
    static std::shared_ptr<Selector> on(EdgeManager& edges) {
        auto node = std::make_shared<Selector>(
            std::make_unique<EdgeSink<av::VideoFrame>>(edges.find<av::VideoFrame>("out")));
        node->createSourcesFromParameters(edges, Parameters{{"src", {"program", "destination", "blend"}}});
        return node;
    }
    std::vector<const av::VideoFrame*> heads() {
        std::vector<const av::VideoFrame*> frames;
        for (auto& edge : source_edges_) frames.push_back(edge->peek());
        return frames;
    }
    int64_t lastNs() {
        return last_output_pts_.isValid() ? last_output_pts_.timestamp({1, 1000000000}) : 0;
    }
};

/// One selection of MixerSelector::process() once an input has a frame: decide under the mutex,
/// then hold or publish the heads examined.
int selectOnce(MixerOrchestrator& control, MixerState& state, Selector& node) {
    std::unique_lock<std::mutex> lock(state.mutex);
    const auto frames = node.heads();
    const int selected = control.selectFrame(false, frames, 0, node.lastNs());
    if (selected == kWait)
        node.holdForInput(frames, state.pvwSourceSwitcherIndex(), state.take_tick_wait_until_ns - monotonicNs(), lock);
    else
        node.processHeads(frames, selected);
    return selected;
}

/// Selections until one is not a hold, `limit_s` of wall time at most.
int selectUntilDecided(MixerOrchestrator& control, MixerState& state, Selector& node, double limit_s) {
    const auto start = std::chrono::steady_clock::now();
    int selected;
    while ((selected = selectOnce(control, state, node)) == kWait && secondsSince(start) < limit_s) {}
    return selected;
}

/// What the selector put out, in order: pts ns, and whether it is the destination's (revision 7).
std::vector<std::pair<int64_t, bool>> drain(Edge<av::VideoFrame>& out) {
    std::vector<std::pair<int64_t, bool>> frames;
    while (auto* frame = out.peek()) {
        const auto* revision = av_dict_get(frame->raw()->metadata, "avp.mixer.scene", nullptr, 0);
        frames.emplace_back(frame->pts().timestamp({1, 1000000000}), revision != nullptr);
        out.pop();
    }
    return frames;
}

using Output = std::vector<std::pair<int64_t, bool>>;

/// A timed move on the selector node, both arrival orders at its tick, and a destination that
/// never brings its frame of the tick.
void timedMoveOnTheSelectorNode(const std::shared_ptr<NodeManager>& manager, int64_t T) {
    // Program's frame of the tick first; the destination's comes from another thread, which takes
    // the mutex as a command does: the hold waits for it without the mutex.
    {
        auto timed = timedTake(MixerState::TransitionMode::Move, T);
        MixerOrchestrator control(std::make_shared<MixerGraph>(manager), timed);
        EdgeManager edges;
        auto node = Selector::on(edges);
        auto program = edges.find<av::VideoFrame>("program"), destination = edges.find<av::VideoFrame>("destination");
        program->enqueue(frameAt(T - 1));
        const int before = selectOnce(control, *timed, *node);
        assert(before == 0);
        program->enqueue(frameAt(T));
        timed->take_tick_wait_until_ns = monotonicNs() + kLongNs;   // a slow host's: only the frame ends it
        const auto start = std::chrono::steady_clock::now();
        std::thread late([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            std::lock_guard<std::mutex> lock(timed->mutex);
            destination->enqueue(frameAt(T, "7"));
        });
        const int landed = selectUntilDecided(control, *timed, *node, 3.0);
        late.join();
        assert(landed == 1 && secondsSince(start) < 2.0);
        assert(timed->move_on_air && timed->clock()->on_air_ns == ns(T));
        auto* out = edges.find<av::VideoFrame>("out").get();
        assert((drain(*out) == Output{{ns(T - 1), false}, {ns(T), true}}));   // program's T never
        assert(!program->peek() && !destination->peek());
    }
    // The destination's frame of the tick first: out at once, program's of it consumed after.
    {
        auto timed = timedTake(MixerState::TransitionMode::Move, T);
        MixerOrchestrator control(std::make_shared<MixerGraph>(manager), timed);
        EdgeManager edges;
        auto node = Selector::on(edges);
        auto program = edges.find<av::VideoFrame>("program"), destination = edges.find<av::VideoFrame>("destination");
        program->enqueue(frameAt(T - 1));
        const int before = selectOnce(control, *timed, *node);
        assert(before == 0);
        destination->enqueue(frameAt(T, "7"));
        const int landed = selectOnce(control, *timed, *node);
        assert(landed == 1 && timed->clock()->on_air_ns == ns(T) && timed->take_tick_wait_until_ns == 0);
        program->enqueue(frameAt(T));
        const int after = selectOnce(control, *timed, *node);
        assert(after == -1);
        auto* out = edges.find<av::VideoFrame>("out").get();
        assert((drain(*out) == Output{{ns(T - 1), false}, {ns(T), true}}));
        assert(!program->peek());
    }
    // The destination's frame of the tick never comes: half a frame later program's goes out, the
    // move lands late on the destination's next frame, and nothing hangs.
    {
        auto timed = timedTake(MixerState::TransitionMode::Move, T);
        MixerOrchestrator control(std::make_shared<MixerGraph>(manager), timed);
        EdgeManager edges;
        auto node = Selector::on(edges);
        auto program = edges.find<av::VideoFrame>("program"), destination = edges.find<av::VideoFrame>("destination");
        program->enqueue(frameAt(T - 1));
        const int before = selectOnce(control, *timed, *node);
        assert(before == 0);
        program->enqueue(frameAt(T));
        const auto start = std::chrono::steady_clock::now();
        const int timed_out = selectUntilDecided(control, *timed, *node, 3.0);
        assert(timed_out == 0 && secondsSince(start) < 2.0 && timed->take_tick_wait_until_ns == -1);
        program->enqueue(frameAt(T + 1));
        destination->enqueue(frameAt(T + 1, "7"));
        const int landed = selectOnce(control, *timed, *node);
        assert(landed == 1 && timed->clock()->on_air_ns == ns(T + 1));
        auto* out = edges.find<av::VideoFrame>("out").get();
        assert((drain(*out) == Output{{ns(T - 1), false}, {ns(T), false}, {ns(T + 1), true}}));
    }
}

/// A hold ends on a command (setObject "active", as switchProgramSelector after a take is
/// replaced or aborted) and on stop(), not at its deadline.
void holdEndsOnCommandsAndStop(int64_t T) {
    std::mutex mutex;
    EdgeManager edges;
    auto node = Selector::on(edges);
    edges.find<av::VideoFrame>("program")->enqueue(frameAt(T));
    const auto frames = node->heads();
    const auto hold = [&](int64_t wait_ns) {
        std::unique_lock<std::mutex> lock(mutex);
        const auto start = std::chrono::steady_clock::now();
        node->holdForInput(frames, 1, wait_ns, lock);
        assert(!lock.owns_lock());
        return secondsSince(start);
    };
    hold(1000000);   // reads the program frame's signal, which a selection finding it left unread
    std::thread command([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        node->setObject("active", Parameters(0));
    });
    const double for_command = hold(kLongNs);
    command.join();
    assert(for_command < 2.0);
    hold(1000000);
    std::thread stop([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        node->stop();
    });
    const double for_stop = hold(kLongNs);
    stop.join();
    assert(for_stop < 2.0);
    const double stopped = hold(kLongNs);   // stopping: no wait at all
    assert(stopped < 1.0);
    assert(node->heads()[0]);   // a hold consumes no program frame
}

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

    const auto orchestratorFor = [&](const std::shared_ptr<MixerState>& timed) {
        return MixerOrchestrator(std::make_shared<MixerGraph>(manager), timed);
    };
    const av::VideoFrame before = frameAt(T - 1), program_t = frameAt(T), program_t1 = frameAt(T + 1),
                         program_t2 = frameAt(T + 2);
    const av::VideoFrame stale = frameAt(T - 1, "7"), destination_t = frameAt(T, "7"),
                         destination_t1 = frameAt(T + 1, "7"), destination_t2 = frameAt(T + 2, "7");
    const av::VideoFrame unready_t = frameAt(T);   // the slot's frame of the tick, on its previous composition

    // A timed cut, program's frame of the tick first: a tick before it goes out as ever; at the
    // tick nothing goes out while the destination's frame of it is missing, nor while its head is
    // of an earlier tick (the selector consumes that one alone). Then the cut lands on the tick.
    {
        auto timed = timedTake(MixerState::TransitionMode::Cut, T);
        auto cut = orchestratorFor(timed);
        std::lock_guard<std::mutex> timed_lock(timed->mutex);
        assert(cut.selectFrame(false, {&before, nullptr, nullptr}, 0, ns(T - 2)) == 0);
        assert(timed->take_tick_wait_until_ns == 0);
        assert(cut.selectFrame(false, {&program_t, nullptr, nullptr}, 0, ns(T - 1)) == kWait);
        assert(timed->take_tick_wait_until_ns > 0 && timed->take_tick_wait_until_ns <= monotonicNs() + ns(1) / 2);
        const int64_t until = timed->take_tick_wait_until_ns = monotonicNs() + kLongNs;   // a slow host's
        assert(cut.selectFrame(false, {&program_t, &stale, nullptr}, 0, ns(T - 1)) == kWait);
        assert(timed->take_tick_wait_until_ns == until);   // one wait, from its start
        const int landed = cut.selectFrame(false, {&program_t, &destination_t, nullptr}, 0, ns(T - 1));
        assert(landed != kWait && cutLandedAt(*timed, ns(T)) && timed->take_not_before_ns == 0);
    }

    // The destination's frame of the tick first: it goes out at once, nothing waits.
    {
        auto timed = timedTake(MixerState::TransitionMode::Cut, T);
        auto cut = orchestratorFor(timed);
        std::lock_guard<std::mutex> timed_lock(timed->mutex);
        const int landed = cut.selectFrame(false, {nullptr, &destination_t, nullptr}, 0, ns(T - 1));
        assert(landed != kWait && cutLandedAt(*timed, ns(T)) && timed->take_tick_wait_until_ns == 0);
    }

    // The destination's frame of the tick comes not ready (its slot applies the new composition a
    // tick late): it ends the wait and program's frame of the tick goes out; the take keeps its
    // tick and the next tick waits again, so the cut lands on the destination's first ready frame
    // in either order. Program's frame of the tick first:
    {
        auto timed = timedTake(MixerState::TransitionMode::Cut, T);
        auto cut = orchestratorFor(timed);
        std::lock_guard<std::mutex> timed_lock(timed->mutex);
        assert(cut.selectFrame(false, {&program_t, nullptr, nullptr}, 0, ns(T - 1)) == kWait);
        assert(cut.selectFrame(false, {&program_t, &unready_t, nullptr}, 0, ns(T - 1)) == 0);
        assert(timed->take_tick_wait_until_ns == 0 && timed->take_not_before_ns == ns(T));
        assert(timed->clock()->generation == 1);
        assert(cut.selectFrame(false, {&program_t1, nullptr, nullptr}, 0, ns(T)) == kWait);
        assert(timed->take_tick_wait_until_ns > 0);
        const int landed = cut.selectFrame(false, {&program_t1, &destination_t1, nullptr}, 0, ns(T));
        assert(landed != kWait && cutLandedAt(*timed, ns(T + 1)) && timed->take_not_before_ns == 0);
    }
    // ... and the destination's (not ready) first: program's then goes out without waiting.
    {
        auto timed = timedTake(MixerState::TransitionMode::Cut, T);
        auto cut = orchestratorFor(timed);
        std::lock_guard<std::mutex> timed_lock(timed->mutex);
        assert(cut.selectFrame(false, {nullptr, &unready_t, nullptr}, 0, ns(T - 1)) == 0);
        assert(cut.selectFrame(false, {&program_t, nullptr, nullptr}, 0, ns(T - 1)) == 0);
        assert(timed->take_tick_wait_until_ns == 0 && timed->take_not_before_ns == ns(T));
        const int landed = cut.selectFrame(false, {nullptr, &destination_t1, nullptr}, 0, ns(T));
        assert(landed != kWait && cutLandedAt(*timed, ns(T + 1)));
    }

    // The destination's frames of the tick and the next never come: once the wait ran out (half a
    // frame, sat out here), program's frame of the tick goes out, the next one without waiting
    // again, and the cut lands late, on the destination's first frame after them.
    {
        auto timed = timedTake(MixerState::TransitionMode::Cut, T);
        auto cut = orchestratorFor(timed);
        std::lock_guard<std::mutex> timed_lock(timed->mutex);
        assert(cut.selectFrame(false, {&program_t, nullptr, nullptr}, 0, ns(T - 1)) == kWait);
        std::this_thread::sleep_for(std::chrono::nanoseconds(ns(1)));
        assert(cut.selectFrame(false, {&program_t, nullptr, nullptr}, 0, ns(T - 1)) == 0);
        assert(timed->take_tick_wait_until_ns == -1);
        assert(cut.selectFrame(false, {&program_t1, nullptr, nullptr}, 0, ns(T)) == 0);
        assert(timed->transition_mode == MixerState::TransitionMode::Cut && timed->clock()->generation == 1);
        cut.selectFrame(false, {&program_t2, &destination_t2, nullptr}, 0, ns(T + 1));
        assert(cutLandedAt(*timed, ns(T + 2)));
    }

    // An untimed cut never waits: program's frame goes out until the destination's.
    {
        auto timed = timedTake(MixerState::TransitionMode::Cut, T);
        timed->take_not_before_ns = 0;
        auto cut = orchestratorFor(timed);
        std::lock_guard<std::mutex> timed_lock(timed->mutex);
        assert(cut.selectFrame(false, {&program_t, nullptr, nullptr}, 0, ns(T - 1)) == 0);
        assert(timed->take_tick_wait_until_ns == 0);
    }

    // A timed move waits as a cut does (else program's frame of the tick or the destination's, at
    // the move's start, goes out by arrival order) and starts its clock on the tick either way.
    for (const bool program_first : {true, false}) {
        auto timed = timedTake(MixerState::TransitionMode::Move, T);
        auto move = orchestratorFor(timed);
        std::lock_guard<std::mutex> timed_lock(timed->mutex);
        if (program_first) assert(move.selectFrame(false, {&program_t, nullptr, nullptr}, 0, ns(T - 1)) == kWait);
        assert(move.selectFrame(false, {program_first ? &program_t : nullptr, &destination_t, nullptr}, 0, ns(T - 1)) == 1);
        assert(timed->move_on_air && timed->clock()->on_air_ns == ns(T));
        assert(move.selectFrame(false, {&program_t1, &destination_t1, nullptr}, 1, ns(T)) == 1);
    }

    // A timed mix waits too, by the same rule; its picture at the tick is program's in either order
    // (the mix's start), and it starts on the tick either way.
    for (const bool program_first : {true, false}) {
        auto timed = timedTake(MixerState::TransitionMode::Crossfade, T);
        auto mix = orchestratorFor(timed);
        std::lock_guard<std::mutex> timed_lock(timed->mutex);
        if (program_first) assert(mix.selectFrame(false, {&program_t, nullptr, nullptr}, 0, ns(T - 1)) == kWait);
        const int armed = mix.selectFrame(false, {program_first ? &program_t : nullptr, &destination_t, nullptr}, 0, ns(T - 1));
        assert(armed != kWait && timed->take_start_ns == ns(T) && timed->take_not_before_ns == 0);
    }

    timedMoveOnTheSelectorNode(manager, T);
    holdEndsOnCommandsAndStop(T);
}
