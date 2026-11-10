// The move take (mixer.move, the lever with `move`): the destination slot draws Program's scene
// A gliding into the target B (primitives/layer_move.hpp) at the take clock's progress, and program
// shows that slot directly; there is no transition_cuda blend. Schedule, pause, lever, AUTO run-on,
// cancel and completion are the crossfade's (fade.cpp), driven through the take clock.
#include "internal.hpp"
#include "../move_plan.hpp"

namespace avp::mixer {

void MixerOrchestrator::move(const std::string& scene_name, double duration_sec, FadeCurve curve,
                             const std::string& unmatched, CutLatency::Clock::time_point received) {
    if (!std::isfinite(duration_sec) || duration_sec <= 0) throw Error("mixer: invalid move duration");
    if (unmatched != "fade" && unmatched != "cut") throw Error("mixer: a move's unmatched sources fade or cut");
    avp::timeline::mark("move pressed: " + scene_name);
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!state_->scenes.count(scene_name)) throw Error("mixer: unknown scene: " + scene_name);
    if (actOnTransitionOnAir(scene_name, MixerState::TransitionMode::Move)) return;
    beginMove(scene_name, duration_sec, curve, unmatched, monotonicNs(received), /*manual=*/false);
}

void MixerOrchestrator::beginMove(const std::string& scene_name, double duration_sec, FadeCurve curve,
                                  const std::string& unmatched, int64_t received_ns, bool manual) {
    const auto fadeInstead = [&](const std::string& why) {
        state_->last_move = {{"scene", scene_name}, {"result", "fade"}, {"reason", why}};
        logstream << "mixer: move to " << scene_name << " fades: " << why;
        beginCrossfade(scene_name, duration_sec, curve, std::nullopt, received_ns, manual);
    };
    // The slot follows the take clock this take starts on air: the next one.
    const uint64_t generation = state_->clock()->generation + 1;
    std::string why;
    Parameters spec = planMove(*state_, state_->pgmScene(), state_->scenes.at(scene_name), generation, unmatched, why);
    if (spec.is_null()) return fadeInstead(why);
    state_->pending_move = std::move(spec);
    state_->pvw_slot_scene.clear();   // a preloaded Preview has no move in its composition
    try {
        beginTake(scene_name, MixerState::TransitionMode::Move);
        state_->take_move = std::exchange(state_->pending_move, Parameters(nullptr));
        state_->take_received_ns = received_ns;
        state_->take_duration_ns = int64_t(duration_sec * 1000000000);
        state_->take_curve = curve;
        state_->take_dip.reset();
        state_->take_manual = manual;
        state_->last_move = {{"generation", generation}, {"scene", scene_name}, {"result", "move"}, {"reason", ""},
                             {"unmatched", unmatched}};
        emitTransitionStart(state_->takeKind(), state_->take_duration_ns, curve, manual, received_ns);
    } catch (const std::exception& e) {
        // The destination refused the composition (max_layers for both scenes' layers): fade.
        state_->pending_move = nullptr;
        abortTransition(state_->transition_generation);
        fadeInstead(e.what());
    }
}

int MixerOrchestrator::selectMove(const std::vector<const av::VideoFrame*>& frames, int target, int64_t last_ns) {
    const av::VideoFrame* frame = frames.at(target);
    const bool ready = destinationReady(frame);
    const int64_t target_pts = frame && frame->isValid() && frame->pts().isValid()
        ? frame->pts().timestamp({1, 1000000000}) : 0;
    if (!state_->move_on_air) {
        // As a mix: Program until a destination frame no older than the last output is ready.
        if (!ready || target_pts < last_ns) return state_->pgmSourceSwitcherIndex();
        const double duration = double(state_->take_duration_ns) / 1000000000;
        const double hold = av_q2d({state_->fps_den, state_->fps_num}) / duration;
        state_->take_start_ns = target_pts;
        FadeRequest request{target_pts / 1000000, duration, !state_->pgm_is_slot_a, state_->take_curve, std::nullopt,
                            hold, state_->canvas_transfer};
        state_->take_paused = state_->take_manual;
        if (state_->take_manual) {   // the lever's move comes up where the lever is, held there
            const int64_t at_ms = target_pts / 1000000;
            request.start_ms = at_ms - int64_t(std::llround(state_->take_lever * duration * 1000));
            request.time_floor_ms = request.time_ceiling_ms = at_ms;
            state_->take_hold_ns = at_ms * 1000000;
        }
        state_->scheduleTake(request, true);
        startTakeClock(target_pts, state_->take_duration_ns, state_->take_curve);
        clampTakeClock();
        state_->take_phase = MixerState::TakePhase::Fade;
        state_->move_on_air = true;
        state_->take_progress = state_->clock()->progress(target_pts);
        avp::timeline::mark("move on air");
        return target_pts == last_ns ? -1 : target;
    }
    // On air every new destination frame goes out, a replaced preset's older revision included
    // until its own arrives; only a frame of the current revision completes the take.
    if (!target_pts || target_pts <= last_ns) return -1;   // never a tick already out
    state_->take_progress = state_->clock()->progress(target_pts);
    emitTransitionProgress(state_->takeKind(), state_->take_progress, target_pts);
    // A paused or manual move holds; it completes on the frame its schedule reaches the end, and
    // the next selection takes the new program slot, which is this one.
    endTakeAt(target_pts, state_->destination_ready,
              !state_->take_paused && target_pts >= state_->take_start_ns + state_->take_duration_ns);
    return target;
}

}  // namespace avp::mixer
