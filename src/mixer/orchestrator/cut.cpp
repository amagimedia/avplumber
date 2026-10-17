#include "internal.hpp"

namespace avp::mixer {

void MixerOrchestrator::prepareScene(const std::string& scene_name) {
    if (state_->pvw_slot_scene != scene_name)
        loadSceneIntoSlot(!state_->pgm_is_slot_a, scene_name, true);
    setNodeObject(state_->pvwSlot().post_otm_name, "outputs", Parameters(1u));
}

void MixerOrchestrator::beginTake(const std::string& scene_name, MixerState::TransitionMode mode) {
    if (!state_->scenes.count(scene_name)) throw Error("mixer: unknown scene: " + scene_name);
    interruptTransition(Interruption::Replaced);
    state_->graph = nodes_;
    state_->transition_mode = mode;
    ++state_->transition_generation;
    state_->transition_scene_name = scene_name;
    state_->take_phase = MixerState::TakePhase::Ready;
    state_->take_start_ns = 0;
    state_->take_progress = 0;
    state_->lever_at_ns = 0;
    state_->wipe_switched = false;
    state_->destination_ready = false;
    state_->take_deadline_ns = monotonicNs() + kTakeReadyTimeoutMs * 1000000;
    prepareScene(scene_name);
}

bool MixerOrchestrator::destinationReady(const av::VideoFrame* frame) const {
    if (!frame || !frame->isValid() || !frame->pts().isValid()) return false;
    const auto* revision = av_dict_get(frame->raw()->metadata, "avp.mixer.scene", nullptr, 0);
    return revision && revision->value == state_->pvwSlot().revision;
}

void MixerOrchestrator::completeTake(int64_t pts_ns) {
    const bool destination_a = !state_->pgm_is_slot_a;
    const std::string scene = state_->transition_scene_name;
    if (state_->wipe_on_air) {
        setNodeObject(state_->wipe_otm_name, "outputs", Parameters(1u));
        retireWipeChain();
    }
    state_->blend_on_air = state_->wipe_on_air = state_->take_paused = false;
    state_->take_request.reset();
    endTakeClock(TakeClock::End::Completed, pts_ns);
    // A take the lever completed leaves it engaged at that end: moving it back starts the next.
    if (state_->take_lever_end == LeverEnd::None) state_->lever.release();
    state_->take_manual = false;
    state_->take_lever_end = LeverEnd::None;
    state_->take_progress = 0;
    state_->publishTakePreview(scene, pts_ns);
    applySceneControls(state_->scenes.at(scene));
    applyPostTransitionRouting(destination_a, scene);
    finishTransition(destination_a, scene, pts_ns);
}

void MixerOrchestrator::cut(const std::string& scene_name, CutLatency::Clock::time_point received) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    const bool preloaded = state_->pvw_slot_scene == scene_name;
    if (!state_->scenes.count(scene_name)) throw Error("mixer: unknown scene: " + scene_name);
    if (actOnTransitionOnAir(scene_name, MixerState::TransitionMode::Cut)) return;
    try {
        beginTake(scene_name, MixerState::TransitionMode::Cut);
        state_->take_received_ns = monotonicNs(received);
        if (state_->cut_latency) {
            state_->cut_latency->timing.begin(scene_name, preloaded, state_->pvwSourceSwitcherIndex(), received);
            state_->cut_latency->timing.arm();
        }
    } catch (...) {
        abortTransition(state_->transition_generation);
        throw;
    }
}

bool MixerOrchestrator::actOnTransitionOnAir(const std::string& scene_name, MixerState::TransitionMode pressed) {
    if (!state_->blend_on_air && !state_->wipe_on_air) return false;
    if (state_->cut_latency) state_->cut_latency->timing.cancel();
    const bool same_scene = scene_name == state_->transition_scene_name;
    if (!same_scene) replacePreset(scene_name);
    if (pressed == MixerState::TransitionMode::Cut) {
        // selectProgram's cut branch takes the preset's next ready frame; until then the
        // transition stays on air as it is, both sources moving.
        state_->transition_mode = MixerState::TransitionMode::Cut;
        state_->take_paused = false;
        // The cut owns the take now: AUTO under it is ignored, and the lever is released when it
        // lands, as after any take that ended without the lever.
        state_->take_manual = false;
        state_->take_lever_end = LeverEnd::None;
        state_->take_deadline_ns = monotonicNs() + kTakeReadyTimeoutMs * 1000000;
        logstream << "mixer: cut completes the transition on air, program " << scene_name;
    } else if (same_scene && state_->take_manual) {
        runOnFromLever();
    } else if (same_scene && state_->transition_mode == MixerState::TransitionMode::Crossfade) {
        toggleCrossfadePause();
    } else if (same_scene) {
        logstream << "mixer: " << (pressed == MixerState::TransitionMode::Wipe ? "wipe" : "fade")
                  << " to " << scene_name << " ignored: the " << state_->takeKind() << " on air continues";
    }
    return true;
}

void MixerOrchestrator::replacePreset(const std::string& scene_name) {
    loadSceneIntoSlot(!state_->pgm_is_slot_a, scene_name, true);
    state_->transition_scene_name = scene_name;
    // Only a frame of the new revision completes the transition; one that never comes ends it
    // through the take deadline like any unready destination.
    state_->destination_ready = false;
    state_->take_deadline_ns = monotonicNs() + kTakeReadyTimeoutMs * 1000000;
    // The Preview shown is what the transition goes to: a CUT or AUTO sent for the shown
    // Preview then acts on this scene, not on the one it replaced.
    state_->publishPreview(scene_name, 0);
    logstream << "mixer: preset replaced under the " << state_->takeKind() << " on air by " << scene_name;
}

void MixerOrchestrator::runOnFromLever() {
    if (!state_->take_request) return;
    // AUTO on the lever's take runs it on from where the lever has it, at AUTO's rate (a wipe's:
    // its clip's own); the lever is picked up again only at one of its ends.
    state_->take_manual = false;
    state_->take_lever_end = LeverEnd::None;
    state_->lever.release();
    const int64_t at_ms = takeScheduleFromNs() / 1000000;
    const double x = requestProgress(*state_->take_request, at_ms);
    state_->scheduleTake(runOn(*state_->take_request, at_ms, double(state_->take_duration_ns) / 1e9), false);
    state_->take_start_ns = state_->take_request->start_ms * 1000000;
    state_->take_paused = false;
    applyTakeSchedule();
    logstream << "mixer: AUTO runs the lever's " << state_->takeKind() << " to " << state_->transition_scene_name
              << " on from " << x;
}

void MixerOrchestrator::toggleCrossfadePause() {
    if (!state_->take_request) return;
    // The held mix never steps back: see mixChangeNs().
    const int64_t at_ns = mixChangeNs();
    FadeRequest request = *state_->take_request;
    if (!state_->take_paused) {
        request.time_ceiling_ms = at_ns / 1000000;
        state_->take_hold_ns = at_ns;
    } else {
        state_->take_start_ns += at_ns - state_->take_hold_ns;
        request.start_ms = state_->take_start_ns / 1000000;
        request.time_floor_ms = at_ns / 1000000;
        request.time_ceiling_ms.reset();
    }
    state_->take_paused = !state_->take_paused;
    state_->scheduleTake(request, false);
    applyTakeSchedule();
    logstream << "mixer: fade to " << state_->transition_scene_name << (state_->take_paused ? " paused" : " resumed");
}

} // namespace avp::mixer
