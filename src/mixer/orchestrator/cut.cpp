#include "internal.hpp"

namespace avp::mixer {

void MixerOrchestrator::prepareScene(const std::string& scene_name) {
    if (state_->pvw_slot_scene != scene_name)
        loadSceneIntoSlot(!state_->pgm_is_slot_a, scene_name, true);
    setNodeObject(state_->pvwSlot().post_otm_name, "outputs", Parameters(1u));
    resetSlotNormFps(nodes_, *state_);
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
    state_->publishTakePreview(scene, pts_ns);
    applySceneControls(state_->scenes.at(scene));
    applyPostTransitionRouting(destination_a, scene);
    finishSnapshot(pts_ns - 1);
    finishTransition(destination_a, scene, pts_ns);
}

void MixerOrchestrator::cut(const std::string& scene_name, CutLatency::Clock::time_point received) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    const bool preloaded = state_->pvw_slot_scene == scene_name;
    if (!state_->scenes.count(scene_name)) throw Error("mixer: unknown scene: " + scene_name);
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

} // namespace avp::mixer
