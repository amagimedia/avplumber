#include "internal.hpp"

namespace avp::mixer {

void MixerOrchestrator::fade(const std::string& scene_name, double duration_sec,
        FadeCurve curve, std::optional<std::array<uint8_t, 3>> dip, CutLatency::Clock::time_point received) {
    if (!std::isfinite(duration_sec) || duration_sec <= 0) throw Error("mixer: invalid fade duration");
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!state_->scenes.count(scene_name)) throw Error("mixer: unknown scene: " + scene_name);
    try {
        beginTake(scene_name, MixerState::TransitionMode::Crossfade);
        state_->take_received_ns = monotonicNs(received);
        state_->take_duration_ns = int64_t(duration_sec * 1000000000);
        state_->take_curve = curve;
        state_->take_dip = dip;
    } catch (...) {
        abortTransition(state_->transition_generation);
        throw;
    }
}

int MixerOrchestrator::selectProgram(const std::vector<const av::VideoFrame*>& frames, int active, int64_t last_ns) {
    const auto pts = [&](int input) {
        const auto* frame = frames.at(input);
        return frame && frame->isValid() && frame->pts().isValid()
            ? frame->pts().timestamp({1, 1000000000}) : int64_t(0);
    };
    const int target = state_->pvwSourceSwitcherIndex();
    const int64_t target_pts = pts(target);
    const auto mode = state_->transition_mode.load();
    if (destinationReady(frames[target])) state_->destination_ready = true;
    if (mode == MixerState::TransitionMode::Cut) {
        if (destinationReady(frames[target])) {
            if (target_pts <= last_ns) return -1;
            completeTake(target_pts);
            return target;
        }
    } else if (mode == MixerState::TransitionMode::Crossfade) {
        if (state_->take_phase == MixerState::TakePhase::Ready) {
            if (!destinationReady(frames[target])) return active;
            if (target_pts <= last_ns) return -1;
            auto snapshot = outputSnapshot();
            {
                std::lock_guard<std::mutex> lock(snapshot->mutex);
                if (snapshot->frames.holding()) return active;
            }
            const double duration = double(state_->take_duration_ns) / 1000000000;
            const double hold = av_q2d({state_->fps_den, state_->fps_num}) / duration;
            state_->take_start_ns = target_pts;
            for (const auto& command : state_->transition_control({target_pts / 1000000, duration,
                    !state_->pgm_is_slot_a, state_->take_curve, state_->take_dip, hold, state_->canvas_transfer}))
                setNodeObject(state_->source_switcher_name + "_transition", command.key, command.value);
            // Keep direct frames available throughout the fade. One selector owns
            // the visible change; routing cleanup follows its endpoint frame.
            setNodeObject(state_->slot_a.post_otm_name, "outputs", Parameters(3u));
            setNodeObject(state_->slot_b.post_otm_name, "outputs", Parameters(3u));
            state_->take_phase = MixerState::TakePhase::Fade;
        }
        const int blend = MixerState::transSourceSwitcherIndex();
        const auto blend_pts = pts(blend);
        if (blend_pts > last_ns && blend_pts >= state_->take_start_ns) {
            if (blend_pts >= state_->take_start_ns + state_->take_duration_ns) {
                completeTake(blend_pts);
                // Emit this endpoint (already entirely the destination). The next
                // invocation selects the new program's direct input.
            }
            return blend;
        }
        return -1; // wait for the prepared blend, without letting direct frames overtake it
    } else if (mode == MixerState::TransitionMode::Wipe &&
               state_->take_phase == MixerState::TakePhase::Wipe && !state_->wipe_switched &&
               destinationReady(frames[target]) &&
               target_pts >= state_->take_start_ns + state_->take_duration_ns / 2) {
        if (target_pts <= last_ns) return -1;
        state_->wipe_switched = true;
        applySceneControls(state_->scenes.at(state_->transition_scene_name));
        return target;
    }
    return active;
}

int MixerOrchestrator::selectFrame(bool wipe, const std::vector<const av::VideoFrame*>& frames,
                                    int active, int64_t last_ns) {
    if (!nodes_->shouldWork()) return active;
    if (state_->transition_mode == MixerState::TransitionMode::Idle)
        return wipe ? 0 : state_->pgmSourceSwitcherIndex();
    try {
        const auto mode = state_->transition_mode.load();
        const int selected = wipe ? selectWipe(frames, active, last_ns) : selectProgram(frames, active, last_ns);
        // Wall time only bounds failure. Progress and completion use actual
        // media frames, including under input stalls or pipeline backpressure.
        if (state_->transition_mode != MixerState::TransitionMode::Idle && monotonicNs() > state_->take_deadline_ns)
            throw Error("mixer: transition stopped producing usable frames");
        if (state_->transition_mode != MixerState::TransitionMode::Idle &&
            selected >= 0 && size_t(selected) < frames.size() && frames[selected] &&
            frames[selected]->pts().isValid()) {
            const auto pts = frames[selected]->pts().timestamp({1, 1000000000});
            if (pts > last_ns && ((mode == MixerState::TransitionMode::Crossfade && selected == 2) ||
                                 (mode == MixerState::TransitionMode::Wipe && wipe && selected == 1)))
                state_->take_deadline_ns = monotonicNs() + kTakeReadyTimeoutMs * 1000000;
        }
        return selected;
    } catch (const std::exception& e) {
        logstream << "mixer: " << e.what();
        abortTransition(state_->transition_generation);
        return wipe ? 0 : state_->pgmSourceSwitcherIndex();
    }
}

} // namespace avp::mixer
