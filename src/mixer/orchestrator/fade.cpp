#include "internal.hpp"

namespace avp::mixer {

void MixerOrchestrator::fade(const std::string& scene_name, double duration_sec,
        FadeCurve curve, std::optional<std::array<uint8_t, 3>> dip, CutLatency::Clock::time_point received) {
    if (!std::isfinite(duration_sec) || duration_sec <= 0) throw Error("mixer: invalid fade duration");
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!state_->scenes.count(scene_name)) throw Error("mixer: unknown scene: " + scene_name);
    if (actOnTransitionOnAir(scene_name, MixerState::TransitionMode::Crossfade)) return;
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
    // What program shows while a take waits: the transition on air, else the program slot.
    const auto on_air = [&] {
        if (state_->blend_on_air) return MixerState::transSourceSwitcherIndex();
        if (state_->wipe_on_air && state_->wipe_switched) return target;
        return state_->pgmSourceSwitcherIndex();
    };
    // A destination frame of the tick program has already shown completes a take without being
    // emitted (-1): the program slot's frame of a tick can reach the selector first on every
    // tick, so waiting for a strictly newer destination frame might never end.
    if (mode == MixerState::TransitionMode::Cut) {
        if (!destinationReady(frames[target]) || target_pts < last_ns) return on_air();
        state_->startTakeKeys(target_pts == last_ns
            ? target_pts + av_rescale(1000000000, state_->fps_den, state_->fps_num) : target_pts, true);
        completeTake(target_pts);
        return target_pts == last_ns ? -1 : target;
    } else if (mode == MixerState::TransitionMode::Crossfade) {
        if (state_->take_phase == MixerState::TakePhase::Ready) {
            // Preview can arrive behind program. Keep selecting the live program
            // until a destination tick no older than the last output can arm the blend.
            if (!destinationReady(frames[target]) || target_pts < last_ns)
                return state_->pgmSourceSwitcherIndex();
            const double duration = double(state_->take_duration_ns) / 1000000000;
            const double hold = av_q2d({state_->fps_den, state_->fps_num}) / duration;
            state_->take_start_ns = target_pts;
            state_->take_request = FadeRequest{target_pts / 1000000, duration,
                    !state_->pgm_is_slot_a, state_->take_curve, state_->take_dip, hold, state_->canvas_transfer};
            state_->take_paused = false;
            for (const auto& command : state_->transition_control(*state_->take_request))
                setNodeObject(state_->source_switcher_name + "_transition", command.key, command.value);
            // Keep direct frames available throughout the fade. One selector owns
            // the visible change; routing cleanup follows its endpoint frame.
            setNodeObject(state_->slot_a.post_otm_name, "outputs", Parameters(3u));
            setNodeObject(state_->slot_b.post_otm_name, "outputs", Parameters(3u));
            state_->take_phase = MixerState::TakePhase::Fade;
            state_->blend_on_air = true;
        }
        const int blend = MixerState::transSourceSwitcherIndex();
        const auto blend_pts = pts(blend);
        if (blend_pts > last_ns && blend_pts >= state_->take_start_ns) {
            state_->startTakeKeys(blend_pts, false);
            // A paused mix holds; a replaced preset completes only once its picture is there.
            if (!state_->take_paused && state_->destination_ready &&
                blend_pts >= state_->take_start_ns + state_->take_duration_ns) {
                completeTake(blend_pts);
                // Emit this endpoint (already entirely the destination). The next
                // invocation selects the new program's direct input.
            }
            return blend;
        }
        // The start tick was routed before blend inputs were enabled. Its
        // program picture is the zero-progress endpoint, even if it arrives
        // after the destination tick. Discarding it forces a duplicate downstream.
        const int program = state_->pgmSourceSwitcherIndex();
        if (pts(program) > last_ns && pts(program) <= state_->take_start_ns)
            return program;
        return -1; // wait for the prepared blend, without letting direct frames overtake it
    } else if (mode == MixerState::TransitionMode::Wipe &&
               state_->take_phase == MixerState::TakePhase::Wipe && !state_->wipe_switched &&
               destinationReady(frames[target]) && target_pts >= last_ns &&
               target_pts >= state_->take_start_ns + state_->take_duration_ns / 2) {
        state_->wipe_switched = true;
        applySceneControls(state_->scenes.at(state_->transition_scene_name));
        return target_pts == last_ns ? -1 : target;
    } else if (mode == MixerState::TransitionMode::Wipe && state_->wipe_switched) {
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
            // A blend whose replaced preset never renders does not keep the take alive.
            if (pts > last_ns && state_->destination_ready &&
                ((mode == MixerState::TransitionMode::Crossfade && selected == 2) ||
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
