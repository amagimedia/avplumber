#include "internal.hpp"
#include "../primitives/ClipScrub.hpp"
#include "../../nodes/clip_cache/ClipCache.hpp"

namespace avp::mixer {

void MixerOrchestrator::fade(const std::string& scene_name, double duration_sec,
        FadeCurve curve, std::optional<std::array<uint8_t, 3>> dip, CutLatency::Clock::time_point received) {
    if (!std::isfinite(duration_sec) || duration_sec <= 0) throw Error("mixer: invalid fade duration");
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!state_->scenes.count(scene_name)) throw Error("mixer: unknown scene: " + scene_name);
    if (actOnTransitionOnAir(scene_name, MixerState::TransitionMode::Crossfade)) return;
    beginCrossfade(scene_name, duration_sec, curve, dip, monotonicNs(received));
}

void MixerOrchestrator::beginCrossfade(const std::string& scene_name, double duration_sec, FadeCurve curve,
                                       std::optional<std::array<uint8_t, 3>> dip, int64_t received_ns) {
    try {
        beginTake(scene_name, MixerState::TransitionMode::Crossfade);
        state_->take_received_ns = received_ns;
        state_->take_duration_ns = int64_t(duration_sec * 1000000000);
        state_->take_curve = curve;
        state_->take_dip = dip;
    } catch (...) {
        abortTransition(state_->transition_generation);
        throw;
    }
}

void MixerOrchestrator::lever(double position, double duration_sec, FadeCurve curve,
                              std::optional<std::array<uint8_t, 3>> dip, const std::string& wipe_file,
                              std::optional<double> span_ms) {
    if (!std::isfinite(position)) throw Error("mixer: invalid lever position");
    if (!std::isfinite(duration_sec) || duration_sec <= 0) throw Error("mixer: invalid fade duration");
    if (span_ms && !(*span_ms > 0 && std::isfinite(*span_ms))) throw Error("mixer: invalid lever span");
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->lever_take = {duration_sec, curve, dip, wipe_file};
    leverTo(position, span_ms);
}

void MixerOrchestrator::leverTo(double position, std::optional<double> span_ms) {
    const auto mode = state_->transition_mode.load();
    const bool manual = state_->take_manual && (mode == MixerState::TransitionMode::Crossfade ||
                                                mode == MixerState::TransitionMode::Wipe);
    // An AUTO, cut or wipe on air owns the program; once the lever reached an end, its positions
    // wait for the picture to get there: a late or repeated one never undoes that end.
    const auto step = leverStep(state_->lever, position, mode == MixerState::TransitionMode::Idle, manual,
                                manual && state_->take_lever_end != LeverEnd::None,
                                state_->blend_on_air || state_->wipe_on_air);
    switch (step.kind) {
        case LeverStep::Kind::Ignore: return;
        case LeverStep::Kind::Drop: cancelLeverTake(); return;
        case LeverStep::Kind::Start: {
            const std::string scene = state_->preview.pvw;
            if (scene.empty() || !state_->scenes.count(scene) || scene == state_->pgm_scene_name) {
                state_->lever.release();
                return;
            }
            const auto& take = state_->lever_take;
            if (take.wipe_file.empty()) {
                beginCrossfade(scene, take.duration_sec, take.curve, take.dip, monotonicNs());
                state_->take_manual = true;
            } else {
                // Checked before the take begins: a take that fails clears the Preview.
                if (!state_->wipeSubgraph() || !state_->wipeChainStaysRunning() ||
                    !InstanceSharedObjects<avp::clipcache::ClipCache>::get(
                        nodes_->instanceData(), state_->wipe_cache_store)->cached(take.wipe_file)) {
                    state_->lever.release();   // one error for the gesture, not one per position
                    throw Error(state_->wipeSubgraph() ? "mixer: the lever wipes only cached clips, not " + take.wipe_file
                                                       : "mixer: wipe requires the wipe subgraph");
                }
                armWipe(scene, take.wipe_file, std::nullopt);
            }
            logstream << "mixer: lever " << (take.wipe_file.empty() ? "mixes" : "wipes with " + take.wipe_file)
                      << " to " << scene;
            break;
        }
        case LeverStep::Kind::Move: break;
    }
    state_->take_lever = step.progress;
    state_->take_lever_end = step.end;
    // The take follows the lever once on air; selectProgram (a mix) and selectWipe bring it up there.
    // A wipe's clip follows it from the arm, so its overlay comes up where the lever is, not at its
    // first frame; below its switch point until on air, the background still Program.
    if (state_->blend_on_air || state_->wipe_on_air)
        moveLever(step.progress, span_ms);
    else if (state_->transition_mode == MixerState::TransitionMode::Wipe &&
             state_->take_phase == MixerState::TakePhase::WipeReady)
        moveLever(std::min(step.progress, kLeverWipeArmedAt), span_ms);
}

void MixerOrchestrator::cancelLeverTake() {
    interruptTransition(Interruption::Replaced);   // Program as it was, Preview kept
    logstream << "mixer: lever back at its start, program unchanged";
}

LeverEnd MixerOrchestrator::landLeverEnd(int64_t pts_ns) {
    if (state_->take_lever_end == LeverEnd::None || !state_->take_request ||
        !leverEndLanded(*state_->take_request, pts_ns))
        return LeverEnd::None;
    const LeverEnd end = state_->take_lever_end;
    if (end == LeverEnd::Cancel) {
        cancelLeverTake();
        resumeLever();
    }
    return end;
}

bool MixerOrchestrator::endTakeAt(int64_t pts_ns, bool ready, bool ran_out) {
    const LeverEnd landed = landLeverEnd(pts_ns);
    if (landed == LeverEnd::Cancel) return true;
    if (!ready || (landed != LeverEnd::Complete && !ran_out)) return false;
    completeTake(pts_ns);
    if (landed == LeverEnd::Complete) resumeLever();
    return true;
}

void MixerOrchestrator::resumeLever() {
    // A position that arrived while the end landed was only kept (leverStep); off that end it
    // starts the next take from where the lever is, as if it arrived now. Its failure is the next
    // take's, never the one that just ended on this frame.
    try {
        leverTo(state_->lever.position(), std::nullopt);
    } catch (const std::exception& e) {
        logstream << "mixer: " << e.what();
        state_->lever.release();
    }
}

void MixerOrchestrator::moveLever(double progress, std::optional<double> span_ms) {
    if (!state_->take_request) return;
    const int64_t now_ns = monotonicNs();
    const int64_t span_ns = leverSpanNs(span_ms, now_ns, state_->lever_at_ns,
                                        av_rescale(1000000000, state_->fps_den, state_->fps_num));
    state_->lever_at_ns = now_ns;
    const int64_t from_ns = takeScheduleFromNs();
    const FadeRequest request = rampTo(*state_->take_request, progress, from_ns / 1000000, (from_ns + span_ns) / 1000000);
    state_->take_hold_ns = *request.time_ceiling_ms * 1000000;
    state_->take_paused = true;
    state_->scheduleTake(request, false);
    applyTakeSchedule();
}

int64_t MixerOrchestrator::takeScheduleFromNs() const {
    if (state_->transition_mode != MixerState::TransitionMode::Wipe) return mixChangeNs();
    const TickGrid grid(av::Rational(state_->fps_num, state_->fps_den));
    return leverAnchorNs(mixChangeNs(), playerNextNs(grid, monotonicNs(), state_->wipe_scrub_delay_ns));
}

void MixerOrchestrator::applyTakeSchedule() {
    const FadeRequest& request = *state_->take_request;
    if (state_->transition_mode == MixerState::TransitionMode::Wipe)
        setNodeObject(state_->wipe_input_node_name, "scrub", scrubSchedule(rampTimesNs(request)));
    else
        sendAlpha(request);
    clampTakeClock();
}

void MixerOrchestrator::sendAlpha(const FadeRequest& request) {
    for (const auto& command : state_->transition_control(request))
        if (command.value.value("command", std::string()) == "alpha")
            setNodeObject(state_->source_switcher_name + "_transition", command.key, command.value);
}

void MixerOrchestrator::startTakeClock(int64_t start_ns, int64_t duration_ns, FadeCurve curve, double switch_at) {
    state_->publishClock(state_->clock()->next(start_ns, duration_ns, curve, switch_at));
}

void MixerOrchestrator::clampTakeClock() {
    TakeClock clock = *state_->clock();
    // Generation 0 is no take's: a lever wipe's clip moves from its arm, before its clock starts.
    if (clock.generation == 0 || clock.end != TakeClock::End::Running || !state_->take_request) return;
    clock.schedule = state_->take_schedule;
    state_->publishClock(clock);
}

void MixerOrchestrator::endTakeClock(TakeClock::End end, int64_t end_ns) {
    TakeClock clock = *state_->clock();
    if (clock.generation == 0 || clock.end != TakeClock::End::Running) return;
    clock.end = end;
    clock.end_ns = end_ns;
    state_->publishClock(clock);
}

int MixerOrchestrator::selectProgram(const std::vector<const av::VideoFrame*>& frames, int64_t last_ns) {
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
        // The keys going with the cut switch on the first frame of the new program: the next
        // tick's when this tick's destination frame is not emitted.
        const int64_t keys_ns = target_pts == last_ns
            ? target_pts + av_rescale(1000000000, state_->fps_den, state_->fps_num) : target_pts;
        // A cut that completes a mix on air ends the mix's clock; any other starts its own.
        if (state_->clock()->end != TakeClock::End::Running) startTakeClock(keys_ns, 0, FadeCurve::Linear);
        endTakeClock(TakeClock::End::Completed, keys_ns);
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
            FadeRequest request{target_pts / 1000000, duration,
                    !state_->pgm_is_slot_a, state_->take_curve, state_->take_dip, hold, state_->canvas_transfer};
            state_->take_paused = state_->take_manual;
            if (state_->take_manual) {
                // The lever's mix comes up at the lever's progress, not at the start of a ramp,
                // held from this frame: a lever already at one of its ends lands here.
                const int64_t at_ms = target_pts / 1000000;
                request.start_ms = at_ms - int64_t(std::llround(state_->take_lever * duration * 1000));
                request.time_floor_ms = request.time_ceiling_ms = at_ms;
                state_->take_hold_ns = at_ms * 1000000;
            }
            state_->scheduleTake(request, true);
            for (const auto& command : state_->transition_control(request))
                setNodeObject(state_->source_switcher_name + "_transition", command.key, command.value);
            startTakeClock(target_pts, state_->take_duration_ns,
                           state_->take_dip ? FadeCurve::Linear : state_->take_curve);
            clampTakeClock();
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
            state_->take_progress = state_->clock()->progress(blend_pts);
            // A paused mix holds, a manual one until the lever's ramp to one of its ends reaches
            // this frame; a replaced preset completes only once its picture is there. This frame
            // goes out either way: at a cancel it is Program alone, at the end entirely the
            // destination, and the next invocation selects the new program's direct input.
            endTakeAt(blend_pts, state_->destination_ready,
                      !state_->take_paused && blend_pts >= state_->take_start_ns + state_->take_duration_ns);
            return blend;
        }
        // The start tick was routed before blend inputs were enabled. Its
        // program picture is the zero-progress endpoint, even if it arrives
        // after the destination tick. Discarding it forces a duplicate downstream.
        const int program = state_->pgmSourceSwitcherIndex();
        if (pts(program) > last_ns && pts(program) <= state_->take_start_ns)
            return program;
        return -1; // wait for the prepared blend, without letting direct frames overtake it
    } else if (mode == MixerState::TransitionMode::Wipe && state_->take_request &&
               state_->take_phase == MixerState::TakePhase::Wipe) {
        // The lever's wipe, or AUTO running it on: the background is Preview where the schedule
        // is past the clip's switch point and Program before it, both ways, from the first frame
        // of a tick, whichever slot's comes first; the keys follow on the same schedule (TakeClock).
        const int program = state_->pgmSourceSwitcherIndex();
        const auto background = wipeBackground(state_->take_schedule, state_->wipe_switched, pts(program),
                                               target_pts, destinationReady(frames[target]), last_ns);
        if (background.preview != state_->wipe_switched) {
            state_->wipe_switched = background.preview;
            applySceneControls(background.preview ? state_->scenes.at(state_->transition_scene_name)
                                                  : state_->pgmScene());
        }
        if (!background.emit) return -1;   // never a tick already out, nor one of the other side
        return background.preview ? target : program;
    } else if (mode == MixerState::TransitionMode::Wipe &&
               state_->take_phase == MixerState::TakePhase::Wipe && !state_->wipe_switched &&
               destinationReady(frames[target]) && target_pts >= last_ns &&
               target_pts >= state_->take_start_ns + state_->take_duration_ns / 2) {
        state_->wipe_switched = true;
        // The keys going with a wipe clip switch where the background does, under the clip.
        startTakeClock(target_pts == last_ns
            ? target_pts + av_rescale(1000000000, state_->fps_den, state_->fps_num) : target_pts, 0, FadeCurve::Linear);
        applySceneControls(state_->scenes.at(state_->transition_scene_name));
        return target_pts == last_ns ? -1 : target;
    } else if (mode == MixerState::TransitionMode::Wipe && state_->wipe_switched) {
        return target;
    }
    // An armed wipe, or AUTO's before its switch point: never what the selector chose last, which
    // may be a tick consumed without a frame (-1) or the blend of a mix the lever just ended.
    return on_air();
}

int MixerOrchestrator::selectFrame(bool wipe, const std::vector<const av::VideoFrame*>& frames,
                                    int active, int64_t last_ns) {
    if (!nodes_->shouldWork()) return active;
    if (state_->transition_mode == MixerState::TransitionMode::Idle)
        return wipe ? 0 : state_->pgmSourceSwitcherIndex();
    try {
        const auto mode = state_->transition_mode.load();
        const int selected = wipe ? selectWipe(frames, active, last_ns) : selectProgram(frames, last_ns);
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
