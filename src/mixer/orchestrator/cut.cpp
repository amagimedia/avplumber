#include "internal.hpp"
#include "../move_plan.hpp"

namespace avp::mixer {

void MixerOrchestrator::prepareScene(const std::string& scene_name) {
    avp::timeline::Scope prepare("prepare scene");
    if (state_->pvw_slot_scene != scene_name)
        loadSceneIntoSlot(!state_->pgm_is_slot_a, scene_name, true);
    setNodeObject(state_->pvwSlot().post_otm_name, "outputs", Parameters(1u));
}

void MixerOrchestrator::beginTake(const std::string& scene_name, MixerState::TransitionMode mode) {
    if (!state_->scenes.count(scene_name)) throw Error("mixer: unknown scene: " + scene_name);
    interruptTransition(Interruption::Replaced);
    state_->graph = nodes_;
    state_->transition_mode = mode;
    state_->timeline_take = avp::timeline::start(std::string(state_->takeKind()) + " " + scene_name);
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
    // The glide starts on the take's first on-air frame; any other take ends a glide on air.
    const Parameters zoom = std::exchange(state_->take_zoom, Parameters(nullptr));
    if (!zoom.is_null() && zoom.value("generation", uint64_t(0)) == state_->transition_generation) {
        setNodeObject((destination_a ? state_->slot_a : state_->slot_b).compositor_name, "zoom_on_air",
                      {{"generation", zoom.at("generation")}, {"start_ns", pts_ns}});
        state_->pgm_zoom = zoom;
        state_->pgm_zoom["start_ns"] = pts_ns;
        state_->pgm_zoom_end_ns = monotonicNs() + zoom.at("duration_ns").get<int64_t>();
        state_->last_zoom["start_ns"] = pts_ns;
    } else {
        state_->pgm_zoom = nullptr;
    }
    const std::string scene = state_->transition_scene_name;
    if (state_->wipe_on_air) {
        setNodeObject(state_->wipe_otm_name, "outputs", Parameters(1u));
        retireWipeChain();
    }
    state_->blend_on_air = state_->wipe_on_air = state_->move_on_air = state_->take_paused = false;
    state_->take_move = nullptr;   // the slot drops it at the clock's end
    state_->take_request.reset();
    endTakeClock(TakeClock::End::Completed, pts_ns);
    // A take the lever completed leaves it engaged at that end: moving it back starts the next.
    if (state_->take_lever_end == LeverEnd::None) state_->lever.release();
    state_->take_manual = false;
    state_->take_lever_end = LeverEnd::None;
    state_->take_progress = 0;
    state_->publishTakePreview(scene, pts_ns);
    const auto& definition = state_->scenes.at(scene);
    applySceneControls(definition);
    applyPostTransitionRouting(destination_a, definition, true, pts_ns);
    finishTransition(destination_a, scene, pts_ns);
}

void MixerOrchestrator::cut(const std::string& scene_name, CutLatency::Clock::time_point received,
                            std::optional<ZoomRequest> zoom) {
    avp::timeline::mark("cut pressed: " + scene_name);   // before the lock: its wait shows
    std::lock_guard<std::mutex> lock(state_->mutex);
    const bool preloaded = state_->pvw_slot_scene == scene_name;
    if (!state_->scenes.count(scene_name)) throw Error("mixer: unknown scene: " + scene_name);
    const uint64_t generation = state_->transition_generation + 1;   // beginTake's
    if (zoom) {
        std::string why;
        Parameters spec = state_->takeOnAir()
            ? Parameters(nullptr) : planCutZoom(scene_name, *zoom, generation, why);
        if (spec.is_null() && why.empty()) why = "a transition is on air";
        state_->last_zoom = {{"generation", generation}, {"scene", scene_name}, {"result", spec.is_null() ? "cut" : "zoom"},
                             {"reason", why}};
        state_->pending_zoom = std::move(spec);
    }
    if (actOnTransitionOnAir(scene_name, MixerState::TransitionMode::Cut)) return;
    // A zoom travels in the destination's composition: reload even a preloaded Preview, or a frame
    // of the composition without it could complete the take showing the bare target framing.
    if (!state_->pending_zoom.is_null()) state_->pvw_slot_scene.clear();
    try {
        beginTake(scene_name, MixerState::TransitionMode::Cut);
        state_->take_zoom = std::exchange(state_->pending_zoom, Parameters(nullptr));
        state_->take_received_ns = monotonicNs(received);
        if (state_->cut_latency) {
            state_->cut_latency->timing.begin(scene_name, preloaded, state_->pvwSourceSwitcherIndex(), received);
            state_->cut_latency->timing.arm();
        }
    } catch (...) {
        state_->pending_zoom = nullptr;
        abortTransition(state_->transition_generation);
        throw;
    }
}

Parameters MixerOrchestrator::planCutZoom(const std::string& scene_name, const ZoomRequest& request, uint64_t generation,
                                          std::string& why) {
    const auto& from = state_->pgmScene();
    const auto& to = state_->scenes.at(scene_name);
    if (from.sources.size() != 1 || to.sources.size() != 1) {
        why = "a zoom needs one layer in each scene";
        return nullptr;
    }
    const auto& [from_name, from_layer] = *from.sources.begin();
    const auto& [to_name, to_layer] = *to.sources.begin();
    const auto from_source = state_->sources.find(from_name), to_source = state_->sources.find(to_name);
    if (from_source == state_->sources.end() || to_source == state_->sources.end()) {
        why = "a scene layer names no source";
        return nullptr;
    }
    std::optional<SourcePicture> picture;
    if (const Parameters& p = to_source->second.picture; !p.is_null()) {
        const auto& r = p.at("reference");
        picture = SourcePicture{{r.at("x"), r.at("y"), r.at("w"), r.at("h")}, p.at("metadata_key").get<std::string>(),
                                p.value("mapped", std::vector<int>{})};
    }
    auto spec = planZoom(from_layer, to_layer, from_source->second.input_index, to_source->second.input_index,
                         picture ? &*picture : nullptr, why);
    if (!spec) return nullptr;
    // From a glide still on air: its own schedule is the from-endpoint; never a third level.
    if (!state_->pgm_zoom.is_null() && monotonicNs() < state_->pgm_zoom_end_ns) {
        if (state_->pgm_zoom.contains("from_zoom")) {
            why = "glide on glide";
            return nullptr;
        }
        spec->from_zoom = std::make_shared<ZoomSpec>(parseZoomSpec(state_->pgm_zoom));
    }
    spec->duration_ns = int64_t(request.duration_sec * 1e9);
    spec->curve = request.curve;
    spec->generation = generation;
    return zoomSpecJson(*spec);
}

void MixerOrchestrator::definePicture(const std::string& source, const Parameters& picture) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    const auto found = state_->sources.find(source);
    if (found == state_->sources.end()) throw Error("mixer.source_picture: unknown source " + source);
    // Validate by parsing it as a zoom would.
    const auto& r = picture.at("reference");
    if (r.at("w").get<int>() <= 0 || r.at("h").get<int>() <= 0 || !picture.at("metadata_key").is_string())
        throw Error("mixer.source_picture: reference {x, y, w, h} and metadata_key required");
    found->second.picture = picture;
}

bool MixerOrchestrator::actOnTransitionOnAir(const std::string& scene_name, MixerState::TransitionMode pressed) {
    if (!state_->takeOnAir()) return false;
    if (state_->cut_latency) state_->cut_latency->timing.cancel();
    const bool same_scene = scene_name == state_->transition_scene_name;
    const bool cut_under_move = pressed == MixerState::TransitionMode::Cut &&
                                state_->transition_mode == MixerState::TransitionMode::Move;
    // A cut under a move lands on the scene itself: its slot reloads without the glide, and the
    // first frame of that revision completes it (selectProgram's cut branch).
    if (!same_scene || cut_under_move) replacePreset(scene_name, /*keep_move=*/!cut_under_move);
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
    } else if (same_scene && (state_->transition_mode == MixerState::TransitionMode::Crossfade ||
                              state_->transition_mode == MixerState::TransitionMode::Move ||
                              (state_->transition_mode == MixerState::TransitionMode::Wipe && state_->take_request))) {
        toggleTakePause();
    } else if (same_scene) {
        logstream << "mixer: " << (pressed == MixerState::TransitionMode::Wipe ? "wipe" : "fade")
                  << " to " << scene_name << " ignored: the " << state_->takeKind() << " on air continues";
    }
    return true;
}

void MixerOrchestrator::replacePreset(const std::string& scene_name, bool keep_move) {
    if (state_->transition_mode == MixerState::TransitionMode::Move && keep_move && !state_->take_move.is_null()) {
        // The glide goes on to the new preset on the clock on air; a plan that cannot run shows
        // the preset as it is for the rest of the take.
        std::string why;
        state_->pending_move = planMove(*state_, state_->pgmScene(), state_->scenes.at(scene_name),
                                        state_->take_move.at("generation").get<uint64_t>(),
                                        state_->take_move.at("unmatched").get<std::string>(), why);
        if (!why.empty()) state_->last_move = {{"scene", scene_name}, {"result", "preset"}, {"reason", why}};
    }
    try {
        loadSceneIntoSlot(!state_->pgm_is_slot_a, scene_name, true);
    } catch (...) {
        state_->pending_move = nullptr;
        throw;
    }
    state_->take_move = std::exchange(state_->pending_move, Parameters(nullptr));
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

void MixerOrchestrator::toggleTakePause() {
    if (!state_->take_request) return;
    // The held mix never steps back (mixChangeNs()), nor a wipe's clip (a tick its player has not pushed).
    const int64_t at_ns = takeScheduleFromNs();
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
    logstream << "mixer: " << state_->takeKind() << " to " << state_->transition_scene_name
              << (state_->take_paused ? " paused" : " resumed");
    emitTransitionPaused(state_->take_paused, state_->take_progress, at_ns);
}

void MixerOrchestrator::emitTransitionPaused(bool paused, double t, int64_t pts_ns) {
    if (state_->audio_router_node_name.empty()) return;
    try {
        setNodeObject(state_->audio_router_node_name, "transition_paused",
                      {{"output", "program"}, {"paused", paused}, {"t", t}, {"pts", pts_ns}});
    } catch (const std::exception& e) {
        logstream << "mixer: transition_paused failed: " << e.what();
    }
}

} // namespace avp::mixer
