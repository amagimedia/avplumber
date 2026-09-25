// Hard cut: arm the PVW direct branch, then flip routing once the ready edge
// has produced a fresh frame at or after the scheduled PTS.
#include "internal.hpp"

namespace avp::mixer {

// ---------------------------------------------------------------------------
// cutInternal: the graph-level work for a hard cut, without touching
// transition_mode or pgm_is_slot_a.  All values are read from pre-flip state.
// Caller must hold state_->mutex.
// Returns the earliest cut PTS (wallclock ms). Cold cuts are gated until the
// incoming direct edge has produced a fresh frame; preloaded PVW cuts only wait
// for the scheduled PTS.
// ---------------------------------------------------------------------------
int64_t MixerOrchestrator::cutInternal(const std::string& scene_name, int64_t start_pts_ms, bool warm_cut) {
    bool pvw_is_slot_a = !state_->pgm_is_slot_a;

    if (state_->pvw_scene_name == scene_name) {
        logstream << "mixer cut: reusing preloaded PVW scene=" << scene_name;
    } else {
        loadSceneIntoSlot(pvw_is_slot_a, scene_name, warm_cut);
    }

    int64_t prep_ms = wallclock.pts();
    int64_t cut_ms = start_pts_ms;
    const auto& new_slot = state_->pvwSlot();

    // Pre-warm the hidden direct branch before the visible `out_sel` switch. Without this,
    // enabling `post_otm` and switching `out_sel` at the same PTS leaves the newly selected
    // path one pipeline-latency late, so the final encoder-side force_fps repeats the last
    // visible frame for a few ticks across the cut. `one_to_many` with timeline runs in
    // drop_dynamic_ mode, so feeding an inactive direct branch here is safe: `source_switcher`
    // drains and drops those pre-roll frames instead of back-pressuring the slot.
    timeline_->clearKey(new_slot.post_otm_name, "outputs");
    setNodeObject(new_slot.post_otm_name, "outputs", Parameters(1u));
    timeline_->set(new_slot.post_otm_name, "outputs", prep_ms, Parameters(1u));

    logstream << "mixer cut armed: scene=" << scene_name << " earliest cut_ms=" << cut_ms
              << " post_otm prep " << new_slot.post_otm_name << "->1";

    resetSlotNormFps(nodes_, *state_);

    return cut_ms;
}

void MixerOrchestrator::readyCutTask(
        const std::shared_ptr<NodeManager>& nodes,
        const std::shared_ptr<MixerState>& state,
        const std::shared_ptr<SharedTimeline>& timeline,
        const std::shared_ptr<TransitionScheduler>& scheduler,
        uint64_t transition_generation,
        bool new_pgm_is_slot_a,
        std::string new_pgm_scene,
        const std::string& ready_edge_name,
        av::Timestamp ready_edge_initial_ts,
        int64_t earliest_switch_pts_ms,
        bool require_new_ready_frame) {
    int64_t waited_ms = 0;
    while (true) {
        if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Cut))
            return;
        const bool time_ready = wallclock.pts() >= earliest_switch_pts_ms;
        bool edge_ready = !require_new_ready_frame;
        if (require_new_ready_frame) {
            auto edge = nodes->edges()->findAny(ready_edge_name);
            if (!edge) {
                logstream << "mixer ready cut: missing ready edge " << ready_edge_name
                          << " for scene=" << new_pgm_scene;
                std::lock_guard<std::mutex> lock(state->mutex);
                MixerOrchestrator(nodes, state, timeline, scheduler).abortTransition(transition_generation);
                return;
            }
            av::Timestamp ts = edge->lastTS();
            edge_ready = ts.isValid() && (!ready_edge_initial_ts.isValid() || ts > ready_edge_initial_ts);
        }
        if (edge_ready && time_ready)
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(kPollMs));
        waited_ms += kPollMs;
    }
    if (waited_ms > 0 || require_new_ready_frame) {
        logstream << "mixer ready cut: scene=" << new_pgm_scene
                  << " waited_ms=" << waited_ms
                  << " require_new_ready_frame=" << (require_new_ready_frame ? "true" : "false");
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Cut))
        return;
    try {
        MixerOrchestrator orch(nodes, state, timeline, scheduler);
        if (state->cut_latency) state->cut_latency->timing.arm();
        orch.applyPostTransitionRouting(new_pgm_is_slot_a, new_pgm_scene);
        orch.finishSnapshot();
    } catch (const std::exception& e) {
        if (state->cut_latency) state->cut_latency->timing.cancel("failed");
        logstream << "mixer: ready cut error restoring routing: " << e.what();
    }
    state->pgm_is_slot_a = new_pgm_is_slot_a;
    state->pgm_scene_name = std::move(new_pgm_scene);
    state->pvw_scene_name = "";
    state->transition_mode = MixerState::TransitionMode::Idle;
}

// ---------------------------------------------------------------------------
// cut: PTS-scheduled hard cut.  Graph work + timeline entries happen now;
// state flip is deferred until the timeline entries have taken effect.
// ---------------------------------------------------------------------------
void MixerOrchestrator::cut(const std::string& scene_name, int64_t start_pts_ms,
                            avp::mixer::CutLatency::Clock::time_point received) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!state_->scenes.count(scene_name))
        throw Error("mixer: unknown scene: " + scene_name);
    int64_t cut_ms = resolveTransitionStartPts(start_pts_ms);
    interruptTransition();
    state_->transition_mode = MixerState::TransitionMode::Cut;
    uint64_t transition_generation = ++state_->transition_generation;
    state_->transition_scene_name = scene_name;
    TransitionGuard prep_guard([&] { abortTransition(transition_generation); });
    bool pvw_is_slot_a = !state_->pgm_is_slot_a;
    bool was_preloaded = state_->pvw_scene_name == scene_name;
    if (state_->cut_latency)
        state_->cut_latency->timing.begin(scene_name, was_preloaded, pvw_is_slot_a ? 0 : 1, received);

    scheduleSceneControls(state_->scenes.at(scene_name), cut_ms);
    cutInternal(scene_name, cut_ms, true);

    const auto& new_slot = state_->pvwSlot();
    std::string ready_edge_name = firstDstEdgeName(nodes_, new_slot.post_otm_name);
    auto ready_edge = nodes_->edges()->findAny(ready_edge_name);
    av::Timestamp ready_edge_initial_ts = ready_edge ? ready_edge->lastTS() : NOTS;
    postTransitionTask("mixer.cut.ready", cut_ms - wallclock.pts(),
        [nodes = nodes_, state = state_, timeline = timeline_, scheduler = scheduler_,
         transition_generation, pvw_is_slot_a, scene_name, ready_edge_name,
         ready_edge_initial_ts, cut_ms, was_preloaded] {
            readyCutTask(nodes, state, timeline, scheduler, transition_generation,
                         pvw_is_slot_a, scene_name, ready_edge_name, ready_edge_initial_ts,
                         cut_ms, !was_preloaded);
        });
    prep_guard.release();
}

}  // namespace avp::mixer
