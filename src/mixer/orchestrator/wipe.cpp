// Media wipe: pre-created wipe subgraph, overlay readiness, midpoint scene
// switch hidden behind the opaque wipe, tail drain and teardown.
#include "internal.hpp"

namespace avp::mixer {

namespace {
constexpr int64_t kWipeSwitchGraceMs = 500;
}

// ---------------------------------------------------------------------------
// runWipeMidpointAndCleanup:
// Phase 1 (midpoint): PVW slot prep + timeline source_switcher (hidden under opaque wipe).
// Phase 2 (end): routing cleanup, tear down wipe chain, flip state.
//
// Generation checks prevent an interrupted wipe from changing new routing.
// ---------------------------------------------------------------------------
void MixerOrchestrator::runWipeMidpointAndCleanup(
        std::shared_ptr<NodeManager> nodes,
        std::shared_ptr<MixerState> state,
        std::shared_ptr<SharedTimeline> timeline,
        std::shared_ptr<TransitionScheduler> scheduler,
        uint64_t transition_generation,
        std::string scene_name,
        bool new_pgm_is_slot_a,
        int64_t remaining_ms) {

    if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
        return;

    // --- Phase 1: midpoint - do invisible scene switch under the fully-opaque wipe ---
    try {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
            return;
        MixerOrchestrator orch(nodes, state, timeline, scheduler);

        // Keep a prewarmed scene intact at the wipe midpoint.
        if (state->pvw_scene_name != scene_name)
            orch.loadSceneIntoSlot(new_pgm_is_slot_a, scene_name);

        // Same post-scene OTM flip as cutInternal: out_sel will read PVW `sc*_direct`, so that slot's
        // `one_to_many` must have outputs=1. If it stays 0 (idle default), frames are popped from
        // norm_* with nowhere to go and the wipe path freezes. Stop the old PGM branch to avoid backup.
        int64_t Tw = wallclock.pts();
        const auto& new_slot = state->pvwSlot();
        const auto& old_slot = state->pgmSlot();
        timeline->clearKey(old_slot.post_otm_name, "outputs");
        orch.setNodeObject(old_slot.post_otm_name, "outputs", Parameters(0u));
        timeline->set(old_slot.post_otm_name, "outputs", Tw, Parameters(0u));
        timeline->clearKey(new_slot.post_otm_name, "outputs");
        orch.setNodeObject(new_slot.post_otm_name, "outputs", Parameters(1u));
        timeline->set(new_slot.post_otm_name, "outputs", Tw, Parameters(1u));

        // Switch source_switcher (invisible behind wipe overlay); timeline for consistency with other switches
        int sw = state->pvwSourceSwitcherIndex();
        timeline->set(state->source_switcher_name, "active", Tw, Parameters(sw));
        logstream << "mixer wipe midpoint: Tw=" << Tw << " scene=" << scene_name << " out_sel.active=" << sw
                  << " new_slot post_otm=" << new_slot.post_otm_name << " old_slot post_otm="
                  << old_slot.post_otm_name;
        resetSlotNormFps(nodes, *state);
    } catch (const std::exception& e) {
        logstream << "mixer: wipe midpoint error: " << e.what();
    }

    // --- Phase 2: wipe end – tear down and flip ---
    // Stage 2a: keep the overlay selected for the planned duration. Input EOF
    // only means the demuxer has read ahead to the end; queued decoded frames
    // can still contain much of the exit animation.
    bool hit_input_eof = false;
    if (remaining_ms > 0) {
        int64_t waited_ms = 0;
        constexpr int64_t kTailPollMs = 10;
        while (waited_ms < remaining_ms) {
            if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
                return;
            if (!hit_input_eof && !nodeWorkingIfExists(nodes, state->wipe_input_node_name)) {
                logstream << "mixer wipe: input EOF after " << waited_ms
                          << "ms of remaining tail; waiting for planned end";
                hit_input_eof = true;
            }
            int64_t step_ms = std::min<int64_t>(kTailPollMs, remaining_ms - waited_ms);
            std::this_thread::sleep_for(std::chrono::milliseconds(step_ms));
            waited_ms += step_ms;
        }
    }

    // Stage 2b: after source EOF, the tail of the wipe is still propagating through
    // wipe_demux → wipe_dec → wipe_fmt → wipe_rt → wipe_rt_fps → wipe_overlay (six
    // queues + filter internal buffering). Flipping `wipe_sel` now would cut those
    // tail frames. Wait until the last pre-overlay edge (`wipe_tail_edge`) has been
    // drained by the overlay, then give the overlay a short grace to emit the final
    // blended frames through `wipe_overlay_out` to `wipe_sel`.
    if (hit_input_eof && !state->wipe_tail_edge.empty()) {
        constexpr int64_t kWipeDrainTimeoutMs = 1000;
        constexpr int64_t kWipeDrainPollMs = 10;
        // Grace period for the overlay filter to emit any frames already buffered
        // in its filter graph after `wipe_tail_edge` drained. 120ms is ~3-4 frames
        // at 30fps and ~7-8 frames at 60fps; both are within the typical libavfilter
        // internal queue depth. If a future wipe overlay graph buffers more (e.g.
        // a multi-stage temporal filter), bump this together with kWipeDrainTimeoutMs.
        // Going below ~80ms risks cutting tail blended frames at 30fps.
        constexpr int64_t kWipeOverlayTailMs = 120;
        int64_t waited = 0;
        while (waited < kWipeDrainTimeoutMs) {
            if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
                return;
            if (edgeOccupiedIfExists(nodes, state->wipe_tail_edge) == 0)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(kWipeDrainPollMs));
            waited += kWipeDrainPollMs;
        }
        for (int64_t tail = 0; tail < kWipeOverlayTailMs; tail += 5) {
            if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe)) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        logstream << "mixer wipe: drained " << state->wipe_tail_edge << " in " << waited
                  << "ms + " << kWipeOverlayTailMs << "ms overlay tail";
    }

    try {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
            return;
        MixerOrchestrator orch(nodes, state, timeline, scheduler);

        int64_t Tw = wallclock.pts();
        if (!state->wipe_otm_name.empty()) {
            timeline->clearKey(state->wipe_otm_name, "outputs");
            orch.setNodeObject(state->wipe_otm_name, "outputs", Parameters(1u));
            timeline->set(state->wipe_otm_name, "outputs", Tw, Parameters(1u));
        }
        if (!state->wipe_selector_name.empty()) {
            timeline->clearKey(state->wipe_selector_name, "active");
            orch.setNodeObject(state->wipe_selector_name, "active", Parameters(0));
            timeline->set(state->wipe_selector_name, "active", Tw, Parameters(0));
        }
        logstream << "mixer wipe cleanup: Tw=" << Tw << " otm_final.outputs=1 wipe_sel.active=0"
                  << " stop_wipe_group_in_ms=" << kWipeSwitchGraceMs;

        uint32_t new_pgm_bit = state->pvwOutputBit();
        auto& scene = state->scenes.at(scene_name);
        for (const auto& [src_name, info] : state->sources) {
            if (info.routed)
                continue;
            uint32_t mask = scene.sources.count(src_name) ? new_pgm_bit : 0u;
            timeline->set(info.otm_node_name, "outputs", Tw, Parameters(state->sourceOutputMask(info, mask)));
        }
        orch.publishRoutedRoutesForProgramOnly(new_pgm_is_slot_a, scene, Tw, true);

        const auto& old_slot = state->pgmSlot();
        timeline->set(old_slot.compositor_name, "active_inputs", Tw, Parameters(0u));
    } catch (const std::exception& e) {
        logstream << "mixer: wipe cleanup error: " << e.what();
        if (transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
            state->transition_mode = MixerState::TransitionMode::Idle;
        return;
    }

    for (int64_t waited = 0; waited < kWipeSwitchGraceMs; waited += 5) {
        if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe)) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
        return;

    try {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
            return;
        MixerOrchestrator orch(nodes, state, timeline, scheduler);

        // Stop the pre-created wipe subgraph only after the direct-path switch
        // has had time to land on the frame timeline. Tearing it down at the
        // same wallclock instant as the switch starves wipe_sel/final_out for a
        // few ticks and the encoder-side force_fps visibly repeats the last wipe frame.
        if (!state->wipe_group_name.empty()) {
            orch.stopGroup(state->wipe_group_name);
            // Release any frames still sitting in wipe pipeline edges so they
            // don't replay at the start of the next wipe.
            orch.flushWipeEdges();
        }

        orch.finishSnapshot();
        state->pgm_is_slot_a = new_pgm_is_slot_a;
        state->pgm_scene_name = scene_name;
        state->pvw_scene_name = "";
        state->transition_mode = MixerState::TransitionMode::Idle;
    } catch (const std::exception& e) {
        logstream << "mixer: wipe teardown error: " << e.what();
        if (transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
            state->transition_mode = MixerState::TransitionMode::Idle;
    }
}

// ---------------------------------------------------------------------------
// wipe: media wipe transition.  Uses the static wipe_otm + wipe_selector
// nodes to route through the overlay without edge rewiring.
// The wipe subgraph (group wipe_group_name) is pre-created but not running
// in steady state; it is started here and stopped at the end of the wipe.
// ---------------------------------------------------------------------------
int64_t MixerOrchestrator::prepareWipe(
        std::shared_ptr<NodeManager> nodes,
        std::shared_ptr<MixerState> state,
        std::shared_ptr<SharedTimeline> timeline,
        std::shared_ptr<TransitionScheduler> scheduler,
        uint64_t transition_generation,
        std::string scene_name,
        std::string wipe_file,
        double duration_sec,
        bool new_pgm_is_slot_a,
        int64_t earliest_visible_pts_ms) {
    std::string overlay_edge_name;
    av::Timestamp overlay_initial_ts = NOTS;
    int64_t prep_ms = wallclock.pts();

    // Only the serialized transition worker reuses wipe nodes. Finish retiring
    // the previous clip outside the control mutex, then recheck cancellation.
    if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe)) return -1;
    try {
        nodes->group(state->wipe_group_name)->stopNodesAndWait();
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
            return -1;
        MixerOrchestrator orch(nodes, state, timeline, scheduler);

        overlay_edge_name = edgeNameAt(nodes, state->wipe_selector_name, "src", 1);
        overlay_initial_ts = edgeLastTsIfExists(nodes, overlay_edge_name);

        nodes->node(state->wipe_input_node_name)->stop(true);
        orch.setNodeParam(state->wipe_input_node_name, "url", wipe_file);
        orch.flushWipeEdges();
        resetInputIf(nodes, state->wipe_base_fps_name);
        orch.startGroup(state->wipe_group_name);

        prep_ms = wallclock.pts();
        timeline->clearKey(state->wipe_otm_name, "outputs");
        orch.setNodeObject(state->wipe_otm_name, "outputs", Parameters(3u));       // 0b11 both direct + wipe_in
        timeline->set(state->wipe_otm_name, "outputs", prep_ms, Parameters(3u));
        timeline->clearKey(state->wipe_selector_name, "active");
        orch.setNodeObject(state->wipe_selector_name, "active", Parameters(0));    // direct branch while prerolling
        timeline->set(state->wipe_selector_name, "active", prep_ms, Parameters(0));

        int64_t total_ms = (int64_t)(duration_sec * 1000);
        int64_t midpoint_ms = total_ms / 2;
        logstream << "mixer wipe: scene=" << scene_name << " file=" << wipe_file << " prep_ms=" << prep_ms
                  << " requested_visible=" << earliest_visible_pts_ms << " total_ms=" << total_ms << " midpoint_ms=" << midpoint_ms
                  << " new_pgm_slot_" << (new_pgm_is_slot_a ? 'A' : 'B');
    } catch (const std::exception& e) {
        logstream << "mixer: wipe prep error: " << e.what();
        std::lock_guard<std::mutex> lock(state->mutex);
        MixerOrchestrator(nodes, state, timeline, scheduler).abortTransition(transition_generation);
        return -1;
    }

    WipeReadyResult ready = waitForWipeOverlayReady(
        nodes, overlay_edge_name, overlay_initial_ts, earliest_visible_pts_ms, state, transition_generation);
    if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe)) return -1;
    if (!ready.ready) {
        logstream << "mixer: wipe overlay did not become ready within " << kWipeReadyTimeoutMs
                  << "ms: edge=" << overlay_edge_name << " last_ts=" << ready.ready_ts;
        std::lock_guard<std::mutex> lock(state->mutex);
        MixerOrchestrator(nodes, state, timeline, scheduler).abortTransition(transition_generation);
        return -1;
    }

    try {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Wipe))
            return -1;
        MixerOrchestrator orch(nodes, state, timeline, scheduler);
        int64_t visible_ms = wallclock.pts();
        orch.setNodeObject(state->wipe_selector_name, "active", Parameters(1));    // wipe_overlay_out
        timeline->set(state->wipe_selector_name, "active", visible_ms, Parameters(1));
        logstream << "mixer wipe overlay ready: edge=" << overlay_edge_name
                  << " waited_ms=" << ready.waited_ms
                  << " ready_ts=" << ready.ready_ts
                  << " visible_ms=" << visible_ms
                  << " requested_visible=" << earliest_visible_pts_ms;
        return visible_ms;
    } catch (const std::exception& e) {
        logstream << "mixer: wipe visible switch error: " << e.what();
        std::lock_guard<std::mutex> lock(state->mutex);
        MixerOrchestrator(nodes, state, timeline, scheduler).abortTransition(transition_generation);
        return -1;
    }
}

void MixerOrchestrator::warmupWipe(const std::string& wipe_file, int64_t timeout_ms) {
    std::string overlay_edge_name;
    av::Timestamp overlay_initial_ts = NOTS;
    uint64_t generation;
    const int64_t t0 = wallclock.pts();
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->wipe_otm_name.empty() || state_->wipe_selector_name.empty() ||
            state_->wipe_group_name.empty() || state_->wipe_input_node_name.empty())
            throw Error("mixer: wipe warm-up requires the wipe subgraph (see mixer.init)");
        if (state_->transition_mode.load() != MixerState::TransitionMode::Idle)
            throw Error("mixer: cannot warm up the wipe during a transition");
        generation = state_->transition_generation.load();
        overlay_edge_name = edgeNameAt(nodes_, state_->wipe_selector_name, "src", 1);
        overlay_initial_ts = edgeLastTsIfExists(nodes_, overlay_edge_name);
        nodes_->group(state_->wipe_group_name)->stopNodesAndWait();
        nodes_->node(state_->wipe_input_node_name)->stop(true);
        setNodeParam(state_->wipe_input_node_name, "url", wipe_file);
        flushWipeEdges();
        resetInputIf(nodes_, state_->wipe_base_fps_name);
        startGroup(state_->wipe_group_name);
        // Feed the overlay's program input as a real wipe would; the selector
        // stays on the direct branch so nothing of this reaches the output.
        timeline_->clearKey(state_->wipe_otm_name, "outputs");
        setNodeObject(state_->wipe_otm_name, "outputs", Parameters(3u));
    }
    WipeReadyResult ready = waitForWipeOverlayReady(nodes_, overlay_edge_name, overlay_initial_ts, 0,
                                                    state_, generation, timeout_ms);
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        timeline_->clearKey(state_->wipe_otm_name, "outputs");
        setNodeObject(state_->wipe_otm_name, "outputs", Parameters(1u));
        stopGroup(state_->wipe_group_name);
        flushWipeEdges();
    }
    logstream << "mixer wipe warm-up: file=" << wipe_file << (ready.ready ? " ready" : " NOT ready")
              << " after " << (wallclock.pts() - t0) << "ms (overlay waited " << ready.waited_ms << "ms)";
    if (!ready.ready)
        throw Error("mixer: wipe warm-up did not produce an overlay frame within " + std::to_string(timeout_ms) + "ms");
}

void MixerOrchestrator::wipe(const std::string& scene_name, const std::string& wipe_file, double duration_sec,
                             int64_t start_pts_ms) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!state_->scenes.count(scene_name))
        throw Error("mixer: unknown scene: " + scene_name);

    if (state_->wipe_otm_name.empty() || state_->wipe_selector_name.empty())
        throw Error("mixer: wipe requires wipe_otm and wipe_selector nodes (see mixer.init)");
    if (state_->wipe_group_name.empty() || state_->wipe_input_node_name.empty())
        throw Error("mixer: wipe requires wipe_group and wipe_input_node (see mixer.init)");

    if (!std::isfinite(duration_sec) || duration_sec <= 0) throw Error("mixer: invalid wipe duration");
    int64_t start_ms = resolveTransitionStartPts(start_pts_ms);
    interruptTransition();
    state_->transition_mode = MixerState::TransitionMode::Wipe;
    uint64_t transition_generation = ++state_->transition_generation;
    state_->transition_scene_name = scene_name;
    TransitionGuard prep_guard([&] { abortTransition(transition_generation); });
    bool pvw_is_slot_a = !state_->pgm_is_slot_a;
    cutInternal(scene_name, start_ms);
    scheduleSceneControls(state_->scenes.at(scene_name), start_ms);

    int64_t total_ms = (int64_t)(duration_sec * 1000);
    int64_t midpoint_ms = total_ms / 2;
    int64_t remaining_ms = total_ms - midpoint_ms;
    int64_t now_ms = wallclock.pts();
    postTransitionTask("mixer.wipe.prepare", start_ms - now_ms,
        [scheduler = scheduler_, nodes = nodes_, state = state_, timeline = timeline_, transition_generation,
         scene_name, wipe_file, duration_sec, pvw_is_slot_a, start_ms, midpoint_ms, remaining_ms] {
            int64_t visible_ms = prepareWipe(nodes, state, timeline, scheduler, transition_generation,
                                            scene_name, wipe_file, duration_sec, pvw_is_slot_a,
                                            start_ms);
            if (visible_ms < 0)
                return;

            scheduler->postAfter("mixer.wipe.midpoint", midpoint_ms,
                [nodes, state, timeline, scheduler, transition_generation, scene_name, pvw_is_slot_a, remaining_ms] {
                    runWipeMidpointAndCleanup(nodes, state, timeline, scheduler, transition_generation,
                                              scene_name, pvw_is_slot_a, remaining_ms);
                });
        });
    prep_guard.release();
}

}  // namespace avp::mixer
