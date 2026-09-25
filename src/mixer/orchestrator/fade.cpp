// Crossfade through the preheated transition node, with the deferred
// routing flip once the last blended frame has been presented.
#include "internal.hpp"

namespace avp::mixer {

// ---------------------------------------------------------------------------
// deferredCleanup: flips internal bookkeeping and deletes nodes that can't be
// removed via timeline. The scheduler decides when this runs.
// ---------------------------------------------------------------------------
void MixerOrchestrator::deferredCleanup(
        const std::shared_ptr<NodeManager>& nodes,
        const std::shared_ptr<MixerState>& state,
        const std::shared_ptr<SharedTimeline>& timeline,
        const std::shared_ptr<TransitionScheduler>& scheduler,
        uint64_t transition_generation,
        bool new_pgm_is_slot_a,
        std::string new_pgm_scene,
        int64_t end_pts_ms) {
    if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Crossfade))
        return;

    // Media time can lag wall time by the configured playout budget. Finish
    // only after the selector has produced the first frame at/after the end,
    // then release the controls immediately. Do not park the scheduler worker
    // while waiting: other commands and shutdown must remain responsive.
    const auto presented = edgeLastTsIfExists(nodes,
        firstDstEdgeName(nodes, state->source_switcher_name));
    if (!presented.isValid() || presented < av::Timestamp(end_pts_ms, {1, 1000})) {
        scheduler->postAfter("mixer.fade.presented", 2,
            [nodes, state, timeline, scheduler, transition_generation,
             new_pgm_is_slot_a, new_pgm_scene, end_pts_ms] {
                deferredCleanup(nodes, state, timeline, scheduler, transition_generation,
                                new_pgm_is_slot_a, new_pgm_scene, end_pts_ms);
            });
        return;
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    if (!transitionIsCurrent(state, transition_generation, MixerState::TransitionMode::Crossfade))
        return;
    try {
        MixerOrchestrator orch(nodes, state, timeline, scheduler);
        orch.applyPostTransitionRouting(new_pgm_is_slot_a, new_pgm_scene);
        orch.finishSnapshot();
    } catch (const std::exception& e) {
        logstream << "mixer: deferred cleanup error restoring routing: " << e.what();
    }
    state->pgm_is_slot_a = new_pgm_is_slot_a;
    state->pgm_scene_name = std::move(new_pgm_scene);
    state->pvw_scene_name = "";
    state->transition_mode = MixerState::TransitionMode::Idle;
}

// ---------------------------------------------------------------------------
// fade: crossfade transition through the permanent preheated node.
// All timeline values are computed from the pre-flip state.
// ---------------------------------------------------------------------------
void MixerOrchestrator::fade(const std::string& scene_name, double duration_sec,
                             int64_t start_pts_ms) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!state_->scenes.count(scene_name)) throw Error("mixer: unknown scene: " + scene_name);
    if (!std::isfinite(duration_sec) || duration_sec <= 0) throw Error("mixer: invalid fade duration");
    const auto start = resolveTransitionStartPts(start_pts_ms);
    interruptTransition();
    state_->transition_mode = MixerState::TransitionMode::Crossfade;
    const auto generation = ++state_->transition_generation;
    state_->transition_scene_name = scene_name;
    TransitionGuard guard([&] { abortTransition(generation); });
    cutInternal(scene_name, start);
    const auto initial = edgeLastTsIfExists(nodes_, firstDstEdgeName(nodes_, state_->pvwSlot().post_otm_name));
    postTransitionTask("mixer.fade.ready", 0,
        [orch = *this, scene_name, duration_sec, start, generation, initial]() mutable {
            orch.startFadeWhenReady(scene_name, duration_sec, start, generation, initial, wallclock.pts() + 2000);
        });
    guard.release();
}

void MixerOrchestrator::startFadeWhenReady(const std::string& scene_name, double duration_sec,
        int64_t requested_pts, uint64_t generation, av::Timestamp initial_ts, int64_t deadline_ms) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!transitionIsCurrent(state_, generation, MixerState::TransitionMode::Crossfade)) return;
    TransitionGuard guard([&] { abortTransition(generation); });
    const auto ready = edgeLastTsIfExists(nodes_, firstDstEdgeName(nodes_, state_->pvwSlot().post_otm_name));
    auto snapshot = InstanceSharedObjects<avp::mixer::OutputSnapshot>::get(
        nodes_->instanceData(), state_->source_switcher_name + "_snapshot");
    bool output_held;
    {
        std::lock_guard<std::mutex> snapshot_lock(snapshot->mutex);
        output_held = snapshot->frames.holding();
    }
    if (output_held || !ready.isValid() || (initial_ts.isValid() && ready <= initial_ts)) {
        if (wallclock.pts() >= deadline_ms) {
            throw Error("mixer.fade: target scene did not produce a fresh frame within 2 seconds");
        }
        postTransitionTask("mixer.fade.ready", 2,
            [orch = *this, scene_name, duration_sec, requested_pts, generation, initial_ts, deadline_ms]() mutable {
                orch.startFadeWhenReady(scene_name, duration_sec, requested_pts, generation, initial_ts, deadline_ms);
            });
        guard.release();
        return;
    }
    startFade(scene_name, duration_sec, std::max(requested_pts, wallclock.pts()), generation);
    guard.release();
}

void MixerOrchestrator::startFade(const std::string& scene_name, double duration_sec,
                                 int64_t start_ms, uint64_t transition_generation) {
    // Capture all needed values from pre-flip state
    bool pvw_is_slot_a = !state_->pgm_is_slot_a;
    uint32_t pvw_bit = state_->pvwOutputBit();
    const auto& target_slot = pvw_is_slot_a ? state_->slot_a : state_->slot_b;
    const auto& old_slot    = pvw_is_slot_a ? state_->slot_b : state_->slot_a;
    int pvw_sw_idx = state_->pvwSourceSwitcherIndex();

    auto& target_scene = state_->scenes.at(scene_name);
    scheduleSceneControls(target_scene, start_ms);

    // 2. Update the preheated transition while its input branches are idle.
    // Legacy callers may construct MixerState without mixer.init.
    const auto control = state_->transition_control ? state_->transition_control : transitionControl("cuda");
    const auto command = control({start_ms, duration_sec, pvw_is_slot_a});
    const std::string transition_node_name = !state_->transition_node_name.empty()
        ? state_->transition_node_name
        : (state_->source_switcher_name.empty() ? transition_node_name_
                                               : state_->source_switcher_name + "_transition");
    setNodeObject(transition_node_name, command.key, command.value);

    // 3. Camera routing: applied in loadSceneIntoSlot via rewriteCameraOutputsForSlot

    // 4–5. Timeline: priming post-scene otms (direct+trans) then visible-path switches
    int64_t prep_ms = wallclock.pts();
    timeline_->set(state_->slot_a.post_otm_name, "outputs", prep_ms, Parameters(3u)); // 0b11 warmup
    timeline_->set(state_->slot_b.post_otm_name, "outputs", prep_ms, Parameters(3u));

    int64_t end_ms = start_ms + (int64_t)(duration_sec * 1000);

    // At start_ms: switch output to transition
    timeline_->set(state_->source_switcher_name, "active", start_ms,
                   Parameters(MixerState::transSourceSwitcherIndex()));
    timeline_->set(state_->slot_a.post_otm_name, "outputs", start_ms, Parameters(2u)); // 0b10 trans only
    timeline_->set(state_->slot_b.post_otm_name, "outputs", start_ms, Parameters(2u));

    // At end_ms: switch output to new PGM direct
    timeline_->set(state_->source_switcher_name, "active", end_ms, Parameters(pvw_sw_idx));
    timeline_->set(target_slot.post_otm_name, "outputs", end_ms, Parameters(1u));  // 0b01 direct only
    timeline_->set(old_slot.post_otm_name, "outputs", end_ms, Parameters(0u));     // idle

    // Camera cleanup at cleanup_ms: converge to new-PGM-only bitmasks
    // pvw_bit == post-flip PGM bit (the PVW slot becomes the new PGM)
    int64_t cleanup_ms = end_ms + 100;
    for (const auto& [src_name, info] : state_->sources) {
        if (info.routed)
            continue;
        uint32_t new_mask = target_scene.sources.count(src_name) ? pvw_bit : 0u;
        timeline_->set(info.otm_node_name, "outputs", cleanup_ms, Parameters(state_->sourceOutputMask(info, new_mask)));
    }
    publishRoutedRoutesForProgramOnly(pvw_is_slot_a, target_scene, cleanup_ms, false);
    timeline_->set(old_slot.compositor_name, "active_inputs", cleanup_ms, Parameters(0u));

    // 6. Deferred state/routing cleanup. The transition node stays hot.
    int64_t flip_delay = end_ms - wallclock.pts();
    postTransitionTask("mixer.fade.cleanup", flip_delay,
        [nodes = nodes_, state = state_, timeline = timeline_, scheduler = scheduler_,
         transition_generation, pvw_is_slot_a, scene_name, end_ms] {
            deferredCleanup(nodes, state, timeline, scheduler, transition_generation,
                            pvw_is_slot_a, scene_name, end_ms);
        });
}

}  // namespace avp::mixer
