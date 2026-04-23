#include "mixer_orchestrator.hpp"
#include "avutils.hpp"
#include <algorithm>
#include <cmath>
#include <thread>
#include <chrono>

namespace {

/// One cuda_rect_overlay layer per compositor src index (see mixer.source). Omitted sources use a dummy rect.
Parameters compositorLayersFromScene(const MixerState& st, const SceneDefinition& scene) {
    static const Parameters kUnusedLayer = Parameters({{"dst_x", 0}, {"dst_y", 0}});

    int max_idx = -1;
    for (const auto& [_, info] : st.sources)
        max_idx = std::max(max_idx, info.input_index);

    Parameters arr = Parameters::array();
    for (int i = 0; i <= max_idx; ++i) {
        std::string name_at;
        for (const auto& [name, info] : st.sources) {
            if (info.input_index == i) {
                name_at = name;
                break;
            }
        }
        if (name_at.empty()) {
            arr.push_back(kUnusedLayer);
            continue;
        }
        auto it = scene.sources.find(name_at);
        if (it == scene.sources.end())
            arr.push_back(kUnusedLayer);
        else
            arr.push_back(it->second.layer);
    }
    return arr;
}

} // namespace

MixerOrchestrator::MixerOrchestrator(
    std::shared_ptr<NodeManager> nodes,
    std::shared_ptr<MixerState> state,
    std::shared_ptr<SharedTimeline> timeline)
    : nodes_(std::move(nodes)), state_(std::move(state)), timeline_(std::move(timeline)) {}

void MixerOrchestrator::setNodeObject(const std::string& node_name, const std::string& key, const Parameters& value) {
    auto node = nodes_->node(node_name);
    node->setObject(key, value);
}

void MixerOrchestrator::publishCameraOtmOutputs(const std::string& otm_name, uint32_t mask) {
    // `one_to_many` with `timeline` uses tlGetRaw("outputs") whenever any entry matches; stale
    // rows (e.g. an old T_cleanup) would override setObject. We drop only the "outputs" key on
    // this OTM channel — not post-scene otms, not source_switcher, not other keys here.
    // cut/fade append new `outputs` rows at T_cleanup *after* loadSceneIntoSlot returns, so
    // those are not cleared by this call. Overlapping mixer commands are rejected by ensureIdle().
    timeline_->clearKey(otm_name, "outputs");
    setNodeObject(otm_name, "outputs", Parameters(mask));
    timeline_->set(otm_name, "outputs", wallclock.pts(), Parameters(mask));
}

void MixerOrchestrator::setNodeParam(const std::string& node_name, const std::string& param, const std::string& value) {
    auto node = nodes_->node(node_name);
    auto& params = node->parameters();
    params[param] = value;
}

void MixerOrchestrator::autoRestartNode(const std::string& node_name) {
    auto node = nodes_->node(node_name);
    node->stop(false);
}

void MixerOrchestrator::createAndStartNode(const Parameters& params) {
    Parameters p = params;
    nodes_->createNode(p, true, true);
}

void MixerOrchestrator::deleteNodeIfExists(const std::string& name) {
    auto node = nodes_->node_if_exists(name);
    if (node) {
        nodes_->deleteNode(name);
    }
}

void MixerOrchestrator::startGroup(const std::string& group_name) {
    nodes_->group(group_name)->startNodes();
}

void MixerOrchestrator::stopGroup(const std::string& group_name) {
    nodes_->group(group_name)->stopNodes();
}

void MixerOrchestrator::flushWipeEdges() {
    for (const auto& name : state_->wipe_flush_edges) {
        auto edge = nodes_->edges()->findAny(name);
        if (edge) edge->clear();
    }
}

void MixerOrchestrator::ensureIdle() const {
    auto mode = state_->transition_mode.load();
    if (mode != MixerState::TransitionMode::Idle)
        throw Error("mixer: transition already in progress");
}

void MixerOrchestrator::defineSource(const std::string& name, const std::string& otm_node, int input_index,
                                      const std::string& cs_node_a, const std::string& cs_node_b) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    MixerState::SourceInfo info;
    info.otm_node_name = otm_node;
    info.input_index = input_index;
    info.cs_node_a = cs_node_a;
    info.cs_node_b = cs_node_b;
    state_->sources[name] = std::move(info);
}

void MixerOrchestrator::defineScene(const std::string& name, const SceneDefinition& def) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    state_->scenes[name] = def;
}

void MixerOrchestrator::rewriteCameraOutputsForSlot(uint32_t slot_bit, const SceneDefinition& scene) {
    uint32_t active = state_->computeActiveInputsMask(scene);
    for (const auto& [src_name, info] : state_->sources) {
        Parameters current_val;
        uint32_t mask = nodes_->node(info.otm_node_name)->getObjectTry("outputs", current_val)
                            ? current_val.get<uint32_t>()
                            : 0u;
        mask &= ~slot_bit;
        if (scene.sources.count(src_name) && (active & (1u << (unsigned)info.input_index)))
            mask |= slot_bit;
        publishCameraOtmOutputs(info.otm_node_name, mask);
    }
}

void MixerOrchestrator::loadSceneIntoSlot(bool is_slot_a, const std::string& scene_name) {
    auto& scene = state_->scenes.at(scene_name);
    const auto& slot = is_slot_a ? state_->slot_a : state_->slot_b;

    for (const auto& [src_name, layout] : scene.sources) {
        auto src_it = state_->sources.find(src_name);
        if (src_it == state_->sources.end()) continue;
        const auto& info = src_it->second;
        const std::string& cs_node = is_slot_a ? info.cs_node_a : info.cs_node_b;

        // Only restart the crop/scale node when the graph string actually changed.
        // An unnecessary restart allocates a new hw_frames_ctx even with identical
        // parameters, propagating a spurious hw_frames_ctx-change event downstream.
        // For wipe transitions this ultimately causes wipe_overlay to rebuild its
        // filter graph mid-wipe, triggering a one-frame EXT_NULL gap (overlay
        // disappears for one frame at the midpoint).
        const auto& node_params = nodes_->node(cs_node)->parameters();
        const std::string old_graph = node_params.value("graph", std::string(""));
        if (old_graph == layout.crop_scale_graph) {
            logstream << "loadSceneIntoSlot: " << cs_node
                      << " graph unchanged – skipping restart to avoid spurious hw_frames_ctx change";
        } else {
            logstream << "loadSceneIntoSlot: " << cs_node
                      << " graph changed (\"" << old_graph << "\" -> \""
                      << layout.crop_scale_graph << "\") – restarting";
            setNodeParam(cs_node, "graph", layout.crop_scale_graph);
            autoRestartNode(cs_node);
        }
    }

    setNodeObject(slot.compositor_name, "layers", compositorLayersFromScene(*state_, scene));

    uint32_t active_mask = state_->computeActiveInputsMask(scene);
    // Same pattern as camera otms: cuda_rect_overlay reads "active_inputs" from timeline only.
    // clearKey does not touch "layers" or other keys on this compositor channel.
    timeline_->clearKey(slot.compositor_name, "active_inputs");
    setNodeObject(slot.compositor_name, "active_inputs", Parameters(active_mask));
    timeline_->set(slot.compositor_name, "active_inputs", wallclock.pts(), Parameters(active_mask));

    state_->pvw_scene_name = scene_name;

    // Drop slot bit for every camera, then enable only sources in scene with active_inputs set.
    // Keeps `outputs` consistent with compositor consumption (no frames into unused inputs).
    const uint32_t slot_bit = is_slot_a ? 1u : 2u;
    rewriteCameraOutputsForSlot(slot_bit, scene);
}

// ---------------------------------------------------------------------------
// cutInternal: the graph-level work for a hard cut, without touching
// transition_mode or pgm_is_slot_a.  All values are read from pre-flip state.
// Caller must hold state_->mutex.
// Returns the T_cleanup PTS (wallclock ms) so the caller can schedule the flip.
// ---------------------------------------------------------------------------
int64_t MixerOrchestrator::cutInternal(const std::string& scene_name) {
    bool pvw_is_slot_a = !state_->pgm_is_slot_a;

    loadSceneIntoSlot(pvw_is_slot_a, scene_name);

    auto& scene = state_->scenes.at(scene_name);
    uint32_t pvw_bit = state_->pvwOutputBit();

    int64_t T_cut = wallclock.pts() + margin_ms_;
    const auto& new_slot = state_->pvwSlot();
    const auto& old_slot = state_->pgmSlot();

    // Post-scene OTMs: must flip at the same PTS as `out_sel`. If the old slot's OTM keeps
    // forwarding to sc{AB}_direct for ~margin_ms after `out_sel` has switched, that edge is
    // unread (capacity 3) and backs up through norm → scene_out → comp — looks like clogging
    // "before" the compositor from the camera side.
    timeline_->clearKey(old_slot.post_otm_name, "outputs");
    setNodeObject(old_slot.post_otm_name, "outputs", Parameters(0u));
    timeline_->set(old_slot.post_otm_name, "outputs", T_cut, Parameters(0u));

    timeline_->clearKey(new_slot.post_otm_name, "outputs");
    setNodeObject(new_slot.post_otm_name, "outputs", Parameters(1u));
    timeline_->set(new_slot.post_otm_name, "outputs", T_cut, Parameters(1u));

    timeline_->set(state_->source_switcher_name, "active", T_cut,
                   Parameters(state_->pvwSourceSwitcherIndex()));

    int64_t T_cleanup = T_cut + 100;

    // Disable old PGM cameras at T_cleanup
    if (!state_->pgm_scene_name.empty()) {
        auto& old_scene = state_->scenes.at(state_->pgm_scene_name);
        for (const auto& [src_name, layout] : old_scene.sources) {
            auto it = state_->sources.find(src_name);
            if (it == state_->sources.end()) continue;
            uint32_t new_mask = scene.sources.count(src_name) ? pvw_bit : 0u;
            timeline_->set(it->second.otm_node_name, "outputs", T_cleanup, Parameters(new_mask));
        }
        timeline_->set(old_slot.compositor_name, "active_inputs", T_cleanup, Parameters(0u));
    }

    // Ensure new-scene cameras converge to pvw_bit at T_cleanup
    for (const auto& [src_name, layout] : scene.sources) {
        auto it = state_->sources.find(src_name);
        if (it == state_->sources.end()) continue;
        timeline_->set(it->second.otm_node_name, "outputs", T_cleanup, Parameters(pvw_bit));
    }

    return T_cleanup;
}

// ---------------------------------------------------------------------------
// deferredCleanup: runs on a detached thread.  Sleeps, then flips internal
// bookkeeping and deletes nodes that can't be removed via timeline.
// ---------------------------------------------------------------------------
void MixerOrchestrator::deferredCleanup(
        std::shared_ptr<NodeManager> nodes,
        std::shared_ptr<MixerState> state,
        int64_t sleep_ms,
        bool new_pgm_is_slot_a,
        std::string new_pgm_scene,
        std::vector<std::string> nodes_to_delete) {
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    for (const auto& name : nodes_to_delete) {
        try {
            auto node = nodes->node_if_exists(name);
            if (node) nodes->deleteNode(name);
        } catch (const std::exception& e) {
            logstream << "mixer: deferred cleanup error deleting " << name << ": " << e.what();
        }
    }
    std::lock_guard<std::mutex> lock(state->mutex);
    state->pgm_is_slot_a = new_pgm_is_slot_a;
    state->pgm_scene_name = std::move(new_pgm_scene);
    state->pvw_scene_name = "";
    state->transition_mode = MixerState::TransitionMode::Idle;
}

// ---------------------------------------------------------------------------
// cut: PTS-scheduled hard cut.  Graph work + timeline entries happen now;
// state flip is deferred until the timeline entries have taken effect.
// ---------------------------------------------------------------------------
void MixerOrchestrator::cut(const std::string& scene_name) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    ensureIdle();

    state_->transition_mode = MixerState::TransitionMode::Cut;
    bool pvw_is_slot_a = !state_->pgm_is_slot_a;

    int64_t T_cleanup = cutInternal(scene_name);

    int64_t flip_delay = (T_cleanup - wallclock.pts()) + 300;
    std::thread(deferredCleanup, nodes_, state_,
                flip_delay, pvw_is_slot_a, scene_name,
                std::vector<std::string>{}).detach();
}

// ---------------------------------------------------------------------------
// fade: crossfade transition.  All timeline values are computed from the
// pre-flip state.  The state flip + transition node deletion are deferred.
// ---------------------------------------------------------------------------
void MixerOrchestrator::fade(const std::string& scene_name, double duration_sec) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    ensureIdle();

    state_->transition_mode = MixerState::TransitionMode::Crossfade;

    // Capture all needed values from pre-flip state
    bool pvw_is_slot_a = !state_->pgm_is_slot_a;
    uint32_t pvw_bit = state_->pvwOutputBit();
    const auto& target_slot = pvw_is_slot_a ? state_->slot_a : state_->slot_b;
    const auto& old_slot    = pvw_is_slot_a ? state_->slot_b : state_->slot_a;
    int pvw_sw_idx = state_->pvwSourceSwitcherIndex();

    // 1. Load target scene into PVW slot
    loadSceneIntoSlot(pvw_is_slot_a, scene_name);

    auto& target_scene = state_->scenes.at(scene_name);
    int frames = (int)std::round(duration_sec * state_->fps_num / state_->fps_den);

    // 2. Create transition_cuda with direction-dependent alpha expression
    std::string alpha_expr;
    if (pvw_is_slot_a) {
        alpha_expr = "clip(1-n/" + std::to_string(frames) + "\\,0\\,1)";
    } else {
        alpha_expr = "clip(n/" + std::to_string(frames) + "\\,0\\,1)";
    }

    Parameters trans_params;
    trans_params["type"] = "filter_video";
    trans_params["name"] = transition_node_name_;
    trans_params["src"] = Parameters::array({"scA_trans", "scB_trans"});
    trans_params["dst"] = transition_edge_name_;
    trans_params["graph"] = "transition_cuda=alpha='" + alpha_expr + "':eval=frame";
    trans_params["hwaccel"] = state_->hwaccel_name;
    trans_params["group"] = "mixer_trans";
    createAndStartNode(trans_params);

    // 3. Camera routing: applied in loadSceneIntoSlot via rewriteCameraOutputsForSlot

    // 4–5. Timeline: priming post-scene otms (direct+trans) then visible-path switches
    int64_t T_prep = wallclock.pts();
    timeline_->set(state_->slot_a.post_otm_name, "outputs", T_prep, Parameters(3u)); // 0b11 warmup
    timeline_->set(state_->slot_b.post_otm_name, "outputs", T_prep, Parameters(3u));

    int64_t T_start = wallclock.pts() + margin_ms_;
    int64_t T_end = T_start + (int64_t)(duration_sec * 1000);

    // At T_start: switch output to transition
    timeline_->set(state_->source_switcher_name, "active", T_start,
                   Parameters(MixerState::transSourceSwitcherIndex()));
    timeline_->set(state_->slot_a.post_otm_name, "outputs", T_start, Parameters(2u)); // 0b10 trans only
    timeline_->set(state_->slot_b.post_otm_name, "outputs", T_start, Parameters(2u));

    // At T_end: switch output to new PGM direct
    timeline_->set(state_->source_switcher_name, "active", T_end, Parameters(pvw_sw_idx));
    timeline_->set(target_slot.post_otm_name, "outputs", T_end, Parameters(1u));  // 0b01 direct only
    timeline_->set(old_slot.post_otm_name, "outputs", T_end, Parameters(0u));     // idle

    // Camera cleanup at T_cleanup: converge to new-PGM-only bitmasks
    // pvw_bit == post-flip PGM bit (the PVW slot becomes the new PGM)
    int64_t T_cleanup = T_end + 100;
    for (const auto& [src_name, info] : state_->sources) {
        uint32_t new_mask = target_scene.sources.count(src_name) ? pvw_bit : 0u;
        timeline_->set(info.otm_node_name, "outputs", T_cleanup, Parameters(new_mask));
    }
    timeline_->set(old_slot.compositor_name, "active_inputs", T_cleanup, Parameters(0u));

    // 6. Deferred cleanup: delete transition node + flip state
    int64_t flip_delay = (T_cleanup - wallclock.pts()) + 300;
    std::thread(deferredCleanup, nodes_, state_,
                flip_delay, pvw_is_slot_a, scene_name,
                std::vector<std::string>{transition_node_name_}).detach();
}

// ---------------------------------------------------------------------------
// wipeThread: runs on a detached thread.
// Phase 1 (midpoint): PVW slot prep + timeline source_switcher (hidden under opaque wipe).
// Phase 2 (end): timeline routing cleanup, tear down wipe chain, flip state.
// ---------------------------------------------------------------------------
void MixerOrchestrator::wipeThread(
        std::shared_ptr<NodeManager> nodes,
        std::shared_ptr<MixerState> state,
        std::shared_ptr<SharedTimeline> timeline,
        std::string scene_name,
        bool new_pgm_is_slot_a,
        int64_t midpoint_delay_ms,
        int64_t total_delay_ms) {

    // --- Phase 1: midpoint – do invisible scene switch under the fully-opaque wipe ---
    std::this_thread::sleep_for(std::chrono::milliseconds(midpoint_delay_ms));
    try {
        std::lock_guard<std::mutex> lock(state->mutex);
        MixerOrchestrator orch(nodes, state, timeline);

        // Reconfigure PVW slot for the target scene
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
        timeline->set(state->source_switcher_name, "active", Tw, Parameters(state->pvwSourceSwitcherIndex()));
    } catch (const std::exception& e) {
        logstream << "mixer: wipe midpoint error: " << e.what();
    }

    // --- Phase 2: wipe end – tear down and flip ---
    int64_t remaining = total_delay_ms - midpoint_delay_ms;
    if (remaining > 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(remaining + 500));

    try {
        std::lock_guard<std::mutex> lock(state->mutex);
        MixerOrchestrator orch(nodes, state, timeline);

        int64_t Tw = wallclock.pts();
        if (!state->wipe_otm_name.empty())
            timeline->set(state->wipe_otm_name, "outputs", Tw, Parameters(1u));
        if (!state->wipe_selector_name.empty())
            timeline->set(state->wipe_selector_name, "active", Tw, Parameters(0));

        uint32_t new_pgm_bit = state->pvwOutputBit();
        auto& scene = state->scenes.at(scene_name);
        for (const auto& [src_name, info] : state->sources) {
            uint32_t mask = scene.sources.count(src_name) ? new_pgm_bit : 0u;
            timeline->set(info.otm_node_name, "outputs", Tw, Parameters(mask));
        }

        const auto& old_slot = state->pgmSlot();
        timeline->set(old_slot.compositor_name, "active_inputs", Tw, Parameters(0u));

        // Stop the pre-created wipe subgraph; nodes are re-used on the next wipe
        if (!state->wipe_group_name.empty()) {
            orch.stopGroup(state->wipe_group_name);
            // Release any frames still sitting in wipe pipeline edges so they
            // don't replay at the start of the next wipe.
            orch.flushWipeEdges();
        }

        // Flip state
        state->pgm_is_slot_a = new_pgm_is_slot_a;
        state->pgm_scene_name = scene_name;
        state->pvw_scene_name = "";
        state->transition_mode = MixerState::TransitionMode::Idle;
    } catch (const std::exception& e) {
        logstream << "mixer: wipe cleanup error: " << e.what();
        state->transition_mode = MixerState::TransitionMode::Idle;
    }
}

// ---------------------------------------------------------------------------
// wipe: media wipe transition.  Uses the static wipe_otm + wipe_selector
// nodes to route through the overlay without edge rewiring.
// The wipe subgraph (group wipe_group_name) is pre-created but not running
// in steady state; it is started here and stopped at the end of the wipe.
// ---------------------------------------------------------------------------
void MixerOrchestrator::wipe(const std::string& scene_name, const std::string& wipe_file, double duration_sec) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    ensureIdle();

    if (state_->wipe_otm_name.empty() || state_->wipe_selector_name.empty())
        throw Error("mixer: wipe requires wipe_otm and wipe_selector nodes (see mixer.init)");
    if (state_->wipe_group_name.empty() || state_->wipe_input_node_name.empty())
        throw Error("mixer: wipe requires wipe_group and wipe_input_node (see mixer.init)");

    state_->transition_mode = MixerState::TransitionMode::Wipe;
    bool pvw_is_slot_a = !state_->pgm_is_slot_a;

    // Reset the input node so it is re-created with the new URL when the group starts.
    // On the first wipe the node_ was pre-created with an empty URL; stopping it (while
    // not running) destroys the pre-created node_ object so createNode() will pick up
    // the updated params on the next start().  On subsequent wipes node_ is already null
    // (destroyed when the group was stopped at the end of the previous wipe), so this
    // stop() call is a no-op.
    nodes_->node(state_->wipe_input_node_name)->stop(true);
    setNodeParam(state_->wipe_input_node_name, "url", wipe_file);
    // Discard any frames left over from a previous wipe run before starting fresh.
    flushWipeEdges();
    startGroup(state_->wipe_group_name);

    // Route output through wipe overlay (timeline so it tracks frame PTS like other mixer nodes)
    int64_t Tw = wallclock.pts();
    timeline_->set(state_->wipe_otm_name, "outputs", Tw, Parameters(3u));   // 0b11 both direct + wipe_in
    timeline_->set(state_->wipe_selector_name, "active", Tw, Parameters(1)); // wipe_overlay_out

    // Launch background thread for midpoint cut + cleanup
    int64_t total_ms = (int64_t)(duration_sec * 1000);
    int64_t midpoint_ms = total_ms / 2;
    std::thread(wipeThread, nodes_, state_, timeline_,
                scene_name, pvw_is_slot_a, midpoint_ms, total_ms).detach();
}

Parameters MixerOrchestrator::status() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    Parameters s;
    s["pgm_scene"] = state_->pgm_scene_name;
    s["pvw_scene"] = state_->pvw_scene_name;
    s["pgm_slot"] = state_->pgm_is_slot_a ? "A" : "B";
    auto mode = state_->transition_mode.load();
    switch (mode) {
        case MixerState::TransitionMode::Idle: s["transition"] = "idle"; break;
        case MixerState::TransitionMode::Cut: s["transition"] = "cut"; break;
        case MixerState::TransitionMode::Crossfade: s["transition"] = "crossfade"; break;
        case MixerState::TransitionMode::Wipe: s["transition"] = "wipe"; break;
    }
    return s;
}
