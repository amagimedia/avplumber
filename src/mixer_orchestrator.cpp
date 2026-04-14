#include "mixer_orchestrator.hpp"
#include "avutils.hpp"
#include <cmath>
#include <thread>
#include <chrono>

MixerOrchestrator::MixerOrchestrator(
    std::shared_ptr<NodeManager> nodes,
    std::shared_ptr<MixerState> state,
    std::shared_ptr<SharedTimeline> timeline)
    : nodes_(std::move(nodes)), state_(std::move(state)), timeline_(std::move(timeline)) {}

void MixerOrchestrator::setNodeObject(const std::string& node_name, const std::string& key, const Parameters& value) {
    auto node = nodes_->node(node_name);
    node->setObject(key, value);
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

void MixerOrchestrator::setCameraOutputBits(const SceneDefinition& scene, uint32_t slot_bit, bool set) {
    for (const auto& [src_name, layout] : scene.sources) {
        auto it = state_->sources.find(src_name);
        if (it == state_->sources.end()) continue;
        const auto& info = it->second;
        Parameters current_val;
        bool got = nodes_->node(info.otm_node_name)->getObjectTry("outputs", current_val);
        uint32_t mask = got ? current_val.get<uint32_t>() : 0u;
        if (set)
            mask |= slot_bit;
        else
            mask &= ~slot_bit;
        setNodeObject(info.otm_node_name, "outputs", Parameters(mask));
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
        setNodeParam(cs_node, "graph", layout.crop_scale_graph);
        autoRestartNode(cs_node);
    }

    setNodeObject(slot.compositor_name, "layers", scene.layers);

    uint32_t active_mask = state_->computeActiveInputsMask(scene);
    setNodeObject(slot.compositor_name, "active_inputs", Parameters(active_mask));

    state_->pvw_scene_name = scene_name;
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
    setCameraOutputBits(scene, pvw_bit, true);

    int64_t T_cut = wallclock.pts() + margin_ms_;
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
        const auto& old_slot = state_->pgmSlot();
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

    // 3. Enable camera outputs for PVW slot (shared cameras get both bits)
    setCameraOutputBits(target_scene, pvw_bit, true);

    // 4. Enable post-scene otm to split mode (direct + trans)
    setNodeObject(state_->slot_a.post_otm_name, "outputs", Parameters(3u)); // 0b11
    setNodeObject(state_->slot_b.post_otm_name, "outputs", Parameters(3u)); // 0b11

    // 5. Timeline writes (all values from pre-flip state)
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
// Phase 1 (midpoint): immediate scene switch hidden behind the opaque wipe.
// Phase 2 (end): tear down wipe chain, flip state.
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

        // Enable cameras for the PVW slot
        auto& scene = state->scenes.at(scene_name);
        uint32_t pvw_bit = state->pvwOutputBit();
        orch.setCameraOutputBits(scene, pvw_bit, true);

        // Immediately switch source_switcher (invisible behind wipe overlay)
        orch.setNodeObject(state->source_switcher_name, "active",
                           Parameters(state->pvwSourceSwitcherIndex()));
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

        // Switch wipe output path back to direct
        if (!state->wipe_otm_name.empty())
            orch.setNodeObject(state->wipe_otm_name, "outputs", Parameters(1u));
        if (!state->wipe_selector_name.empty())
            orch.setNodeObject(state->wipe_selector_name, "active", Parameters(0));

        // Converge camera bitmasks to new PGM only
        // (pvwOutputBit() still returns pre-flip PVW bit = post-flip PGM bit)
        uint32_t new_pgm_bit = state->pvwOutputBit();
        auto& scene = state->scenes.at(scene_name);
        for (const auto& [src_name, info] : state->sources) {
            uint32_t mask = scene.sources.count(src_name) ? new_pgm_bit : 0u;
            orch.setNodeObject(info.otm_node_name, "outputs", Parameters(mask));
        }

        // Disable old compositor
        const auto& old_slot = state->pgmSlot();
        orch.setNodeObject(old_slot.compositor_name, "active_inputs", Parameters(0u));

        // Delete wipe chain
        for (const auto& name : {"wipe_overlay", "wipe_rt", "wipe_fmt", "wipe_dec", "wipe_demux", "wipe_input"})
            orch.deleteNodeIfExists(name);

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
// ---------------------------------------------------------------------------
void MixerOrchestrator::wipe(const std::string& scene_name, const std::string& wipe_file, double duration_sec) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    ensureIdle();

    if (state_->wipe_otm_name.empty() || state_->wipe_selector_name.empty())
        throw Error("mixer: wipe requires wipe_otm and wipe_selector nodes (see mixer.init)");

    state_->transition_mode = MixerState::TransitionMode::Wipe;
    bool pvw_is_slot_a = !state_->pgm_is_slot_a;

    // Phase 1: create wipe playback chain
    Parameters wipe_input;
    wipe_input["type"] = "input_rec";
    wipe_input["name"] = "wipe_input";
    wipe_input["url"] = wipe_file;
    wipe_input["loop"] = false;
    wipe_input["dst"] = "wipe_raw_pkt";
    wipe_input["group"] = "mixer_wipe";
    createAndStartNode(wipe_input);

    Parameters wipe_demux;
    wipe_demux["type"] = "demux";
    wipe_demux["name"] = "wipe_demux";
    wipe_demux["src"] = "wipe_raw_pkt";
    wipe_demux["routing"] = Parameters({{"v:0", "wipe_v_pkt"}});
    wipe_demux["group"] = "mixer_wipe";
    createAndStartNode(wipe_demux);

    Parameters wipe_dec;
    wipe_dec["type"] = "dec_video";
    wipe_dec["name"] = "wipe_dec";
    wipe_dec["src"] = "wipe_v_pkt";
    wipe_dec["dst"] = "wipe_dec_out";
    wipe_dec["pixel_format"] = "?cuda";
    wipe_dec["hwaccel"] = state_->hwaccel_name;
    wipe_dec["group"] = "mixer_wipe";
    createAndStartNode(wipe_dec);

    Parameters wipe_fmt;
    wipe_fmt["type"] = "filter_video";
    wipe_fmt["name"] = "wipe_fmt";
    wipe_fmt["src"] = "wipe_dec_out";
    wipe_fmt["dst"] = "wipe_fmt_out";
    wipe_fmt["graph"] = "format=yuva420p,hwupload_cuda";
    wipe_fmt["hwaccel"] = state_->hwaccel_name;
    wipe_fmt["group"] = "mixer_wipe";
    createAndStartNode(wipe_fmt);

    Parameters wipe_rt;
    wipe_rt["type"] = "realtime";
    wipe_rt["name"] = "wipe_rt";
    wipe_rt["src"] = "wipe_fmt_out";
    wipe_rt["dst"] = "wipe_rt_out";
    wipe_rt["set_pts"] = true;
    wipe_rt["group"] = "mixer_wipe";
    createAndStartNode(wipe_rt);

    // Phase 2: create overlay node writing to the wipe_selector's second input
    Parameters wipe_overlay;
    wipe_overlay["type"] = "filter_video";
    wipe_overlay["name"] = "wipe_overlay";
    wipe_overlay["src"] = Parameters::array({"final_wipe_in", "wipe_rt_out"});
    wipe_overlay["dst"] = "wipe_overlay_out";
    wipe_overlay["graph"] = "overlay_many_cuda";
    wipe_overlay["hwaccel"] = state_->hwaccel_name;
    wipe_overlay["group"] = "mixer_wipe";
    createAndStartNode(wipe_overlay);

    // Phase 3: route output through wipe overlay
    setNodeObject(state_->wipe_otm_name, "outputs", Parameters(3u));  // 0b11 both direct + wipe_in
    setNodeObject(state_->wipe_selector_name, "active", Parameters(1)); // wipe_overlay_out

    // Phase 4: launch background thread for midpoint cut + cleanup
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
