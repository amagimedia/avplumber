#include "mixer_orchestrator.hpp"
#include "avutils.hpp"
#include "graph_interfaces.hpp"
#include <algorithm>
#include <cmath>
#include <thread>
#include <chrono>

namespace {

constexpr int64_t kWipeSwitchGraceMs = 250;

void resetInputIf(std::shared_ptr<NodeManager> nodes, const std::string& name) {
    if (name.empty())
        return;
    auto w = nodes->node_if_exists(name);
    if (!w || !w->node())
        return;
    if (auto r = std::dynamic_pointer_cast<IInputReset>(w->node()))
        r->resetInput();
}

/// After a PGM path change, slot compositors may have been idle (`active_inputs=0`) while cameras
/// advanced in PTS; `force_fps` on `norm_*` would otherwise compare the next real frame to a stale
/// grid and produce a big `Discontinuity` jump with a duplicate burst across the cut.
///
/// Deliberately NOT resetting the final post-`out_sel` `force_fps` (e.g. `mixer_norm_fps`): it is
/// the last VFR guard before the encoder and resetting it would drop that guarantee.
void resetSlotNormFps(std::shared_ptr<NodeManager> nodes, const MixerState& st) {
    resetInputIf(nodes, st.slot_a.norm_ts_name);
    resetInputIf(nodes, st.slot_b.norm_ts_name);
}

bool nodeWorkingIfExists(std::shared_ptr<NodeManager> nodes, const std::string& name) {
    if (name.empty())
        return false;
    auto w = nodes->node_if_exists(name);
    return w && w->isWorking();
}

bool setNodeObjectIfCreated(std::shared_ptr<NodeManager> nodes,
                            const std::string& node_name,
                            const std::string& key,
                            const Parameters& value) {
    auto wrapper = nodes->node_if_exists(node_name);
    if (!wrapper)
        throw Error("Node " + node_name + " doesn't exist.");

    wrapper->parameters()[key] = value;
    if (!wrapper->node())
        return false;

    try {
        wrapper->setObject(key, value);
        return true;
    } catch (const std::exception& e) {
        if (std::string(e.what()) == "Node not created")
            return false;
        throw;
    }
}

int edgeOccupiedIfExists(std::shared_ptr<NodeManager> nodes, const std::string& name) {
    if (name.empty())
        return 0;
    auto e = nodes->edges()->findAny(name);
    return e ? e->occupied() : 0;
}

std::string firstDstEdgeName(std::shared_ptr<NodeManager> nodes, const std::string& node_name) {
    auto node = nodes->node_if_exists(node_name);
    if (!node)
        return "";
    const auto& params = node->parameters();
    if (!params.count("dst"))
        return "";
    auto names = jsonToStringList(params["dst"]);
    return names.empty() ? "" : names.front();
}


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

class TransitionPrepGuard {
    std::shared_ptr<MixerState> state_;
    bool active_ = true;

public:
    explicit TransitionPrepGuard(std::shared_ptr<MixerState> state)
        : state_(std::move(state)) {}

    ~TransitionPrepGuard() {
        if (active_)
            state_->transition_mode = MixerState::TransitionMode::Idle;
    }

    void release() {
        active_ = false;
    }
};

} // namespace

MixerOrchestrator::MixerOrchestrator(
    std::shared_ptr<NodeManager> nodes,
    std::shared_ptr<MixerState> state,
    std::shared_ptr<SharedTimeline> timeline)
    : nodes_(std::move(nodes)), state_(std::move(state)), timeline_(std::move(timeline)) {}

void MixerOrchestrator::setNodeObject(const std::string& node_name, const std::string& key, const Parameters& value) {
    try {
        auto node = nodes_->node(node_name);
        node->setObject(key, value);
    } catch (const std::exception& e) {
        throw Error("mixer: set " + node_name + "." + key + " failed: " + e.what());
    }
}

void MixerOrchestrator::publishCameraOtmOutputs(const std::string& otm_name, uint32_t mask) {
    // `one_to_many` with `timeline` uses tlGetRaw("outputs") whenever any entry matches; stale
    // rows (e.g. an old T_cleanup) would override setObject. We drop only the "outputs" key on
    // this OTM channel — not post-scene otms, not source_switcher, not other keys here.
    // cut/fade append new `outputs` rows at T_cleanup *after* loadSceneIntoSlot returns, so
    // those are not cleared by this call. Overlapping mixer commands are rejected by ensureIdle().
    timeline_->clearKey(otm_name, "outputs");
    if (!setNodeObjectIfCreated(nodes_, otm_name, "outputs", Parameters(mask))) {
        logstream << "mixer: queued " << otm_name << ".outputs=" << mask
                  << " for node not created yet";
    }
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

void MixerOrchestrator::flushSlotEdges(bool is_slot_a) {
    const auto& slot = is_slot_a ? state_->slot_a : state_->slot_b;

    auto clearEdge = [this](const std::string& name) {
        if (name.empty())
            return;
        auto edge = nodes_->edges()->findAny(name);
        if (edge && edge->occupied() > 0) {
            logstream << "mixer: flushing stale slot edge " << name
                      << " (" << edge->occupied() << " queued)";
            edge->clear();
        }
    };

    for (const auto& [_, info] : state_->sources) {
        const std::string& cs_node = is_slot_a ? info.cs_node_a : info.cs_node_b;
        auto node = nodes_->node_if_exists(cs_node);
        if (!node)
            continue;
        const auto& params = node->parameters();
        if (params.count("src")) {
            for (const auto& edge_name : jsonToStringList(params["src"]))
                clearEdge(edge_name);
        }
        if (params.count("dst")) {
            for (const auto& edge_name : jsonToStringList(params["dst"]))
                clearEdge(edge_name);
        }
    }

    for (const std::string& node_name : {slot.compositor_name, slot.norm_ts_name, slot.post_otm_name}) {
        auto node = nodes_->node_if_exists(node_name);
        if (!node)
            continue;
        const auto& params = node->parameters();
        if (params.count("dst")) {
            for (const auto& edge_name : jsonToStringList(params["dst"]))
                clearEdge(edge_name);
        }
    }
}

void MixerOrchestrator::ensureIdle() const {
    auto mode = state_->transition_mode.load();
    if (mode != MixerState::TransitionMode::Idle)
        throw Error("mixer: transition already in progress");
}

int64_t MixerOrchestrator::resolveTransitionStartPts(int64_t requested_start_pts_ms) const {
    int64_t now = wallclock.pts();
    if (requested_start_pts_ms < 0)
        return now;
    int64_t earliest = now + margin_ms_;
    if (requested_start_pts_ms < earliest)
        throw Error("mixer: start_pts_ms must be at least " + std::to_string(margin_ms_) +
                    "ms in the future");
    return requested_start_pts_ms;
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

    // The slot being loaded is the broadcast-inactive PVW slot.  Its compositor
    // was previously idled with active_inputs=0, so it may still hold frames on
    // its input edges. If left there, the next activation starts by rendering
    // stale frames and appears to lag behind the scene switch.
    flushSlotEdges(is_slot_a);

    for (const auto& [src_name, layout] : scene.sources) {
        auto src_it = state_->sources.find(src_name);
        if (src_it == state_->sources.end()) continue;
        const auto& info = src_it->second;
        const std::string& cs_node = is_slot_a ? info.cs_node_a : info.cs_node_b;

        // Only restart the crop/scale node when the graph string actually changed.
        // Restarting a filter_video node tears down and rebuilds its FFmpeg filter
        // graph, which briefly stops producing frames and allocates a new
        // hw_frames_ctx pool.  Downstream filter_video nodes now absorb pool
        // rotations via a semantic hw_frames_ctx comparison so this no longer
        // causes a mid-wipe EXT_NULL gap, but the restart is still a wasted
        // stall and a frame-timing hiccup when the graph string is unchanged.
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

    // Drop slot bit for every camera, then enable only sources in scene with active_inputs set.
    // Keeps `outputs` consistent with compositor consumption (no frames into unused inputs).
    const uint32_t slot_bit = is_slot_a ? 1u : 2u;
    rewriteCameraOutputsForSlot(slot_bit, scene);

    state_->pvw_scene_name = scene_name;
}

void MixerOrchestrator::scheduleSceneControls(const SceneDefinition& scene, int64_t at_pts_ms) {
    for (const auto& control : scene.controls) {
        timeline_->set(control.node_name, control.key, at_pts_ms, control.value);
        logstream << "mixer scene control: " << control.node_name << "." << control.key
                  << " at " << at_pts_ms << " -> " << control.value;
    }
}

void MixerOrchestrator::preview(const std::string& scene_name) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    ensureIdle();
    if (!state_->scenes.count(scene_name))
        throw Error("mixer.preview: unknown scene: " + scene_name);

    bool pvw_is_slot_a = !state_->pgm_is_slot_a;
    const auto& slot = state_->pvwSlot();

    if (state_->pvw_scene_name == scene_name) {
        logstream << "mixer preview: scene already loaded in PVW: " << scene_name;
    } else {
        loadSceneIntoSlot(pvw_is_slot_a, scene_name);
        resetInputIf(nodes_, slot.norm_ts_name);
    }

    int64_t T_prep = wallclock.pts();
    timeline_->clearKey(slot.post_otm_name, "outputs");
    setNodeObject(slot.post_otm_name, "outputs", Parameters(1u));
    timeline_->set(slot.post_otm_name, "outputs", T_prep, Parameters(1u));
    logstream << "mixer preview armed: scene=" << scene_name
              << " slot=" << (pvw_is_slot_a ? 'A' : 'B')
              << " post_otm " << slot.post_otm_name << "->1";
}

// ---------------------------------------------------------------------------
// cutInternal: the graph-level work for a hard cut, without touching
// transition_mode or pgm_is_slot_a.  All values are read from pre-flip state.
// Caller must hold state_->mutex.
// Returns the earliest cut PTS (wallclock ms). Cold cuts are gated until the
// incoming direct edge has produced a fresh frame; preloaded PVW cuts only wait
// for the scheduled PTS.
// ---------------------------------------------------------------------------
int64_t MixerOrchestrator::cutInternal(const std::string& scene_name, int64_t start_pts_ms) {
    bool pvw_is_slot_a = !state_->pgm_is_slot_a;

    if (state_->pvw_scene_name == scene_name) {
        logstream << "mixer cut: reusing preloaded PVW scene=" << scene_name;
    } else {
        loadSceneIntoSlot(pvw_is_slot_a, scene_name);
    }

    int64_t T_prep = wallclock.pts();
    int64_t T_cut = start_pts_ms;
    const auto& new_slot = state_->pvwSlot();

    // Pre-warm the hidden direct branch before the visible `out_sel` switch. Without this,
    // enabling `post_otm` and switching `out_sel` at the same PTS leaves the newly selected
    // path one pipeline-latency late, so the final encoder-side force_fps repeats the last
    // visible frame for a few ticks across the cut. `one_to_many` with timeline runs in
    // drop_dynamic_ mode, so feeding an inactive direct branch here is safe: `source_switcher`
    // drains and drops those pre-roll frames instead of back-pressuring the slot.
    timeline_->clearKey(new_slot.post_otm_name, "outputs");
    setNodeObject(new_slot.post_otm_name, "outputs", Parameters(1u));
    timeline_->set(new_slot.post_otm_name, "outputs", T_prep, Parameters(1u));

    logstream << "mixer cut armed: scene=" << scene_name << " earliest T_cut=" << T_cut
              << " post_otm prep " << new_slot.post_otm_name << "->1";

    resetSlotNormFps(nodes_, *state_);

    return T_cut;
}

// ---------------------------------------------------------------------------
// deferredCleanup: runs on a detached thread.  Sleeps, then flips internal
// bookkeeping and deletes nodes that can't be removed via timeline.
// ---------------------------------------------------------------------------
void MixerOrchestrator::deferredCleanup(
        std::shared_ptr<NodeManager> nodes,
        std::shared_ptr<MixerState> state,
        std::shared_ptr<SharedTimeline> timeline,
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
    try {
        const uint32_t pgm_bit = new_pgm_is_slot_a ? 1u : 2u;
        const auto scene_it = state->scenes.find(new_pgm_scene);
        if (scene_it != state->scenes.end()) {
            const SceneDefinition& scene = scene_it->second;
            const uint32_t active = state->computeActiveInputsMask(scene);
            for (const auto& [src_name, info] : state->sources) {
                const bool in_scene = scene.sources.count(src_name) > 0;
                const bool active_input = (active & (1u << (unsigned)info.input_index)) != 0;
                const uint32_t mask = (in_scene && active_input) ? pgm_bit : 0u;
                timeline->clearKey(info.otm_node_name, "outputs");
                setNodeObjectIfCreated(nodes, info.otm_node_name, "outputs", Parameters(mask));
            }
            const auto& new_slot = new_pgm_is_slot_a ? state->slot_a : state->slot_b;
            const auto& old_slot = new_pgm_is_slot_a ? state->slot_b : state->slot_a;
            timeline->clearKey(new_slot.post_otm_name, "outputs");
            timeline->clearKey(old_slot.post_otm_name, "outputs");
            timeline->clearKey(new_slot.compositor_name, "active_inputs");
            timeline->clearKey(old_slot.compositor_name, "active_inputs");
            timeline->clearKey(state->source_switcher_name, "active");
            nodes->node(new_slot.post_otm_name)->setObject("outputs", Parameters(1u));
            nodes->node(old_slot.post_otm_name)->setObject("outputs", Parameters(0u));
            nodes->node(new_slot.compositor_name)->setObject("active_inputs", Parameters(active));
            nodes->node(old_slot.compositor_name)->setObject("active_inputs", Parameters(0u));
            nodes->node(state->source_switcher_name)->setObject(
                "active", Parameters(new_pgm_is_slot_a ? 0 : 1));
        }
    } catch (const std::exception& e) {
        logstream << "mixer: deferred cleanup error restoring routing: " << e.what();
    }
    state->pgm_is_slot_a = new_pgm_is_slot_a;
    state->pgm_scene_name = std::move(new_pgm_scene);
    state->pvw_scene_name = "";
    state->transition_mode = MixerState::TransitionMode::Idle;
}

void MixerOrchestrator::readyCutThread(
        std::shared_ptr<NodeManager> nodes,
        std::shared_ptr<MixerState> state,
        std::shared_ptr<SharedTimeline> timeline,
        bool new_pgm_is_slot_a,
        std::string new_pgm_scene,
        std::string ready_edge_name,
        av::Timestamp ready_edge_initial_ts,
        int64_t earliest_switch_pts_ms,
        bool require_new_ready_frame) {
    constexpr int64_t kPollMs = 5;
    constexpr int64_t kMaxWaitMs = 1500;
    int64_t waited_ms = 0;
    while (waited_ms < kMaxWaitMs) {
        const bool time_ready = wallclock.pts() >= earliest_switch_pts_ms;
        bool edge_ready = !require_new_ready_frame;
        if (require_new_ready_frame) {
            auto edge = nodes->edges()->findAny(ready_edge_name);
            if (!edge)
                break;
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
    try {
        const uint32_t pgm_bit = new_pgm_is_slot_a ? 1u : 2u;
        const auto scene_it = state->scenes.find(new_pgm_scene);
        if (scene_it != state->scenes.end()) {
            const SceneDefinition& scene = scene_it->second;
            const uint32_t active = state->computeActiveInputsMask(scene);
            const auto& new_slot = new_pgm_is_slot_a ? state->slot_a : state->slot_b;
            const auto& old_slot = new_pgm_is_slot_a ? state->slot_b : state->slot_a;

            timeline->clearKey(state->source_switcher_name, "active");
            nodes->node(state->source_switcher_name)->setObject(
                "active", Parameters(new_pgm_is_slot_a ? 0 : 1));

            for (const auto& [src_name, info] : state->sources) {
                const bool in_scene = scene.sources.count(src_name) > 0;
                const bool active_input = (active & (1u << (unsigned)info.input_index)) != 0;
                const uint32_t mask = (in_scene && active_input) ? pgm_bit : 0u;
                timeline->clearKey(info.otm_node_name, "outputs");
                setNodeObjectIfCreated(nodes, info.otm_node_name, "outputs", Parameters(mask));
            }
            timeline->clearKey(new_slot.post_otm_name, "outputs");
            timeline->clearKey(old_slot.post_otm_name, "outputs");
            timeline->clearKey(new_slot.compositor_name, "active_inputs");
            timeline->clearKey(old_slot.compositor_name, "active_inputs");
            nodes->node(new_slot.post_otm_name)->setObject("outputs", Parameters(1u));
            nodes->node(old_slot.post_otm_name)->setObject("outputs", Parameters(0u));
            nodes->node(new_slot.compositor_name)->setObject("active_inputs", Parameters(active));
            nodes->node(old_slot.compositor_name)->setObject("active_inputs", Parameters(0u));
        }
    } catch (const std::exception& e) {
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
void MixerOrchestrator::cut(const std::string& scene_name, int64_t start_pts_ms) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    ensureIdle();

    int64_t T_cut = resolveTransitionStartPts(start_pts_ms);
    state_->transition_mode = MixerState::TransitionMode::Cut;
    TransitionPrepGuard prep_guard(state_);
    bool pvw_is_slot_a = !state_->pgm_is_slot_a;
    bool was_preloaded = state_->pvw_scene_name == scene_name;

    scheduleSceneControls(state_->scenes.at(scene_name), T_cut);
    cutInternal(scene_name, T_cut);

    const auto& new_slot = state_->pvwSlot();
    std::string ready_edge_name = firstDstEdgeName(nodes_, new_slot.post_otm_name);
    auto ready_edge = nodes_->edges()->findAny(ready_edge_name);
    av::Timestamp ready_edge_initial_ts = ready_edge ? ready_edge->lastTS() : NOTS;
    std::thread(readyCutThread, nodes_, state_, timeline_,
                pvw_is_slot_a, scene_name, ready_edge_name, ready_edge_initial_ts,
                T_cut, !was_preloaded).detach();
    prep_guard.release();
}

// ---------------------------------------------------------------------------
// fade: crossfade transition.  All timeline values are computed from the
// pre-flip state.  The state flip + transition node deletion are deferred.
// ---------------------------------------------------------------------------
void MixerOrchestrator::fade(const std::string& scene_name, double duration_sec, int64_t start_pts_ms) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    ensureIdle();

    int64_t T_start = resolveTransitionStartPts(start_pts_ms);
    state_->transition_mode = MixerState::TransitionMode::Crossfade;
    TransitionPrepGuard prep_guard(state_);

    // Capture all needed values from pre-flip state
    bool pvw_is_slot_a = !state_->pgm_is_slot_a;
    uint32_t pvw_bit = state_->pvwOutputBit();
    const auto& target_slot = pvw_is_slot_a ? state_->slot_a : state_->slot_b;
    const auto& old_slot    = pvw_is_slot_a ? state_->slot_b : state_->slot_a;
    int pvw_sw_idx = state_->pvwSourceSwitcherIndex();

    // 1. Ensure target scene is loaded into PVW slot.
    if (state_->pvw_scene_name == scene_name) {
        logstream << "mixer fade: reusing preloaded PVW scene=" << scene_name;
    } else {
        loadSceneIntoSlot(pvw_is_slot_a, scene_name);
    }

    auto& target_scene = state_->scenes.at(scene_name);
    int frames = (int)std::round(duration_sec * state_->fps_num / state_->fps_den);
    scheduleSceneControls(target_scene, T_start);

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
    int64_t T_prep = T_start - margin_ms_;
    timeline_->set(state_->slot_a.post_otm_name, "outputs", T_prep, Parameters(3u)); // 0b11 warmup
    timeline_->set(state_->slot_b.post_otm_name, "outputs", T_prep, Parameters(3u));

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
    std::thread(deferredCleanup, nodes_, state_, timeline_,
                flip_delay, pvw_is_slot_a, scene_name,
                std::vector<std::string>{transition_node_name_}).detach();
    prep_guard.release();
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
    // Stage 2a: wait for the wipe source to EOF (or until the planned duration elapses).
    int64_t remaining = total_delay_ms - midpoint_delay_ms;
    bool hit_input_eof = false;
    if (remaining > 0) {
        int64_t waited_ms = 0;
        constexpr int64_t kPollMs = 10;
        while (waited_ms < remaining) {
            if (!nodeWorkingIfExists(nodes, state->wipe_input_node_name)) {
                logstream << "mixer wipe: cleanup pulled to wipe EOF after " << waited_ms
                          << "ms of remaining tail";
                hit_input_eof = true;
                break;
            }
            int64_t step_ms = std::min<int64_t>(kPollMs, remaining - waited_ms);
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
        constexpr int64_t kWipeOverlayTailMs = 120; // ~3-4 frames @30fps
        int64_t waited = 0;
        while (waited < kWipeDrainTimeoutMs) {
            if (edgeOccupiedIfExists(nodes, state->wipe_tail_edge) == 0)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(kWipeDrainPollMs));
            waited += kWipeDrainPollMs;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kWipeOverlayTailMs));
        logstream << "mixer wipe: drained " << state->wipe_tail_edge << " in " << waited
                  << "ms + " << kWipeOverlayTailMs << "ms overlay tail";
    }

    try {
        std::lock_guard<std::mutex> lock(state->mutex);
        MixerOrchestrator orch(nodes, state, timeline);

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
            uint32_t mask = scene.sources.count(src_name) ? new_pgm_bit : 0u;
            timeline->set(info.otm_node_name, "outputs", Tw, Parameters(mask));
        }

        const auto& old_slot = state->pgmSlot();
        timeline->set(old_slot.compositor_name, "active_inputs", Tw, Parameters(0u));
    } catch (const std::exception& e) {
        logstream << "mixer: wipe cleanup error: " << e.what();
        state->transition_mode = MixerState::TransitionMode::Idle;
        return;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(kWipeSwitchGraceMs));

    try {
        std::lock_guard<std::mutex> lock(state->mutex);
        MixerOrchestrator orch(nodes, state, timeline);

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

        state->pgm_is_slot_a = new_pgm_is_slot_a;
        state->pgm_scene_name = scene_name;
        state->pvw_scene_name = "";
        state->transition_mode = MixerState::TransitionMode::Idle;
    } catch (const std::exception& e) {
        logstream << "mixer: wipe teardown error: " << e.what();
        state->transition_mode = MixerState::TransitionMode::Idle;
    }
}

// ---------------------------------------------------------------------------
// wipe: media wipe transition.  Uses the static wipe_otm + wipe_selector
// nodes to route through the overlay without edge rewiring.
// The wipe subgraph (group wipe_group_name) is pre-created but not running
// in steady state; it is started here and stopped at the end of the wipe.
// ---------------------------------------------------------------------------
void MixerOrchestrator::wipePrepAndRun(
        std::shared_ptr<NodeManager> nodes,
        std::shared_ptr<MixerState> state,
        std::shared_ptr<SharedTimeline> timeline,
        std::string scene_name,
        std::string wipe_file,
        double duration_sec,
        bool new_pgm_is_slot_a,
        int64_t start_pts_ms,
        int64_t margin_ms) {
    int64_t T_prep = start_pts_ms - margin_ms;
    int64_t prep_delay = T_prep - wallclock.pts();
    if (prep_delay > 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(prep_delay));

    try {
        std::lock_guard<std::mutex> lock(state->mutex);
        MixerOrchestrator orch(nodes, state, timeline);

        nodes->node(state->wipe_input_node_name)->stop(true);
        orch.setNodeParam(state->wipe_input_node_name, "url", wipe_file);
        orch.flushWipeEdges();
        resetInputIf(nodes, state->wipe_base_fps_name);
        orch.startGroup(state->wipe_group_name);

        timeline->set(state->wipe_otm_name, "outputs", T_prep, Parameters(3u));       // 0b11 both direct + wipe_in
        timeline->set(state->wipe_selector_name, "active", start_pts_ms, Parameters(1)); // wipe_overlay_out

        int64_t total_ms = (int64_t)(duration_sec * 1000);
        int64_t midpoint_ms = total_ms / 2;
        logstream << "mixer wipe: scene=" << scene_name << " file=" << wipe_file << " T_prep=" << T_prep
                  << " T_start=" << start_pts_ms << " total_ms=" << total_ms << " midpoint_ms=" << midpoint_ms
                  << " new_pgm_slot_" << (new_pgm_is_slot_a ? 'A' : 'B');
    } catch (const std::exception& e) {
        logstream << "mixer: wipe prep error: " << e.what();
        std::lock_guard<std::mutex> lock(state->mutex);
        state->transition_mode = MixerState::TransitionMode::Idle;
        return;
    }

    int64_t total_ms = (int64_t)(duration_sec * 1000);
    int64_t midpoint_ms = total_ms / 2;
    wipeThread(nodes, state, timeline, std::move(scene_name), new_pgm_is_slot_a,
               midpoint_ms + margin_ms, total_ms + margin_ms);
}

void MixerOrchestrator::wipe(const std::string& scene_name, const std::string& wipe_file, double duration_sec,
                             int64_t start_pts_ms) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    ensureIdle();

    if (state_->wipe_otm_name.empty() || state_->wipe_selector_name.empty())
        throw Error("mixer: wipe requires wipe_otm and wipe_selector nodes (see mixer.init)");
    if (state_->wipe_group_name.empty() || state_->wipe_input_node_name.empty())
        throw Error("mixer: wipe requires wipe_group and wipe_input_node (see mixer.init)");

    int64_t T_start = resolveTransitionStartPts(start_pts_ms);
    state_->transition_mode = MixerState::TransitionMode::Wipe;
    bool pvw_is_slot_a = !state_->pgm_is_slot_a;
    scheduleSceneControls(state_->scenes.at(scene_name), T_start);

    if (start_pts_ms < 0) {
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
        resetInputIf(nodes_, state_->wipe_base_fps_name);
        startGroup(state_->wipe_group_name);

        // Pre-roll the wipe branch before making it visible. Switching `wipe_sel`
        // immediately after `startGroup()` can expose the startup latency of the
        // wipe decode/filter chain as a freeze on the last direct frame.
        int64_t T_prep = T_start - margin_ms_;
        timeline_->set(state_->wipe_otm_name, "outputs", T_prep, Parameters(3u));    // 0b11 both direct + wipe_in
        timeline_->set(state_->wipe_selector_name, "active", T_start, Parameters(1)); // wipe_overlay_out

        // Launch background thread for midpoint cut + cleanup
        int64_t total_ms = (int64_t)(duration_sec * 1000);
        int64_t midpoint_ms = total_ms / 2;
        logstream << "mixer wipe: scene=" << scene_name << " file=" << wipe_file << " T_prep=" << T_prep
                  << " T_start=" << T_start << " total_ms=" << total_ms << " midpoint_ms=" << midpoint_ms
                  << " new_pgm_slot_" << (pvw_is_slot_a ? 'A' : 'B');
        std::thread(wipeThread, nodes_, state_, timeline_,
                    scene_name, pvw_is_slot_a, midpoint_ms + margin_ms_, total_ms + margin_ms_).detach();
        return;
    }

    std::thread(wipePrepAndRun, nodes_, state_, timeline_,
                scene_name, wipe_file, duration_sec, pvw_is_slot_a, T_start, margin_ms_).detach();
}

std::vector<std::string> MixerOrchestrator::sceneNames() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    std::vector<std::string> names;
    names.reserve(state_->scenes.size());
    for (const auto& [name, _] : state_->scenes)
        names.push_back(name);
    std::sort(names.begin(), names.end());
    return names;
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
