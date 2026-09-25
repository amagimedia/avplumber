// Sources, scenes and slot loading: what a scene means in node parameters and
// how the PVW slot is prepared before a transition.
#include "internal.hpp"

namespace avp::mixer {

void MixerOrchestrator::defineSource(const std::string& name, const std::string& otm_node, int input_index,
                                      const std::string& cs_node_a, const std::string& cs_node_b) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (input_index < 0 || input_index >= SourceMask::kBits)
        throw Error("mixer source input_index must be in 0.." + std::to_string(SourceMask::kBits - 1));
    MixerState::SourceInfo info;
    info.otm_node_name = otm_node;
    info.input_index = input_index;
    info.cs_node_a = cs_node_a;
    info.cs_node_b = cs_node_b;
    state_->sources[name] = std::move(info);
}

void MixerOrchestrator::defineRoutedSource(const std::string& name, const std::string& router_node,
                                           int input_index,
                                           const std::string& route_output_label_a,
                                           const std::string& route_output_label_b,
                                           const std::string& cs_node_a, const std::string& cs_node_b) {
    const int output_count = routerOutputCount(nodes_, router_node);
    const int route_output_a = routerOutputIndexFromLabel(nodes_, router_node, route_output_label_a);
    const int route_output_b = routerOutputIndexFromLabel(nodes_, router_node, route_output_label_b);

    std::lock_guard<std::mutex> lock(state_->mutex);
    if (input_index < 0 || input_index >= SourceMask::kBits)
        throw Error("mixer source input_index must be in 0.." + std::to_string(SourceMask::kBits - 1));
    MixerState::SourceInfo info;
    info.input_index = input_index;
    info.cs_node_a = cs_node_a;
    info.cs_node_b = cs_node_b;
    info.routed = true;
    info.router_node_name = router_node;
    info.route_output_label_a = route_output_label_a;
    info.route_output_label_b = route_output_label_b;
    info.route_output_a = route_output_a;
    info.route_output_b = route_output_b;
    state_->sources[name] = std::move(info);

    int& stored_output_count = state_->router_output_counts[router_node];
    if (stored_output_count != 0 && stored_output_count != output_count) {
        throw Error("mixer.routed_source: router " + router_node + " output count changed from " +
                    std::to_string(stored_output_count) + " to " +
                    std::to_string(output_count));
    }
    stored_output_count = output_count;
    ensureRouteTableSize(*state_, router_node);
}

void MixerOrchestrator::defineScene(const std::string& name, const SceneDefinition& def) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (state_->scene_definitions_frozen)
        throw Error("mixer.scene: aux-enabled setup has fixed scene definitions; reload setup to edit");
    if (state_->prewarm_cut_scenes.count(name) &&
            (!canPrewarmScene(def) || (state_->computeActiveInputsMask(def) & ~state_->prewarm_source_mask).any()))
        state_->prewarm_cut_scenes.erase(name); // Edited source identity takes the ordinary cold path.
    state_->scenes[name] = def;
}

bool MixerOrchestrator::canPrewarmScene(const SceneDefinition& scene) const {
    if (!scene.routes.empty() || !scene.controls.empty()) return false;
    for (const auto& [name, layout] : scene.sources) {
        const auto source = state_->sources.find(name);
        if (source == state_->sources.end() || source->second.routed ||
                !source->second.cs_node_a.empty() || !source->second.cs_node_b.empty() ||
                !layout.crop_scale_graph.empty()) return false;
    }
    return !scene.sources.empty();
}

void MixerOrchestrator::prewarmCuts(const std::vector<std::string>& scenes) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    ensureIdle();
    SourceMask mask;
    for (const auto& name : scenes) {
        const auto scene = state_->scenes.find(name);
        if (scene == state_->scenes.end() || !canPrewarmScene(scene->second))
            throw Error("mixer.prewarm: scenes require fixed, filter-free sources without routes or controls: " + name);
        mask |= state_->computeActiveInputsMask(scene->second);
    }
    for (const auto& slot : {state_->slot_a, state_->slot_b}) {
        const auto node = nodes_->node(slot.compositor_name);
        if (!node->node() || !node->parameters().contains("fps"))
            throw Error("mixer.prewarm: requires created clocked compositors");
    }
    for (const auto& slot : {state_->slot_a, state_->slot_b})
        setNodeObject(slot.compositor_name, "prewarm_inputs", toParameters(mask));
    state_->prewarm_cut_scenes = {scenes.begin(), scenes.end()};
    state_->prewarm_source_mask = mask;
    for (const auto& [name, source] : state_->sources) {
        if (source.routed) continue;
        uint32_t outputs = state_->scenes.at(state_->pgm_scene_name).sources.count(name) ? state_->pgmOutputBit() : 0u;
        if (!state_->pvw_scene_name.empty() && state_->scenes.at(state_->pvw_scene_name).sources.count(name))
            outputs |= state_->pvwOutputBit();
        publishCameraOtmOutputs(source.otm_node_name, state_->sourceOutputMask(source, outputs));
    }
}

void MixerOrchestrator::applyRoutedSceneRoutesForSlot(bool is_slot_a, const SceneDefinition& scene,
                                                       int64_t at_pts_ms, bool immediate) {
    auto tables = currentRouterTables(*state_);
    setRoutedSlotInTables(*state_, tables, is_slot_a, &scene);

    for (const auto& [router_name, routes] : tables) {
        Parameters value = routesToParameters(routes);
        if (immediate) {
            timeline_->clearKey(router_name, "routes");
            if (!setNodeObjectIfCreated(nodes_, router_name, "routes", value)) {
                logstream << "mixer: queued " << router_name
                          << ".routes for router node not created yet";
            }
            state_->router_routes[router_name] = routes;
        }
        timeline_->set(router_name, "routes", at_pts_ms, value);
    }
}

void MixerOrchestrator::publishRoutedRoutesForProgramOnly(bool pgm_is_slot_a,
                                                           const SceneDefinition& scene,
                                                           int64_t at_pts_ms,
                                                           bool immediate) {
    auto tables = currentRouterTables(*state_);
    setRoutedSlotInTables(*state_, tables, true, nullptr);
    setRoutedSlotInTables(*state_, tables, false, nullptr);
    setRoutedSlotInTables(*state_, tables, pgm_is_slot_a, &scene);

    for (const auto& [router_name, routes] : tables) {
        Parameters value = routesToParameters(routes);
        if (immediate) {
            timeline_->clearKey(router_name, "routes");
            if (!setNodeObjectIfCreated(nodes_, router_name, "routes", value)) {
                logstream << "mixer: queued " << router_name
                          << ".routes for router node not created yet";
            }
            state_->router_routes[router_name] = routes;
        }
        timeline_->set(router_name, "routes", at_pts_ms, value);
    }
}

void MixerOrchestrator::initializeRoutedRoutes() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    for (const auto& [router_name, expected_count] : state_->router_output_counts) {
        const int actual_count = routerOutputCount(nodes_, router_name);
        if (expected_count != actual_count) {
            throw Error("mixer.init_routes: router " + router_name + " expected output count " +
                        std::to_string(expected_count) + " but node dst has " +
                        std::to_string(actual_count));
        }
        ensureRouteTableSize(*state_, router_name);
    }
    if (state_->pgm_scene_name.empty() || !state_->scenes.count(state_->pgm_scene_name))
        return;
    publishRoutedRoutesForProgramOnly(
        state_->pgm_is_slot_a,
        state_->scenes.at(state_->pgm_scene_name),
        wallclock.pts(),
        true);
}

void MixerOrchestrator::applyPostTransitionRouting(bool new_pgm_is_slot_a,
                                                   const std::string& new_pgm_scene) {
    const auto scene_it = state_->scenes.find(new_pgm_scene);
    if (scene_it == state_->scenes.end())
        return;

    const SceneDefinition& scene = scene_it->second;
    const uint32_t pgm_bit = new_pgm_is_slot_a ? 1u : 2u;
    const SourceMask active = state_->computeActiveInputsMask(scene);
    const auto& new_slot = new_pgm_is_slot_a ? state_->slot_a : state_->slot_b;
    const auto& old_slot = new_pgm_is_slot_a ? state_->slot_b : state_->slot_a;

    // Source_switcher first: this is the only setting visible at the SDI output.
    // Any short window between this and the OTM/compositor flips below would only
    // surface if the new direct path were not already producing frames; in both
    // callers (ready cut and deferred fade cleanup) it is.
    timeline_->clearKey(state_->source_switcher_name, "active");
    setNodeObject(state_->source_switcher_name, "active",
                  Parameters(new_pgm_is_slot_a ? 0 : 1));

    // The encoder must not make the receiver wait for the next periodic keyframe:
    // a cut changes the whole picture, and a P-frame carrying it can exceed what
    // the receiver can recover from. The node coalesces bursts into one keyframe.
    if (!state_->keyframe_node_name.empty()) {
        try {
            setNodeObject(state_->keyframe_node_name, "trigger", Parameters(true));
        } catch (const std::exception& e) {
            logstream << "mixer: keyframe trigger failed: " << e.what();
        }
    }

    for (const auto& [src_name, info] : state_->sources) {
        if (info.routed)
            continue;
        const bool in_scene = scene.sources.count(src_name) > 0;
        const bool active_input = active.test(info.input_index);
        const uint32_t mask = state_->sourceOutputMask(info, (in_scene && active_input) ? pgm_bit : 0u);
        timeline_->clearKey(info.otm_node_name, "outputs");
        setNodeObjectIfCreated(nodes_, info.otm_node_name, "outputs", Parameters(mask));
    }
    publishRoutedRoutesForProgramOnly(new_pgm_is_slot_a, scene, wallclock.pts(), true);

    timeline_->clearKey(new_slot.post_otm_name, "outputs");
    timeline_->clearKey(old_slot.post_otm_name, "outputs");
    timeline_->clearKey(new_slot.compositor_name, "active_inputs");
    timeline_->clearKey(old_slot.compositor_name, "active_inputs");
    nodes_->node(new_slot.post_otm_name)->setObject("outputs", Parameters(1u));
    nodes_->node(old_slot.post_otm_name)->setObject("outputs", Parameters(0u));
    nodes_->node(new_slot.compositor_name)->setObject("active_inputs", toParameters(active));
    nodes_->node(old_slot.compositor_name)->setObject("active_inputs", Parameters(0u));
}

void MixerOrchestrator::rewriteCameraOutputsForSlot(uint32_t slot_bit, const SceneDefinition& scene) {
    const SourceMask active = state_->computeActiveInputsMask(scene);
    for (const auto& [src_name, info] : state_->sources) {
        if (info.routed)
            continue;
        Parameters current_val;
        uint32_t mask = nodes_->node(info.otm_node_name)->getObjectTry("outputs", current_val)
                            ? current_val.get<uint32_t>()
                            : 0u;
        mask &= ~slot_bit;
        if (scene.sources.count(src_name) && active.test(info.input_index))
            mask |= slot_bit;
        publishCameraOtmOutputs(info.otm_node_name, state_->sourceOutputMask(info, mask));
    }
}

void MixerOrchestrator::loadSceneIntoSlot(bool is_slot_a, const std::string& scene_name, bool warm_cut) {
    auto& scene = state_->scenes.at(scene_name);
    const auto& slot = is_slot_a ? state_->slot_a : state_->slot_b;

    // The slot being loaded is the broadcast-inactive PVW slot.  Its compositor
    // was previously idled with active_inputs=0, so it may still hold frames on
    // its input edges. If left there, the next activation starts by rendering
    // stale frames and appears to lag behind the scene switch.
    flushSlotEdges(is_slot_a);
    // Reset on the compositor's worker thread and reject frames from before
    // this load, including those still travelling through live upstream edges.
    if (warm_cut && state_->prewarm_cut_scenes.count(scene_name) && canPrewarmScene(scene))
        setNodeObject(slot.compositor_name, "warm_reset", Parameters(true));
    else
        resetInputIf(nodes_, slot.compositor_name);

    for (const auto& [src_name, layout] : scene.sources) {
        auto src_it = state_->sources.find(src_name);
        if (src_it == state_->sources.end()) continue;
        const auto& info = src_it->second;
        const std::string& cs_node = is_slot_a ? info.cs_node_a : info.cs_node_b;

        if (cs_node.empty()) {
            if (!layout.crop_scale_graph.empty())
                throw Error("mixer: source " + src_name + " has no filter node for its scene graph");
            continue;
        }

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
            logstream << "mixer: " << cs_node << " graph unchanged, no restart";
        } else {
            logstream << "mixer: " << cs_node << " graph changed (\"" << old_graph << "\" -> \""
                      << layout.crop_scale_graph << "\"), restarting";
            setNodeParam(cs_node, "graph", layout.crop_scale_graph);
            autoRestartNode(cs_node);
        }
    }

    setNodeObject(slot.compositor_name, "layers", compositorLayersFromScene(*state_, scene));

    const SourceMask active_mask = state_->computeActiveInputsMask(scene);
    // Same pattern as camera otms: cuda_rect_overlay reads "active_inputs" from timeline only.
    // clearKey does not touch "layers" or other keys on this compositor channel.
    timeline_->clearKey(slot.compositor_name, "active_inputs");
    setNodeObject(slot.compositor_name, "active_inputs", toParameters(active_mask));
    timeline_->set(slot.compositor_name, "active_inputs", wallclock.pts(), toParameters(active_mask));

    // Drop slot bit for every camera, then enable only sources in scene with active_inputs set.
    // Keeps `outputs` consistent with compositor consumption (no frames into unused inputs).
    const uint32_t slot_bit = is_slot_a ? 1u : 2u;
    rewriteCameraOutputsForSlot(slot_bit, scene);
    applyRoutedSceneRoutesForSlot(is_slot_a, scene, wallclock.pts(), true);

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

    int64_t prep_ms = wallclock.pts();
    timeline_->clearKey(slot.post_otm_name, "outputs");
    setNodeObject(slot.post_otm_name, "outputs", Parameters(1u));
    timeline_->set(slot.post_otm_name, "outputs", prep_ms, Parameters(1u));
    // Direct transitions also load this slot; only an explicit preview should
    // publish its scene to the control UI and AUX preview follower.
    state_->pvw_scene_name = scene_name;
    logstream << "mixer preview armed: scene=" << scene_name
              << " slot=" << (pvw_is_slot_a ? 'A' : 'B')
              << " post_otm " << slot.post_otm_name << "->1";
}

}  // namespace avp::mixer
