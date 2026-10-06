// Sources, scenes and slot loading: what a scene means in node parameters and
// how the PVW slot is prepared before a transition.
#include "internal.hpp"

namespace avp::mixer {

namespace {
void checkInputIndex(int input_index) {
    if (input_index < 0 || input_index >= kSourceMaskBits)
        throw Error("mixer source input_index must be in 0.." + std::to_string(kSourceMaskBits - 1));
}
}

void MixerOrchestrator::defineSource(const std::string& name, const std::string& otm_node, int input_index) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    checkInputIndex(input_index);
    MixerState::SourceInfo info;
    info.otm_node_name = otm_node;
    info.input_index = input_index;
    state_->sources[name] = std::move(info);
}

void MixerOrchestrator::defineRoutedSource(const std::string& name, const std::string& router_node,
                                           int input_index,
                                           const std::string& route_output_label_a,
                                           const std::string& route_output_label_b) {
    const int output_count = routerOutputCount(nodes_, router_node);
    const int route_output_a = routerOutputIndexFromLabel(nodes_, router_node, route_output_label_a);
    const int route_output_b = routerOutputIndexFromLabel(nodes_, router_node, route_output_label_b);

    std::lock_guard<std::mutex> lock(state_->mutex);
    checkInputIndex(input_index);
    MixerState::SourceInfo info;
    info.input_index = input_index;
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
        if (source == state_->sources.end() || source->second.routed) return false;
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
            throw Error("mixer.prewarm: scenes require fixed sources without routes or controls: " + name);
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
    const auto* pgm = &state_->scenes.at(state_->pgm_scene_name);
    const auto* pvw = state_->pvw_slot_scene.empty() ? nullptr : &state_->scenes.at(state_->pvw_slot_scene);
    publishSourceRoutes(state_->pgm_is_slot_a ? pgm : pvw, state_->pgm_is_slot_a ? pvw : pgm);
}

void MixerOrchestrator::publishSourceRoutes(const SceneDefinition* scene_a, const SceneDefinition* scene_b) {
    auto tables = currentRouterTables(*state_);
    setRoutedSlotInTables(*state_, tables, true, scene_a);
    setRoutedSlotInTables(*state_, tables, false, scene_b);
    for (const auto& [name, source] : state_->sources) {
        if (!source.routed)
            publishRuntimeObject(source.otm_node_name, "outputs",
                sourceOutputsForScenes(*state_, name, source, scene_a, scene_b));
    }
    for (const auto& [router, routes] : tables) {
        publishRuntimeObject(router, "routes", routesToParameters(routes));
        state_->router_routes[router] = routes;
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
    const auto* scene = &state_->scenes.at(state_->pgm_scene_name);
    publishSourceRoutes(state_->pgm_is_slot_a ? scene : nullptr, state_->pgm_is_slot_a ? nullptr : scene);
}

int64_t MixerOrchestrator::switchProgramSelector(bool new_pgm_is_slot_a) {
    setNodeObject(state_->source_switcher_name, "active",
                  Parameters(new_pgm_is_slot_a ? 0 : 1));
    // Read at once: the selector drains its inactive inputs, so the new program's first frame
    // arrives at the next main deadline, and the routing could take until then. The followers
    // of a take get their change now, for the same reason.
    return selectorOutputNs();
}

void MixerOrchestrator::applyPostTransitionRouting(bool new_pgm_is_slot_a,
                                                   const std::string& new_pgm_scene,
                                                   bool picture_changed) {
    const auto scene_it = state_->scenes.find(new_pgm_scene);
    if (scene_it == state_->scenes.end())
        return;

    const SceneDefinition& scene = scene_it->second;
    const SourceMask active = state_->computeActiveInputsMask(scene);
    const auto& new_slot = new_pgm_is_slot_a ? state_->slot_a : state_->slot_b;
    const auto& old_slot = new_pgm_is_slot_a ? state_->slot_b : state_->slot_a;

    // The encoder must not make the receiver wait for the next periodic keyframe:
    // a cut changes the whole picture, and a P-frame carrying it can exceed what
    // the receiver can recover from. The node coalesces bursts into one keyframe.
    if (picture_changed && !state_->keyframe_node_name.empty()) {
        try {
            setNodeObject(state_->keyframe_node_name, "trigger", Parameters(true));
        } catch (const std::exception& e) {
            logstream << "mixer: keyframe trigger failed: " << e.what();
        }
    }

    publishSourceRoutes(new_pgm_is_slot_a ? &scene : nullptr, new_pgm_is_slot_a ? nullptr : &scene);

    setNodeObject(new_slot.post_otm_name, "outputs", Parameters(1u));
    setNodeObject(old_slot.post_otm_name, "outputs", Parameters(0u));
    setNodeObject(new_slot.compositor_name, "active_inputs", toParameters(active));
    setNodeObject(old_slot.compositor_name, "active_inputs", Parameters(0u));
}

void MixerOrchestrator::loadSceneIntoSlot(bool is_slot_a, const std::string& scene_name, bool warm_cut) {
    auto& scene = state_->scenes.at(scene_name);
    auto& slot = is_slot_a ? state_->slot_a : state_->slot_b;
    // Only frames carrying the new composition revision can complete this take.
    // The compositor resets stale input on its own render thread.
    state_->pvw_slot_scene.clear();

    const SourceMask active_mask = state_->computeActiveInputsMask(scene);
    slot.revision = std::to_string(++state_->scene_revision);
    setNodeObject(slot.compositor_name, "composition", {
        {"layers", compositorLayersFromScene(*state_, scene)},
        {"active_inputs", toParameters(active_mask)}, {"revision", slot.revision},
        {"warm", warm_cut && state_->prewarm_cut_scenes.count(scene_name) && canPrewarmScene(scene)}});

    const auto* pgm = &state_->scenes.at(state_->pgm_scene_name);
    publishSourceRoutes(is_slot_a ? &scene : pgm, is_slot_a ? pgm : &scene);
    state_->pvw_slot_scene = scene_name;
}

void MixerOrchestrator::applySceneControls(const SceneDefinition& scene) {
    for (const auto& control : scene.controls) {
        setNodeObject(control.node_name, control.key, control.value);
        logstream << "mixer scene control: " << control.node_name << "." << control.key
                  << " -> " << control.value;
    }
}

void MixerOrchestrator::preview(const std::string& scene_name) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    ensureIdle();
    if (!state_->scenes.count(scene_name))
        throw Error("mixer.preview: unknown scene: " + scene_name);

    bool pvw_is_slot_a = !state_->pgm_is_slot_a;
    const auto& slot = state_->pvwSlot();

    if (state_->pvw_slot_scene == scene_name) {
        logstream << "mixer preview: scene already loaded in PVW: " << scene_name;
    } else {
        loadSceneIntoSlot(pvw_is_slot_a, scene_name);
    }

    int64_t prep_ms = wallclock.pts();
    setNodeObject(slot.post_otm_name, "outputs", Parameters(1u));
    // Direct transitions also load this slot; only an explicit preview should
    // publish its scene to the control UI and the AUX preview followers.
    state_->publishPreview(scene_name, 0);
    logstream << "mixer preview armed: scene=" << scene_name
              << " slot=" << (pvw_is_slot_a ? 'A' : 'B')
              << " post_otm " << slot.post_otm_name << "->1";
}

}  // namespace avp::mixer
