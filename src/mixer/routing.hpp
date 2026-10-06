#pragma once
// Pure functions from MixerState/SceneDefinition to node parameters: router
// route tables and the compositor layer array. No node access, unit-testable.
#include "primitives/MixerState.hpp"
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace avp::mixer {

inline Parameters routesToParameters(const std::vector<int>& routes) {
    Parameters arr = Parameters::array();
    for (int input_index : routes)
        arr.push_back(input_index);
    return arr;
}

inline void ensureRouteTableSize(MixerState& st, const std::string& router_name) {
    int count = st.router_output_counts[router_name];
    auto& routes = st.router_routes[router_name];
    if ((int)routes.size() != count)
        routes.assign(count, -1);
}

inline std::unordered_map<std::string, std::vector<int>> currentRouterTables(MixerState& st) {
    std::unordered_map<std::string, std::vector<int>> tables;
    for (const auto& [router_name, count] : st.router_output_counts) {
        ensureRouteTableSize(st, router_name);
        tables[router_name] = st.router_routes[router_name];
    }
    return tables;
}

inline int routeOutputForSlot(const MixerState::SourceInfo& info, bool is_slot_a) {
    return is_slot_a ? info.route_output_a : info.route_output_b;
}

inline std::string routeOutputLabelForSlot(const MixerState::SourceInfo& info, bool is_slot_a) {
    return is_slot_a ? info.route_output_label_a : info.route_output_label_b;
}

/// Rewrite one slot's entries in every router table: clear this slot's outputs, then point the
/// outputs of sources used by `scene` (nullptr: none) at the scene's explicit routes.
inline void setRoutedSlotInTables(MixerState& st,
                           std::unordered_map<std::string, std::vector<int>>& tables,
                           bool is_slot_a,
                           const SceneDefinition* scene) {
    for (const auto& [src_name, info] : st.sources) {
        if (!info.routed)
            continue;
        const int output_index = routeOutputForSlot(info, is_slot_a);
        if (output_index < 0)
            continue;

        auto& routes = tables[info.router_node_name];
        const int count = st.router_output_counts[info.router_node_name];
        if ((int)routes.size() != count)
            routes.assign(count, -1);
        if (output_index >= (int)routes.size())
            throw Error("mixer: routed source " + src_name + " output index " +
                        std::to_string(output_index) + " (" +
                        routeOutputLabelForSlot(info, is_slot_a) + ") exceeds router " +
                        info.router_node_name + " route table");

        routes[output_index] = -1;
        if (!scene || !scene->sources.count(src_name))
            continue;

        auto route_it = scene->routes.find(src_name);
        if (route_it == scene->routes.end()) {
            throw Error("mixer: scene " + scene->name + " uses routed source " +
                        src_name + " without an explicit route");
        }
        routes[output_index] = route_it->second;
    }
}

/// Fanout bits are derived from the two requested scenes, never read back from nodes.
inline uint32_t sourceOutputsForScenes(const MixerState& state, const std::string& name,
        const MixerState::SourceInfo& source, const SceneDefinition* scene_a, const SceneDefinition* scene_b) {
    const uint32_t requested = (scene_a && scene_a->sources.count(name) ? 1u : 0u) |
                               (scene_b && scene_b->sources.count(name) ? 2u : 0u);
    return state.sourceOutputMask(source, requested);
}

/// Explicit inputs allow sparse layers. Input order preserves ties in compositor z-order.
inline Parameters compositorLayersFromScene(const MixerState& st, const SceneDefinition& scene) {
    std::map<int, Parameters> ordered;
    for (const auto& [name, spec] : scene.sources) {
        const int input = st.sources.at(name).input_index;
        auto layer = spec;
        layer["input"] = input;
        ordered.emplace(input, std::move(layer));
    }
    Parameters layers = Parameters::array();
    for (auto& [input, layer] : ordered) layers.push_back(std::move(layer));
    return layers;
}

}
