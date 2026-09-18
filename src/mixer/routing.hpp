#pragma once
// Pure functions from MixerState/SceneDefinition to node parameters: router
// route tables and the compositor layer array. No node access, unit-testable.
#include "primitives/MixerState.hpp"
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
        if ((int)tables[router_name].size() != count)
            tables[router_name].assign(count, -1);
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

/// One cuda_rect_overlay layer per compositor src index (see mixer.source). Omitted sources use a dummy rect.
inline Parameters compositorLayersFromScene(const MixerState& st, const SceneDefinition& scene) {
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

}
