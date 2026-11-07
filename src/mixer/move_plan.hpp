#pragma once
// Planning a move take (primitives/layer_move.hpp): Program's scene A and the target B are drawn
// together on B's slot, so that slot takes the inputs and routes of both. Pure, over MixerState;
// MixerOrchestrator::move (orchestrator/move.cpp) uses it.
#include "routing.hpp"
#include <optional>
#include <string>

namespace avp::mixer {

/// What the destination slot draws its inputs and routes for while the move runs: B, plus A's
/// sources B lacks, with A's routes. Nullopt with `why` when a routed source is in both scenes on
/// different routes: the slot's one router output for it carries one route.
inline std::optional<SceneDefinition> moveUnionScene(const SceneDefinition& a, const SceneDefinition& b, std::string& why) {
    SceneDefinition together = b;
    for (const auto& [name, layer] : a.sources) {
        const auto route_a = a.routes.find(name);
        if (b.sources.count(name)) {
            const auto route_b = b.routes.find(name);
            if (route_a != a.routes.end() && route_b != b.routes.end() && route_a->second != route_b->second) {
                why = "routed source " + name + " changes route";
                return std::nullopt;
            }
            continue;
        }
        together.sources.emplace(name, layer);
        if (route_a != a.routes.end()) together.routes[name] = route_a->second;
    }
    return together;
}

/// The move spec (layer_move.hpp MoveSpec as JSON) for a take from `a` to `b` that the take clock's
/// `generation` runs, or null with `why`, for which the take fades instead.
inline Parameters planMove(const MixerState& st, const SceneDefinition& a, const SceneDefinition& b, uint64_t generation,
                           const std::string& unmatched, std::string& why) {
    if (unmatched != "fade" && unmatched != "cut") throw Error("mixer: a move's unmatched sources fade or cut");
    if (!moveUnionScene(a, b, why)) return nullptr;
    return {{"generation", generation}, {"from", compositorLayersFromScene(st, a)}, {"unmatched", unmatched}};
}

}
