#pragma once
// Lookups the orchestrator used to do on NodeWrapper, now forwarded to MixerGraph.
// The NodeManager bodies live on NodeManagerMixerGraph.
#include "primitives/MixerState.hpp"
#include "MixerGraph.hpp"
#include "../avutils.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <chrono>

namespace avp::mixer::graph {

/// Poll period of every readiness wait in the orchestrator.
constexpr int64_t kPollMs = 5;
constexpr int64_t kWipeReadyTimeoutMs = 5000;
/// How long a cut or fade waits for its target slot's first fresh frame before it gives
/// up and keeps the program, so a dead source never parks the mixer in a transition.
constexpr int64_t kTakeReadyTimeoutMs = 2000;

inline void resetInputIf(const std::shared_ptr<MixerGraph>& nodes, const std::string& name) {
    nodes->resetInput(name);
}

inline bool setNodeObjectIfCreated(const std::shared_ptr<MixerGraph>& nodes, const std::string& node_name,
                                   const std::string& key, const Parameters& value) {
    return nodes->setObjectIfCreated(node_name, key, value);
}

inline std::string firstDstEdgeName(const std::shared_ptr<MixerGraph>& nodes, const std::string& node_name) {
    return nodes->firstDstEdgeName(node_name);
}

inline std::string edgeNameAt(const std::shared_ptr<MixerGraph>& nodes, const std::string& node_name,
                              const std::string& param_name, size_t index) {
    return nodes->edgeNameAt(node_name, param_name, index);
}

inline av::Timestamp edgeLastTsIfExists(const std::shared_ptr<MixerGraph>& nodes, const std::string& name) {
    return nodes->edgeLastTimestamp(name);
}

inline int routerOutputCount(const std::shared_ptr<MixerGraph>& nodes, const std::string& router_name) {
    return nodes->routerOutputCount(router_name);
}

inline int routerOutputIndexFromLabel(const std::shared_ptr<MixerGraph>& nodes, const std::string& router_name,
                                      const std::string& label) {
    return nodes->routerOutputIndexFromLabel(router_name, label);
}

struct WipeReadyResult {
    bool ready = false;
    int64_t waited_ms = 0;
};

/// Poll until `edge_name` carries a frame newer than `initial_ts`, the transition generation
/// moves on, or `timeout_ms` passes.
inline WipeReadyResult waitForWipeOverlayReady(const std::shared_ptr<MixerGraph>& nodes, const std::string& edge_name,
                                               av::Timestamp initial_ts,
                                               const std::shared_ptr<MixerState>& state, uint64_t generation,
                                               int64_t timeout_ms = kWipeReadyTimeoutMs) {
    WipeReadyResult result;
    while (result.waited_ms < timeout_ms) {
        if (state->transition_generation.load() != generation) return result;
        av::Timestamp ts = nodes->edgeLastTimestamp(edge_name);
        const bool frame_ready = ts.isValid() && (!initial_ts.isValid() || ts > initial_ts);
        if (frame_ready) {
            result.ready = true;
            return result;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kPollMs));
        result.waited_ms += kPollMs;
    }
    return result;
}

}
