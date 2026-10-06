#pragma once
// Thin helpers over NodeManager/edges used by the orchestrator: lookups that
// tolerate missing nodes, setObject that queues for not-yet-created nodes,
// router label resolution, and the readiness poll for wipe warmup.
#include "primitives/MixerState.hpp"
#include "MixerGraph.hpp"
#include "../avutils.hpp"
#include "../graph_mgmt.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace avp::mixer::graph {

/// Poll period of every readiness wait in the orchestrator.
constexpr int64_t kPollMs = 5;
constexpr int64_t kWipeReadyTimeoutMs = 5000;
/// How long a cut or fade waits for its target slot's first fresh frame before it gives
/// up and keeps the program, so a dead source never parks the mixer in a transition.
constexpr int64_t kTakeReadyTimeoutMs = 2000;

void resetInputIf(const std::shared_ptr<MixerGraph>& nodes, const std::string& name);

/// After a PGM path change, slot compositors may have been idle (`active_inputs=0`) while cameras
/// advanced in PTS; `force_fps` on `norm_*` would otherwise compare the next real frame to a stale
/// grid and produce a big `Discontinuity` jump with a duplicate burst across the cut.
///
/// Deliberately NOT resetting the final post-`out_sel` `force_fps` (e.g. `mixer_norm_fps`): it is
/// the last VFR guard before the encoder and resetting it would drop that guarantee.
void resetSlotNormFps(const std::shared_ptr<MixerGraph>& nodes, const MixerState& st);

std::shared_ptr<NodeWrapper> workingConsumerForEdge(const std::shared_ptr<MixerGraph>& nodes,
                                                    const std::string& edge_name);
/// Stores the value in the wrapper's parameters and applies it when the node exists; false when
/// the node is not created yet (the value is picked up at creation).
bool setNodeObjectIfCreated(const std::shared_ptr<MixerGraph>& nodes, const std::string& node_name,
                            const std::string& key, const Parameters& value);
std::string firstDstEdgeName(const std::shared_ptr<MixerGraph>& nodes, const std::string& node_name);
std::string edgeNameAt(const std::shared_ptr<MixerGraph>& nodes, const std::string& node_name,
                       const std::string& param_name, size_t index);
av::Timestamp edgeLastTsIfExists(const std::shared_ptr<MixerGraph>& nodes, const std::string& name);
int routerOutputCount(const std::shared_ptr<MixerGraph>& nodes, const std::string& router_name);
int routerOutputIndexFromLabel(const std::shared_ptr<MixerGraph>& nodes, const std::string& router_name,
                               const std::string& label);

struct WipeReadyResult {
    bool ready = false;
    int64_t waited_ms = 0;
    av::Timestamp ready_ts = NOTS;
};

/// Poll until `edge_name` carries a frame newer than `initial_ts` and wallclock reached
/// `earliest_visible_pts_ms`, or the transition generation moved on, or `timeout_ms` passed.
WipeReadyResult waitForWipeOverlayReady(const std::shared_ptr<MixerGraph>& nodes, const std::string& edge_name,
                                        av::Timestamp initial_ts, int64_t earliest_visible_pts_ms,
                                        const std::shared_ptr<MixerState>& state, uint64_t generation,
                                        int64_t timeout_ms = kWipeReadyTimeoutMs);

}
