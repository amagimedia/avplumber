#pragma once
// Thin helpers over NodeManager/edges used by the orchestrator: lookups that
// tolerate missing nodes, setObject that queues for not-yet-created nodes,
// router label resolution, and the readiness polls for wipe/overlay branches.
#include "primitives/MixerState.hpp"
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

void resetInputIf(std::shared_ptr<NodeManager> nodes, const std::string& name);

/// After a PGM path change, slot compositors may have been idle (`active_inputs=0`) while cameras
/// advanced in PTS; `force_fps` on `norm_*` would otherwise compare the next real frame to a stale
/// grid and produce a big `Discontinuity` jump with a duplicate burst across the cut.
///
/// Deliberately NOT resetting the final post-`out_sel` `force_fps` (e.g. `mixer_norm_fps`): it is
/// the last VFR guard before the encoder and resetting it would drop that guarantee.
void resetSlotNormFps(std::shared_ptr<NodeManager> nodes, const MixerState& st);

bool nodeWorkingIfExists(std::shared_ptr<NodeManager> nodes, const std::string& name);
std::shared_ptr<NodeWrapper> workingConsumerForEdge(std::shared_ptr<NodeManager> nodes,
                                                    const std::string& edge_name);
/// Stores the value in the wrapper's parameters and applies it when the node exists; false when
/// the node is not created yet (the value is picked up at creation).
bool setNodeObjectIfCreated(std::shared_ptr<NodeManager> nodes, const std::string& node_name,
                            const std::string& key, const Parameters& value);
int edgeOccupiedIfExists(std::shared_ptr<NodeManager> nodes, const std::string& name);
std::string firstDstEdgeName(std::shared_ptr<NodeManager> nodes, const std::string& node_name);
std::string edgeNameAt(std::shared_ptr<NodeManager> nodes, const std::string& node_name,
                       const std::string& param_name, size_t index);
av::Timestamp edgeLastTsIfExists(std::shared_ptr<NodeManager> nodes, const std::string& name);
int routerOutputCount(std::shared_ptr<NodeManager> nodes, const std::string& router_name);
int routerOutputIndexFromLabel(std::shared_ptr<NodeManager> nodes, const std::string& router_name,
                               const std::string& label);

struct WipeReadyResult {
    bool ready = false;
    int64_t waited_ms = 0;
    av::Timestamp ready_ts = NOTS;
};

struct OverlayReadyResult {
    bool ready = false;
    bool cancelled = false;
    int64_t waited_ms = 0;
    av::Timestamp ready_ts = NOTS;
};

/// Poll until `edge_name` carries a frame newer than `initial_ts` and wallclock reached
/// `earliest_visible_pts_ms`, or the transition generation moved on, or `timeout_ms` passed.
WipeReadyResult waitForWipeOverlayReady(std::shared_ptr<NodeManager> nodes, const std::string& edge_name,
                                        av::Timestamp initial_ts, int64_t earliest_visible_pts_ms,
                                        const std::shared_ptr<MixerState>& state, uint64_t generation,
                                        int64_t timeout_ms = kWipeReadyTimeoutMs);
bool overlayCommandCurrent(const std::shared_ptr<MixerState>& state, uint64_t generation);
OverlayReadyResult waitForOverlayBranchReady(std::shared_ptr<NodeManager> nodes,
                                             std::shared_ptr<MixerState> state, uint64_t generation,
                                             const std::string& edge_name, av::Timestamp initial_ts,
                                             av::Timestamp minimum_ts, int64_t timeout_ms, int64_t poll_ms);

}
