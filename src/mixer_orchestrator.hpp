#pragma once
#include "MixerState.hpp"
#include "SharedTimeline.hpp"
#include "graph_mgmt.hpp"
#include "instance_shared.hpp"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

class MixerTransitionScheduler : public InstanceShared<MixerTransitionScheduler>, public IShutdownable {
    struct Impl;
    std::unique_ptr<Impl> impl_;

public:
    MixerTransitionScheduler();
    ~MixerTransitionScheduler();

    void post(std::string label, std::function<void()> task);
    void postAfter(std::string label, int64_t delay_ms, std::function<void()> task);
    void shutdown();
};

class MixerOrchestrator {
    std::shared_ptr<NodeManager> nodes_;
    std::shared_ptr<MixerState> state_;
    std::shared_ptr<SharedTimeline> timeline_;
    std::shared_ptr<MixerTransitionScheduler> scheduler_;

    void setNodeObject(const std::string& node_name, const std::string& key, const Parameters& value);
    void postTransitionTask(std::string label, int64_t delay_ms, std::function<void()> task);
    void publishRuntimeObject(const std::string& node_name, const std::string& key,
                              const Parameters& value);

    /// Camera `one_to_many` uses `timeline` + `tlGetRaw`: stale `outputs` entries would override
    /// `node.object.set`. Clear prior `outputs` schedule, update atomically, then publish at current wallclock.
    void publishCameraOtmOutputs(const std::string& otm_name, uint32_t mask);
    void setNodeParam(const std::string& node_name, const std::string& param, const std::string& value);
    void autoRestartNode(const std::string& node_name);
    void createAndStartNode(const Parameters& params);
    void deleteNodeIfExists(const std::string& name);
    void startGroup(const std::string& group_name);
    void stopGroup(const std::string& group_name);

    // Flush all wipe pipeline edges listed in state_->wipe_flush_edges.
    // Skips edges whose configured consumer is still working; readerwriterqueue
    // clear() is a consumer-side operation and must not race that node.
    void flushWipeEdges();
    void flushSlotEdges(bool is_slot_a);

    void loadSceneIntoSlot(bool is_slot_a, const std::string& scene_name);
    void scheduleSceneControls(const SceneDefinition& scene, int64_t at_pts_ms);

    /// Rewrite every camera `one_to_many` bitmask for one slot bit from scene + active_inputs.
    void rewriteCameraOutputsForSlot(uint32_t slot_bit, const SceneDefinition& scene);
    void applyRoutedSceneRoutesForSlot(bool is_slot_a, const SceneDefinition& scene,
                                       int64_t at_pts_ms, bool immediate);
    void publishRoutedRoutesForProgramOnly(bool pgm_is_slot_a, const SceneDefinition& scene,
                                           int64_t at_pts_ms, bool immediate);

    /// Restore steady-state routing after a cut or crossfade has concluded.
    /// Centralizes the per-source OTM masks, post-otm/compositor flips, and
    /// source_switcher reset that cut+fade cleanup paths used to duplicate.
    /// Caller must hold state_->mutex. Wipe end has different semantics
    /// (timeline-driven, doesn't immediately mutate node objects) and uses
    /// its own logic.
    void applyPostTransitionRouting(bool new_pgm_is_slot_a, const std::string& new_pgm_scene);

    void ensureIdle() const;
    int64_t resolveTransitionStartPts(int64_t requested_start_pts_ms) const;

    // Core hard-cut logic: ensure PVW is configured, enable cameras, write timeline entries.
    // Does NOT modify pgm_is_slot_a, pgm_scene_name, or transition_mode.
    // Caller must hold state_->mutex. Returns T_cleanup timestamp.
    int64_t cutInternal(const std::string& scene_name, int64_t start_pts_ms);

    // Generic deferred cleanup: flip state + optionally delete nodes. The caller
    // schedules this for non-PTS-expressible work (node deletion and bookkeeping).
    // `scheduler` is forwarded into the locally-constructed MixerOrchestrator so
    // any future scheduler-using helper called from this path won't blow up with
    // "transition scheduler is not configured".
    static void deferredCleanup(std::shared_ptr<NodeManager> nodes,
                                 std::shared_ptr<MixerState> state,
                                 std::shared_ptr<SharedTimeline> timeline,
                                 std::shared_ptr<MixerTransitionScheduler> scheduler,
                                 uint64_t transition_generation,
                                 bool new_pgm_is_slot_a,
                                 std::string new_pgm_scene,
                                 std::vector<std::string> nodes_to_delete);
    static void readyCutTask(std::shared_ptr<NodeManager> nodes,
                             std::shared_ptr<MixerState> state,
                             std::shared_ptr<SharedTimeline> timeline,
                             std::shared_ptr<MixerTransitionScheduler> scheduler,
                             uint64_t transition_generation,
                             bool new_pgm_is_slot_a,
                             std::string new_pgm_scene,
                             std::string ready_edge_name,
                             av::Timestamp ready_edge_initial_ts,
                             int64_t earliest_switch_pts_ms,
                             bool require_new_ready_frame);

    // Wipe lifecycle: midpoint scene switch (immediate, hidden behind opaque wipe)
    // + end-of-wipe teardown.
    static void runWipeMidpointAndCleanup(std::shared_ptr<NodeManager> nodes,
                                          std::shared_ptr<MixerState> state,
                                          std::shared_ptr<SharedTimeline> timeline,
                                          std::shared_ptr<MixerTransitionScheduler> scheduler,
                                          uint64_t transition_generation,
                                          std::string scene_name,
                                          bool new_pgm_is_slot_a,
                                          int64_t remaining_ms);
    static int64_t prepareWipe(std::shared_ptr<NodeManager> nodes,
                               std::shared_ptr<MixerState> state,
                               std::shared_ptr<SharedTimeline> timeline,
                               std::shared_ptr<MixerTransitionScheduler> scheduler,
                               uint64_t transition_generation,
                               std::string scene_name,
                               std::string wipe_file,
                               double duration_sec,
                               bool new_pgm_is_slot_a,
                               int64_t earliest_visible_pts_ms);

    std::string transition_node_name_ = "mixer_transition";
    std::string transition_edge_name_ = "trans_out";

public:
    MixerOrchestrator(std::shared_ptr<NodeManager> nodes,
                      std::shared_ptr<MixerState> state,
                      std::shared_ptr<SharedTimeline> timeline,
                      std::shared_ptr<MixerTransitionScheduler> scheduler = nullptr);

    void defineSource(const std::string& name, const std::string& otm_node, int input_index,
                      const std::string& cs_node_a, const std::string& cs_node_b);
    void defineRoutedSource(const std::string& name, const std::string& router_node,
                            int input_index,
                            const std::string& route_output_label_a,
                            const std::string& route_output_label_b,
                            const std::string& cs_node_a, const std::string& cs_node_b);
    void defineScene(const std::string& name, const SceneDefinition& def);
    void initializeRoutedRoutes();

    void preview(const std::string& scene_name);
    void cut(const std::string& scene_name, int64_t start_pts_ms = -1);
    void fade(const std::string& scene_name, double duration_sec, int64_t start_pts_ms = -1);
    void wipe(const std::string& scene_name, const std::string& wipe_file, double duration_sec,
              int64_t start_pts_ms = -1);
    void setOverlayEnabled(bool enabled, int64_t ready_timeout_ms = -1);

    /// Returns the names of all registered scenes, sorted alphabetically.
    std::vector<std::string> sceneNames() const;
    Parameters status() const;
};
