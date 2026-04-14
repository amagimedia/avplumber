#pragma once
#include "MixerState.hpp"
#include "SharedTimeline.hpp"
#include "graph_mgmt.hpp"
#include <memory>
#include <string>
#include <vector>

class MixerOrchestrator {
    std::shared_ptr<NodeManager> nodes_;
    std::shared_ptr<MixerState> state_;
    std::shared_ptr<SharedTimeline> timeline_;

    void setNodeObject(const std::string& node_name, const std::string& key, const Parameters& value);
    void setNodeParam(const std::string& node_name, const std::string& param, const std::string& value);
    void autoRestartNode(const std::string& node_name);
    void createAndStartNode(const Parameters& params);
    void deleteNodeIfExists(const std::string& name);

    void setCameraOutputBits(const SceneDefinition& scene, uint32_t slot_bit, bool set);
    void loadSceneIntoSlot(bool is_slot_a, const std::string& scene_name);

    void ensureIdle() const;

    // Core hard-cut logic: reconfigure PVW, enable cameras, write timeline entries.
    // Does NOT modify pgm_is_slot_a, pgm_scene_name, or transition_mode.
    // Caller must hold state_->mutex. Returns T_cleanup timestamp.
    int64_t cutInternal(const std::string& scene_name);

    // Generic deferred cleanup: sleep, then flip state + optionally delete nodes.
    // Runs on a detached thread — only used for non-PTS-expressible work
    // (node deletion and internal bookkeeping flip).
    static void deferredCleanup(std::shared_ptr<NodeManager> nodes,
                                std::shared_ptr<MixerState> state,
                                int64_t sleep_ms,
                                bool new_pgm_is_slot_a,
                                std::string new_pgm_scene,
                                std::vector<std::string> nodes_to_delete);

    // Wipe lifecycle: midpoint scene switch (immediate, hidden behind opaque wipe)
    // + end-of-wipe teardown.
    static void wipeThread(std::shared_ptr<NodeManager> nodes,
                           std::shared_ptr<MixerState> state,
                           std::shared_ptr<SharedTimeline> timeline,
                           std::string scene_name,
                           bool new_pgm_is_slot_a,
                           int64_t midpoint_delay_ms,
                           int64_t total_delay_ms);

    int64_t margin_ms_ = 200;
    std::string transition_node_name_ = "mixer_transition";
    std::string transition_edge_name_ = "trans_out";

public:
    MixerOrchestrator(std::shared_ptr<NodeManager> nodes,
                      std::shared_ptr<MixerState> state,
                      std::shared_ptr<SharedTimeline> timeline);

    void defineSource(const std::string& name, const std::string& otm_node, int input_index,
                      const std::string& cs_node_a, const std::string& cs_node_b);
    void defineScene(const std::string& name, const SceneDefinition& def);

    void cut(const std::string& scene_name);
    void fade(const std::string& scene_name, double duration_sec);
    void wipe(const std::string& scene_name, const std::string& wipe_file, double duration_sec);

    Parameters status() const;
};
