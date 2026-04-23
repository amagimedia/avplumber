#pragma once
#include "instance_shared.hpp"
#include "util.hpp"
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <atomic>

struct SourceLayout {
    std::string crop_scale_graph; // e.g., "crop=1920:1080:0:0,scale_cuda=640:360"
    /// Layer fields for cuda_rect_overlay (dst_x, dst_y, …) — not including `graph`.
    Parameters layer;
};

struct SceneDefinition {
    std::string name;
    /// Logical source name -> crop/scale graph + per-source layer (see mixer.source input_index).
    std::unordered_map<std::string, SourceLayout> sources;
    int width = 1920;
    int height = 1080;

    std::unordered_set<std::string> sourceNames() const {
        std::unordered_set<std::string> r;
        for (const auto& [k, v] : sources)
            r.insert(k);
        return r;
    }
};

struct MixerState : public InstanceShared<MixerState> {
    std::mutex mutex;

    struct SourceInfo {
        std::string otm_node_name;          // "otm_cam1"
        int input_index;                    // index within compositor src array
        std::string cs_node_a, cs_node_b;   // "cs_cam1_a", "cs_cam1_b"
    };
    std::unordered_map<std::string, SourceInfo> sources;

    std::unordered_map<std::string, SceneDefinition> scenes;

    bool pgm_is_slot_a = true;
    std::string pgm_scene_name;
    std::string pvw_scene_name;

    int fps_num = 30, fps_den = 1;

    enum class TransitionMode { Idle, Cut, Crossfade, Wipe };
    std::atomic<TransitionMode> transition_mode{TransitionMode::Idle};

    struct SlotNodes {
        std::string compositor_name;   // "comp_a" / "comp_b"
        std::string norm_ts_name;      // "norm_a" / "norm_b"
        std::string post_otm_name;     // "otm_scene_a" / "otm_scene_b"
    };
    SlotNodes slot_a, slot_b;
    std::string source_switcher_name;  // "out_sel"
    std::string timeline_name;         // "mixer_tl"
    std::string hwaccel_name;          // "@gpu"

    // Static nodes for wipe output path (otm splits mixer_out, selector chooses direct vs overlay)
    std::string wipe_otm_name;         // "otm_final"
    std::string wipe_selector_name;    // "wipe_sel"

    // Pre-created wipe subgraph: group is started at wipe begin, stopped at wipe end
    std::string wipe_group_name;       // "mixer_wipe"
    std::string wipe_input_node_name;  // "wipe_input" (input_rec whose url is set per wipe)

    // Edges to flush before each wipe starts and after each wipe stops.
    // Prevents frames from a previous wipe run from bleeding into the next one.
    std::vector<std::string> wipe_flush_edges;

    const SlotNodes& pgmSlot() const { return pgm_is_slot_a ? slot_a : slot_b; }
    const SlotNodes& pvwSlot() const { return pgm_is_slot_a ? slot_b : slot_a; }

    int pgmSourceSwitcherIndex() const { return pgm_is_slot_a ? 0 : 1; }
    int pvwSourceSwitcherIndex() const { return pgm_is_slot_a ? 1 : 0; }
    static constexpr int transSourceSwitcherIndex() { return 2; }

    uint32_t pgmOutputBit() const { return pgm_is_slot_a ? 1u : 2u; }
    uint32_t pvwOutputBit() const { return pgm_is_slot_a ? 2u : 1u; }

    uint32_t computeActiveInputsMask(const SceneDefinition& scene) const {
        uint32_t mask = 0;
        for (const auto& [src_name, layout] : scene.sources) {
            auto it = sources.find(src_name);
            if (it != sources.end())
                mask |= (1u << it->second.input_index);
        }
        return mask;
    }
};
