#pragma once
#include "../../instance_shared.hpp"
#include "../../util.hpp"
#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <atomic>
#include "CutLatencyProbe.hpp"
#include "MonotonicClock.hpp"
#include "source_mask.hpp"
#include "../transition_control.hpp"

extern "C" {
#include <libavutil/pixfmt.h>
}

namespace avp::mixer {

struct SourceLayout {
    std::string crop_scale_graph; // e.g., "crop=1920:1080:0:0,scale_cuda=640:360"
    /// Layer fields for cuda_rect_overlay (dst_x, dst_y, …) — not including `graph`.
    Parameters layer;
};

struct SceneControl {
    std::string node_name;
    std::string key;
    Parameters value;
};

struct SceneDefinition {
    std::string name;
    /// Logical source name -> crop/scale graph + per-source layer (see mixer.source input_index).
    std::unordered_map<std::string, SourceLayout> sources;
    /// Logical routed source name -> router input index selected by this scene.
    std::unordered_map<std::string, int> routes;
    std::vector<SceneControl> controls;
    int width = 1920;
    int height = 1080;

};

struct MixerState : public InstanceShared<MixerState> {
    std::mutex mutex;
    TransitionControl transition_control = nullptr;
    std::string transition_node_name;

    struct SourceInfo {
        std::string otm_node_name;          // "otm_cam1"
        int input_index;                    // index within compositor src array (0..SourceMask::kBits-1)
        std::string cs_node_a, cs_node_b;   // "cs_cam1_a", "cs_cam1_b"
        bool routed = false;
        std::string router_node_name;
        std::string route_output_label_a;
        std::string route_output_label_b;
        int route_output_a = -1;
        int route_output_b = -1;
    };
    std::unordered_map<std::string, SourceInfo> sources;

    std::unordered_map<std::string, SceneDefinition> scenes;
    bool scene_definitions_frozen = false;
    std::unordered_set<std::string> prewarm_cut_scenes;
    SourceMask prewarm_source_mask;
    uint32_t sourceOutputMask(const SourceInfo& source, uint32_t requested) const {
        return prewarm_source_mask.test(source.input_index) ? requested | 3u : requested;
    }
    std::unordered_map<std::string, int> router_output_counts;
    std::unordered_map<std::string, std::vector<int>> router_routes;

    bool pgm_is_slot_a = true;
    std::string pgm_scene_name;
    /// The preview as shown to the operator: what `mixer.preview` armed, or with `swap_preview` the
    /// scene that just left program. Followers (mixer_pvw_follow) draw it; it says nothing about
    /// the PVW slot's contents, see pvw_slot_scene.
    std::string pvw_scene_name;
    /// The scene loaded in the PVW slot, "" while the slot is cold (after every completed or
    /// dropped transition: the old program slot's sources are routed away). A take reuses the
    /// slot only for this scene; anything else reloads it, so a swapped preview costs no GPU
    /// time until it is taken.
    std::string pvw_slot_scene;
    /// OBS's "Swap Preview/Program Scenes After Transitioning": a completed take previews the
    /// scene that left program. Off, a take clears the preview.
    bool swap_preview = true;
    /// Preview change feed for the followers, written under `mutex`, waited on with
    /// `preview_changed`: the revision, the pts (ns, on the monotonic clock the compositors stamp
    /// with) of the first program frame that shows the new program (0: the change is immediate),
    /// when it was published (monotonic ns), for the followers' latency metric, and when the
    /// take that ended in it was received (monotonic ns; 0: not a take), for their latency
    /// from the operator's command.
    uint64_t pvw_revision = 0;
    int64_t pvw_effective_ns = 0;
    int64_t pvw_published_ns = 0;
    int64_t pvw_received_ns = 0;
    std::condition_variable preview_changed;
    std::atomic<int> preview_followers{0};
    /// Receipt (monotonic ns) of the take command being prepared or running, 0 without one;
    /// completeTransition hands it to the followers.
    int64_t take_received_ns = 0;
    /// The last timed preview change every follower (mixer_pvw_follow, keyed by its node name)
    /// applied: pvw_latency_ms, pgm_latency_ms, pvw_minus_pgm_ms and the rest of the follower's
    /// status, for `mixer.status` `pvw_latency`. Written by the follower under `mutex`.
    std::map<std::string, Parameters> preview_follow_samples;

    /// Caller holds `mutex`. Shows `scene` ("" clears) as the preview from the program frame at
    /// `effective_ns` (0: now), received as a take at `received_ns` (0: not one), and wakes
    /// every follower.
    void publishPreview(const std::string& scene, int64_t effective_ns, int64_t received_ns = 0) {
        pvw_scene_name = scene;
        pvw_effective_ns = effective_ns;
        pvw_published_ns = monotonicNs();
        pvw_received_ns = received_ns;
        ++pvw_revision;
        preview_changed.notify_all();
    }

    /// Caller holds `mutex`. Program moves to `new_pgm` on slot A or B from the frame at
    /// `effective_ns`; the other slot is cold, and the preview is the scene that left program
    /// (swap_preview, when it differs) or nothing. Ends the transition, whose command receipt
    /// (`take_received_ns`) goes to the followers with the change.
    void completeTransition(bool new_pgm_is_slot_a, std::string new_pgm, int64_t effective_ns) {
        const std::string old_pgm = std::move(pgm_scene_name);
        pgm_is_slot_a = new_pgm_is_slot_a;
        pgm_scene_name = std::move(new_pgm);
        pvw_slot_scene.clear();
        publishPreview(swap_preview && old_pgm != pgm_scene_name ? old_pgm : "", effective_ns, take_received_ns);
        take_received_ns = 0;
        transition_mode = TransitionMode::Idle;
    }

    int fps_num = 30, fps_den = 1;
    int64_t switch_margin_ms = 100;
    /// The compositors' canvas transfer (mixer.init "color"); a dip colour is converted for it.
    AVColorTransferCharacteristic canvas_transfer = AVCOL_TRC_BT709;

    enum class TransitionMode { Idle, Cut, Crossfade, Wipe };
    std::atomic<TransitionMode> transition_mode{TransitionMode::Idle};
    std::atomic<uint64_t> transition_generation{0};
    std::string transition_scene_name;

    struct SlotNodes {
        std::string compositor_name;   // "comp_a" / "comp_b"
        std::string norm_ts_name;      // "norm_a" / "norm_b"
        std::string post_otm_name;     // "otm_scene_a" / "otm_scene_b"
    };
    SlotNodes slot_a, slot_b;
    std::string source_switcher_name;  // "out_sel"
    /// Optional force_keyframe node triggered when a transition reaches the output,
    /// so a WebRTC receiver can decode the new picture immediately instead of
    /// waiting for the next periodic keyframe.
    std::string keyframe_node_name;
    std::shared_ptr<avp::mixer::CutLatencyProbe> cut_latency;
    std::string timeline_name;         // "mixer_tl"
    std::string hwaccel_name;          // "@gpu"

    // Static nodes for wipe output path (otm splits mixer_out, selector chooses direct vs overlay)
    std::string wipe_otm_name;         // "otm_final"
    std::string wipe_base_fps_name;    // "wipe_base_fps"
    std::string wipe_selector_name;    // "wipe_sel"

    // Pre-created wipe subgraph: group is started at wipe begin, stopped at wipe end
    std::string wipe_group_name;       // "mixer_wipe"
    std::string wipe_input_node_name;  // "wipe_input" (input_rec whose url is set per wipe)
    /// ClipCache store holding decoded wipe clips ("clips"), reported by mixer.status.
    /// Empty when wipes decode per take.
    std::string wipe_cache_store;
    /// Edge feeding the overlay's wipe input (e.g. "wipe_rt_fps_out"). Polled at
    /// wipe end to ensure the tail of the wipe has been consumed by the overlay
    /// before `wipe_selector` flips back to the direct path; otherwise the last
    /// ~pipeline-latency worth of wipe frames is cut off at the selector.
    std::string wipe_tail_edge;

    // Edges to flush before each wipe starts and after each wipe stops.
    // Prevents frames from a previous wipe run from bleeding into the next one.
    std::vector<std::string> wipe_flush_edges;

    // Optional post-mixer HTML/DMA overlay path.  The native mixer.overlay
    // command uses these static graph nodes to arm the hidden branch, wait for
    // a monotonic candidate frame, and then switch the final selector.
    std::string overlay_source_otm_name;  // e.g. "otm_html_overlay_src"
    std::string overlay_otm_name;         // e.g. "otm_html_overlay"
    std::string overlay_selector_name;    // e.g. "overlay_sel"
    int64_t overlay_ready_timeout_ms = 1000;
    int64_t overlay_ready_poll_ms = 5;
    bool overlay_enabled = false;
    std::atomic<uint64_t> overlay_generation{0};

    const SlotNodes& pgmSlot() const { return pgm_is_slot_a ? slot_a : slot_b; }
    const SlotNodes& pvwSlot() const { return pgm_is_slot_a ? slot_b : slot_a; }

    int pgmSourceSwitcherIndex() const { return pgm_is_slot_a ? 0 : 1; }
    int pvwSourceSwitcherIndex() const { return pgm_is_slot_a ? 1 : 0; }
    static constexpr int transSourceSwitcherIndex() { return 2; }

    uint32_t pgmOutputBit() const { return pgm_is_slot_a ? 1u : 2u; }
    uint32_t pvwOutputBit() const { return pgm_is_slot_a ? 2u : 1u; }

    SourceMask computeActiveInputsMask(const SceneDefinition& scene) const {
        SourceMask mask;
        for (const auto& [src_name, layout] : scene.sources) {
            auto it = sources.find(src_name);
            if (it != sources.end())
                mask.set(it->second.input_index);
        }
        return mask;
    }
};

}  // namespace avp::mixer
