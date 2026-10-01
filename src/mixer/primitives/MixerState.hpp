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
    /// The preview as shown to the operator: what `mixer.preview` armed, or after a take the
    /// scene that just left program (the swap). Followers (mixer_pvw_follow) draw it; it says
    /// nothing about the PVW slot's contents, see pvw_slot_scene.
    std::string pvw_scene_name;
    /// The scene loaded in the PVW slot, "" while the slot is cold (after every completed or
    /// dropped transition: the old program slot's sources are routed away). A take reuses the
    /// slot only for this scene; anything else reloads it, so a swapped preview costs no GPU
    /// time until it is taken.
    std::string pvw_slot_scene;
    /// The preview change feed for the followers (mixer_pvw_follow): the newest change, under
    /// `preview_mutex`, a lock of its own that is never held while any other is taken, so a take
    /// publishes right after switching the selector and a follower wakes on `preview_changed`
    /// while the take's routing still holds `mutex`. Writers hold `mutex` too (`mutex`, then
    /// `preview_mutex`); a follower takes `preview_mutex` alone.
    struct PreviewChange {
        uint64_t revision = 0;
        std::string pvw;           // the preview shown, "" for none: pvw_scene_name's copy
        std::string pgm;           // the program shown beside it (a take: the program it moves to)
        int64_t effective_ns = 0;  // pts (ns, the compositors' monotonic clock) of the first program
                                   // frame that shows the new program; 0: the change is immediate
        int64_t published_ns = 0;  // when it was published (monotonic ns)
        int64_t received_ns = 0;   // when the take that ended in it was received (monotonic ns;
                                   // 0: not a take), for the followers' latency from the command
        std::string kind;          // the take: "cut", "fade", "wipe"; "" for a preview or a clear
    };
    std::mutex preview_mutex;
    PreviewChange preview;
    std::condition_variable preview_changed;
    std::atomic<int> preview_followers{0};
    /// The last timed preview change every follower (keyed by its node name) applied:
    /// pvw_latency_ms, pgm_latency_ms, pvw_minus_pgm_ms and the rest of the follower's status,
    /// for `mixer.status` `pvw_latency`. Under `preview_mutex`.
    std::map<std::string, Parameters> preview_follow_samples;
    /// Receipt (monotonic ns) of the take command being prepared or running, 0 without one;
    /// publishTakePreview hands it to the followers.
    int64_t take_received_ns = 0;
    /// The running take published its preview already (publishTakePreview), so
    /// completeTransition does not publish a second revision.
    bool take_preview_published = false;

    /// Caller holds `mutex`. Shows `scene` ("" clears) as the preview from the program frame at
    /// `effective_ns` (0: now) beside the program `pgm` (the current one when empty), received
    /// as a take of `kind` at `received_ns` (0: not one), and wakes every follower.
    void publishPreview(const std::string& scene, int64_t effective_ns, int64_t received_ns = 0,
                        const std::string& pgm = "", const std::string& kind = "") {
        pvw_scene_name = scene;
        std::lock_guard<std::mutex> lock(preview_mutex);
        preview = {preview.revision + 1, scene, pgm.empty() ? pgm_scene_name : pgm, effective_ns,
                   monotonicNs(), received_ns, kind};
        preview_changed.notify_all();
    }

    /// Caller holds `mutex`. The running take has switched the program to `new_pgm` from the
    /// frame at `effective_ns` (0: not timed): previews the scene that leaves program (OBS's
    /// "Swap Preview/Program Scenes After Transitioning", always on) or nothing when the take
    /// keeps the program scene, timed for the followers with the take's receipt and kind. Once
    /// per take, right after the selector switch and before its routing,
    /// so the followers start meanwhile; completeTransition publishes for a take that did not.
    void publishTakePreview(const std::string& new_pgm, int64_t effective_ns) {
        publishPreview(pgm_scene_name != new_pgm ? pgm_scene_name : "", effective_ns,
                       take_received_ns, new_pgm, takeKind());
        take_received_ns = 0;
        take_preview_published = true;
    }

    /// Caller holds `mutex`. Program moves to `new_pgm` on slot A or B from the frame at
    /// `effective_ns`; the other slot is cold, and the preview is what publishTakePreview
    /// published (here, for a take that did not). Ends the transition.
    void completeTransition(bool new_pgm_is_slot_a, std::string new_pgm, int64_t effective_ns) {
        if (!take_preview_published) publishTakePreview(new_pgm, effective_ns);
        take_preview_published = false;
        pgm_is_slot_a = new_pgm_is_slot_a;
        pgm_scene_name = std::move(new_pgm);
        pvw_slot_scene.clear();
        transition_mode = TransitionMode::Idle;
    }

    /// Caller holds `mutex`. A dropped or failed take: its preview is cleared, at once.
    void clearTakePreview() {
        take_received_ns = 0;
        take_preview_published = false;
        publishPreview("", 0);
    }

    int fps_num = 30, fps_den = 1;
    int64_t switch_margin_ms = 100;
    /// The compositors' canvas transfer (mixer.init "color"); a dip colour is converted for it.
    AVColorTransferCharacteristic canvas_transfer = AVCOL_TRC_BT709;

    enum class TransitionMode { Idle, Cut, Crossfade, Wipe };
    std::atomic<TransitionMode> transition_mode{TransitionMode::Idle};
    /// The running take's kind as the followers report it; "" while idle.
    const char* takeKind() const {
        switch (transition_mode.load()) {
            case TransitionMode::Cut: return "cut";
            case TransitionMode::Crossfade: return "fade";
            case TransitionMode::Wipe: return "wipe";
            default: return "";
        }
    }
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

    // Pre-created wipe subgraph. Decoding per take: the group is started at wipe begin and
    // stopped at wipe end. With the clip cache it runs for the life of the graph and a take
    // only arms it (see wipeChainStaysRunning()).
    std::string wipe_group_name;       // "mixer_wipe"
    std::string wipe_input_node_name;  // "wipe_input" (input_rec whose url is set per wipe)
                                       // or the clip_cache player armed with "play"
    /// ClipCache store holding decoded wipe clips ("clips"), reported by mixer.status.
    /// Empty when wipes decode per take.
    std::string wipe_cache_store;
    /// The wipe compositor (cuda_rect_overlay), parked with active_inputs=0 between cached
    /// wipes and armed per take. Required when wipe_cache_store is set.
    std::string wipe_overlay_name;
    /// Cached wipes keep the player group running: no node is created, started or stopped
    /// for a take, so its CUDA allocations and threads never churn under the program.
    bool wipeChainStaysRunning() const { return !wipe_cache_store.empty(); }
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
