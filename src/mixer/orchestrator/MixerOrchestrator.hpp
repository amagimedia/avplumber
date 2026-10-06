#pragma once
#include "../primitives/MixerState.hpp"
#include "../MixerGraph.hpp"
#include "../../graph_mgmt.hpp"
#include "../../instance_shared.hpp"
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace avp::mixer {

struct OutputSnapshot;

class MixerOrchestrator {
    std::shared_ptr<MixerGraph> nodes_;
    std::shared_ptr<MixerState> state_;

    void setNodeObject(const std::string& node_name, const std::string& key, const Parameters& value);
    void publishRuntimeObject(const std::string& node_name, const std::string& key,
                              const Parameters& value);

    void publishCameraOtmOutputs(const std::string& otm_name, uint32_t mask);
    void setNodeParam(const std::string& node_name, const std::string& param, const std::string& value);
    void autoRestartNode(const std::string& node_name);
    void startGroup(const std::string& group_name);
    void stopGroup(const std::string& group_name);

    // Flush all wipe pipeline edges listed in state_->wipe_flush_edges.
    // Skips edges whose configured consumer is still working; readerwriterqueue
    // clear() is a consumer-side operation and must not race that node.
    void flushWipeEdges();
    /// Cached wipes: wake the resident wipe compositor and start the clip replay.
    /// Returns the output tick of the clip's first frame: the compositor's output is
    /// this take's from that tick on. Caller holds state_->mutex.
    int64_t armWipeChain(const std::string& wipe_file);
    /// Decoding per take: point the stopped wipe reader at `wipe_file`, flush what the previous
    /// clip left in the chain's edges and start the wipe group. Caller holds state_->mutex.
    void startWipeDecode(const std::string& wipe_file);
    /// Take the wipe chain off duty: park the resident chain (cached wipes), or stop
    /// the per-take decode group. Caller holds state_->mutex.
    void retireWipeChain();
    void flushSlotEdges(bool is_slot_a);

    void loadSceneIntoSlot(bool is_slot_a, const std::string& scene_name, bool warm_cut = false);
    bool canPrewarmScene(const SceneDefinition& scene) const;
    void applySceneControls(const SceneDefinition& scene);

    /// Rewrite every camera `one_to_many` bitmask for one slot bit from scene + active_inputs.
    void rewriteCameraOutputsForSlot(uint32_t slot_bit, const SceneDefinition& scene);
    void applyRoutedSceneRoutesForSlot(bool is_slot_a, const SceneDefinition& scene);
    void publishRoutedRoutesForProgramOnly(bool pgm_is_slot_a, const SceneDefinition& scene);

    /// Caller holds state_->mutex. Points the source_switcher at slot A or B, the only setting
    /// visible at the output, before applyPostTransitionRouting flips the rest: the window
    /// between the two would show only if the new direct path were not producing frames yet,
    /// and at every caller it is. Returns selectorOutputNs() read right after the switch, for
    /// finishSnapshot(); a finishing take publishes its preview between the two calls
    /// (MixerState::publishTakePreview), so the AUX followers start while the routing runs.
    int64_t switchProgramSelector(bool new_pgm_is_slot_a);
    /// Restore steady-state routing after a cut or crossfade has concluded, once
    /// switchProgramSelector has switched the selector: the per-source OTM masks and the
    /// post-otm/compositor flips. Caller must hold state_->mutex.
    /// `picture_changed` false (the on-air picture stays the same) skips
    /// the encoder keyframe request. Does nothing for an unknown scene.
    void applyPostTransitionRouting(bool new_pgm_is_slot_a, const std::string& new_pgm_scene,
                                    bool picture_changed = true);

    void ensureIdle() const;
    /// Why a transition being prepared or running is dropped. `Replaced`: by the take that
    /// drops it, whose own switch publishes the next preview change, so the preview shown stays
    /// as it is meanwhile (a multiview's PVW tile does not blank between takes under cut spam).
    /// `Dropped`: with nothing to follow (`mixer.interrupt`), which clears the preview, as a
    /// failed take does (abortTransition).
    enum class Interruption : std::uint8_t { Dropped, Replaced };
    void interruptTransition(Interruption why);
    std::shared_ptr<OutputSnapshot> outputSnapshot() const;
    /// Stops the slot substitution and releases a held output at the first selected frame
    /// newer than `emitted`, the selector's newest output read after the selector was switched
    /// (switchProgramSelector's return, or read now).
    void finishSnapshot(int64_t emitted);
    void finishSnapshot() { finishSnapshot(selectorOutputNs()); }
    /// pts (ns) of the newest frame the selector has emitted, 0 before its first.
    int64_t selectorOutputNs() const;
    /// Caller holds state_->mutex. Ends a transition: program on `new_pgm_scene`, the preview
    /// swapped or cleared (MixerState::completeTransition) from the program frame at
    /// `effective_ns` (0: now).
    void finishTransition(bool new_pgm_is_slot_a, std::string new_pgm_scene, int64_t effective_ns);
    // Caller holds state_->mutex; restores live program after failed preparation.
    // Requests a keyframe unless the dropped transition was a cut that had not flipped
    // and no frozen picture is on air: the caller then finishes the snapshot, which
    // replaces a frozen picture with the live program.
    void restoreProgramRouting(MixerState::TransitionMode dropped);
    void abortTransition(uint64_t generation) noexcept;
    void beginTake(const std::string& scene_name, MixerState::TransitionMode mode);
    void prepareScene(const std::string& scene_name);
    bool destinationReady(const av::VideoFrame* frame) const;
    void completeTake(int64_t pts_ns);
    int selectProgram(const std::vector<const av::VideoFrame*>& frames, int active, int64_t last_ns);
    int selectWipe(const std::vector<const av::VideoFrame*>& frames, int active, int64_t last_ns);

public:
    /// Drop an armed or running transition and keep the current program picture (`mixer.interrupt`).
    void interrupt() {
        std::lock_guard<std::mutex> lock(state_->mutex);
        interruptTransition(Interruption::Dropped);
    }
    MixerOrchestrator(std::shared_ptr<NodeManager> nodes, std::shared_ptr<MixerState> state);
    MixerOrchestrator(std::shared_ptr<MixerGraph> nodes, std::shared_ptr<MixerState> state);
    // Called with state.mutex held, on the selector's consumer thread before
    // publishing a frame. Returns exactly one selected input for this tick.
    int selectFrame(bool wipe, const std::vector<const av::VideoFrame*>& frames,
                    int active, int64_t last_ns);


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
    void cut(const std::string& scene_name,
             avp::mixer::CutLatency::Clock::time_point received = avp::mixer::CutLatency::Clock::now());
    void enableCutMeasurements(const std::string& mixer_name, const std::string& encoder_name);
    void prewarmCuts(const std::vector<std::string>& scenes);
    /// A set `dip` (opaque SDR RGB) fades through that colour instead of mixing.
    void fade(const std::string& scene_name, double duration_sec,
              FadeCurve curve = FadeCurve::Linear, std::optional<std::array<uint8_t, 3>> dip = std::nullopt,
              avp::mixer::CutLatency::Clock::time_point received = avp::mixer::CutLatency::Clock::now());
    void wipe(const std::string& scene_name, const std::string& wipe_file, double duration_sec);
    /// Run the wipe subgraph once on *wipe_file* with the output kept on the
    /// direct branch, so file open, decoder and GPU filter initialisation (PTX
    /// compilation included) happen before the first real wipe. Blocks until
    /// the overlay produced a frame or *timeout_ms* passed.
    void warmupWipe(const std::string& wipe_file, int64_t timeout_ms);

    /// Returns the names of all registered scenes, sorted alphabetically.
    std::vector<std::string> sceneNames() const;
    Parameters status() const;
};

}  // namespace avp::mixer
