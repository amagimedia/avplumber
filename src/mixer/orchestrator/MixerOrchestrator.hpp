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


class MixerOrchestrator {
    std::shared_ptr<MixerGraph> nodes_;
    std::shared_ptr<MixerState> state_;

    void setNodeObject(const std::string& node_name, const std::string& key, const Parameters& value);
    void publishRuntimeObject(const std::string& node_name, const std::string& key,
                              const Parameters& value);

    void setNodeParam(const std::string& node_name, const std::string& param, const std::string& value);
    void startGroup(const std::string& group_name);
    void stopGroup(const std::string& group_name);

    // Flush all wipe pipeline edges listed in state_->wipe_flush_edges.
    // Skips edges whose configured consumer is still working; readerwriterqueue
    // clear() is a consumer-side operation and must not race that node.
    void flushWipeEdges();
    /// Cached wipes: wake the resident wipe compositor and start the clip replay, or with
    /// `scrub` the clip at the frame that schedule shows. Returns the player's status, whose
    /// start_tick is the output tick of the clip's first frame: the compositor's output is
    /// this take's from that tick on. Caller holds state_->mutex.
    Parameters armWipeChain(const std::string& wipe_file, std::optional<RampNs> scrub);
    /// Decoding per take: point the stopped wipe reader at `wipe_file`, flush what the previous
    /// clip left in the chain's edges and start the wipe group. Caller holds state_->mutex.
    void startWipeDecode(const std::string& wipe_file);
    /// Take the wipe chain off duty: park the resident chain (cached wipes), or stop
    /// the per-take decode group. Caller holds state_->mutex.
    void retireWipeChain();

    void loadSceneIntoSlot(bool is_slot_a, const std::string& scene_name, bool warm_cut = false);
    bool canPrewarmScene(const SceneDefinition& scene) const;
    void applySceneControls(const SceneDefinition& scene);

    /// Publish both slots' source routes, preserving subscriptions and prewarmed inputs.
    void publishSourceRoutes(const SceneDefinition* scene_a, const SceneDefinition* scene_b);

    /// Caller holds state_->mutex. Points the source_switcher at slot A or B, the only setting
    /// visible at the output, before applyPostTransitionRouting flips the rest: the window
    /// between the two would show only if the new direct path were not producing frames yet,
    /// and at every caller it is.
    void switchProgramSelector(bool new_pgm_is_slot_a);
    /// Restore steady-state routing after a cut or crossfade has concluded, once
    /// switchProgramSelector has switched the selector: the per-source OTM masks and the
    /// post-otm/compositor flips, for `scene` as the program slot draws it. Caller must hold
    /// state_->mutex. `picture_changed` false (the on-air picture stays the same) skips
    /// the encoder keyframe request.
    /// `pts_ns`: media time (ns, same clock as realtime set_pts / compositor frames) at which
    /// the new program picture applies; 0 means apply as soon as the router can.
    void applyPostTransitionRouting(bool new_pgm_is_slot_a, const SceneDefinition& scene,
                                    bool picture_changed = true, int64_t pts_ns = 0);

    /// Notify audio_router_node that program sources have committed (transition_ended).
    void emitTransitionEnded(const SceneDefinition& scene, int64_t pts_ns);
    /// Notify audio_router_node of fade/dip/wipe progress (transition_progress from/to/t/pts).
    void emitTransitionProgress(const char* kind, double t, int64_t pts_ns);

    void ensureIdle() const;
    /// Why a transition being prepared or running is dropped. `Replaced`: by the take that
    /// drops it, whose own switch publishes the next preview change, so the preview shown stays
    /// as it is meanwhile (a multiview's PVW tile does not blank between takes under cut spam).
    /// `Dropped`: with nothing to follow (`mixer.interrupt`), which clears the preview, as a
    /// failed take does (abortTransition).
    enum class Interruption : std::uint8_t { Dropped, Replaced };
    void interruptTransition(Interruption why);
    /// pts (ns) of the newest frame the selector has emitted, 0 before its first.
    int64_t selectorOutputNs() const;
    /// Two ticks past the newest program frame: where a change of the mix's progress starts, so
    /// frames the compositor drew meanwhile keep the progress they were drawn with.
    int64_t mixChangeNs() const;
    /// Caller holds state_->mutex. Ends a transition: program on `new_pgm_scene`, the preview
    /// swapped or cleared (MixerState::completeTransition) from the program frame at
    /// `effective_ns` (0: now).
    void finishTransition(bool new_pgm_is_slot_a, std::string new_pgm_scene, int64_t effective_ns);
    // Caller holds state_->mutex; restores live program after failed preparation.
    // Requests a keyframe unless the dropped transition was a cut that had not flipped.
    void restoreProgramRouting(MixerState::TransitionMode dropped);
    /// Caller holds state_->mutex: program routing restored, the wipe chain parked, nothing of
    /// the dropped transition left in the state. Cleanup failures are logged, never thrown.
    void dropTransition(MixerState::TransitionMode dropped);
    void abortTransition(uint64_t generation) noexcept;
    void beginTake(const std::string& scene_name, MixerState::TransitionMode mode);
    /// Caller holds state_->mutex. A mix to `scene_name` being prepared, as AUTO and the lever start one.
    void beginCrossfade(const std::string& scene_name, double duration_sec, FadeCurve curve,
                        std::optional<std::array<uint8_t, 3>> dip, int64_t received_ns);
    /// Caller holds state_->mutex. A wipe of `wipe_file` to `scene_name` armed, on air with the
    /// clip's first overlay frame (selectWipe); `duration_sec` unset: the lever's wipe, its cached
    /// clip scrubbed by the lever.
    void armWipe(const std::string& scene_name, const std::string& wipe_file, std::optional<double> duration_sec);
    /// A take pressed while a transition is on air acts on that transition, as a vision
    /// switcher's buttons do (Grass Valley Kayenne): CUT completes it at the next frame
    /// boundary; AUTO for its scene runs the lever's take on, or pauses or resumes a crossfade
    /// (a wipe ignores it); another scene first replaces the preset picture under the running
    /// transition. Returns false, doing nothing, when no transition is on air. Caller holds
    /// state_->mutex.
    bool actOnTransitionOnAir(const std::string& scene_name, MixerState::TransitionMode pressed);
    /// Load `scene_name` into the preset slot under the transition on air, like a preset-bus
    /// selection: the incoming picture changes as soon as the slot renders it.
    void replacePreset(const std::string& scene_name);
    void toggleTakePause();
    /// Caller holds state_->mutex. The lever at `position`, moving there over `span_ms` (lever()),
    /// for the take state_->lever_take describes.
    void leverTo(double position, std::optional<double> span_ms);
    /// Caller holds state_->mutex. The lever's take ends back at its start, Program unchanged.
    void cancelLeverTake();
    /// Caller holds state_->mutex. The lever's end that lands on the program frame at `pts_ns`, or
    /// None. A Cancel ends the take here (cancelLeverTake, then resumeLever); a Complete is the
    /// caller's to complete, then resumeLever.
    LeverEnd landLeverEnd(int64_t pts_ns);
    /// Caller holds state_->mutex. The take ends on the program frame at `pts_ns`: where the lever's
    /// end lands (landLeverEnd), or where it `ran_out` without the lever, a completion only once
    /// `ready`; resumeLever after the lever's. Returns whether it ended.
    bool endTakeAt(int64_t pts_ns, bool ready, bool ran_out);
    /// Caller holds state_->mutex. After the lever's take ended at one of its ends: where the lever
    /// moved while that end landed starts the next take.
    void resumeLever();
    /// Caller holds state_->mutex. AUTO takes the lever's take over from where the lever has it.
    void runOnFromLever();
    /// Caller holds state_->mutex. The lever's take ramps to `progress` over `span_ms` (unset: the
    /// time since the previous position) and holds there, as a pause holds.
    void moveLever(double progress, std::optional<double> span_ms);
    /// Where a change of the take's schedule starts: mixChangeNs(), for a scrubbed wipe clip never
    /// before a tick the player may have pushed already.
    int64_t takeScheduleFromNs() const;
    /// Caller holds state_->mutex. Sends take_request to what follows it: the mix's alpha or the
    /// lever's wipe clip, then the keys' clock.
    void applyTakeSchedule();
    /// Re-sends only the mix's progress: its mode and colour stay as the take set them.
    void sendAlpha(const FadeRequest& request);
    /// Caller holds state_->mutex. The keys' clock (TakeClock): a take on air from `start_ns`
    /// over `duration_ns` (0: they switch at start_ns; a set `switch_at`: where its progress
    /// reaches that); the mix's clamps after a pause, resume or lever move; the take's end, if
    /// the clock is still running.
    void startTakeClock(int64_t start_ns, int64_t duration_ns, FadeCurve curve, double switch_at = -1);
    void clampTakeClock();
    void endTakeClock(TakeClock::End end, int64_t end_ns);
    void prepareScene(const std::string& scene_name);
    bool destinationReady(const av::VideoFrame* frame) const;
    void completeTake(int64_t pts_ns);
    int selectProgram(const std::vector<const av::VideoFrame*>& frames, int64_t last_ns);
    int selectWipe(const std::vector<const av::VideoFrame*>& frames, int active, int64_t last_ns);

public:
    /// Drop an armed or running transition and keep the current program picture (`mixer.interrupt`).
    void interrupt() {
        std::lock_guard<std::mutex> lock(state_->mutex);
        state_->take_lever_end = LeverEnd::None;   // dropped, not ended by the lever
        interruptTransition(Interruption::Dropped);
    }
    MixerOrchestrator(std::shared_ptr<NodeManager> nodes, std::shared_ptr<MixerState> state);
    MixerOrchestrator(std::shared_ptr<MixerGraph> nodes, std::shared_ptr<MixerState> state);
    // Called with state.mutex held, on the selector's consumer thread before
    // publishing a frame. Returns exactly one selected input for this tick.
    int selectFrame(bool wipe, const std::vector<const av::VideoFrame*>& frames,
                    int active, int64_t last_ns);


    void defineSource(const std::string& name, const std::string& otm_node, int input_index);
    void defineRoutedSource(const std::string& name, const std::string& router_node,
                            int input_index,
                            const std::string& route_output_label_a,
                            const std::string& route_output_label_b);
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
    /// The lever arm (T-bar) at `position` in [0, 1] (mixer.lever). From its rest it starts a
    /// manual mix to Preview with AUTO's `duration_sec`, `curve` and `dip`, or with a `wipe_file`
    /// a wipe of that cached clip, whose frame follows the lever; held at the lever's progress,
    /// moving there over `span_ms` when set. The far end completes it, the start cancels it with
    /// Program unchanged, each once the picture got there; AUTO runs it on from there (a wipe at
    /// its clip's rate). An AUTO, cut or other take on air ignores it; see Lever for pickup.
    void lever(double position, double duration_sec, FadeCurve curve, std::optional<std::array<uint8_t, 3>> dip,
               const std::string& wipe_file, std::optional<double> span_ms);
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
