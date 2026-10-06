#include "internal.hpp"

namespace avp::mixer {

void MixerOrchestrator::wipe(const std::string& scene_name, const std::string& wipe_file, double duration_sec) {
    if (!std::isfinite(duration_sec) || duration_sec <= 0) throw Error("mixer: invalid wipe duration");
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!state_->scenes.count(scene_name)) throw Error("mixer: unknown scene: " + scene_name);
    if (state_->wipe_otm_name.empty() || state_->wipe_selector_name.empty() ||
        state_->wipe_group_name.empty() || state_->wipe_input_node_name.empty())
        throw Error("mixer: wipe requires the wipe subgraph");
    try {
        beginTake(scene_name, MixerState::TransitionMode::Wipe);
        state_->take_received_ns = monotonicNs();
        state_->take_duration_ns = int64_t(duration_sec * 1000000000);
        state_->take_deadline_ns = monotonicNs() + kWipeReadyTimeoutMs * 1000000;
        state_->take_phase = MixerState::TakePhase::WipeReady;
        resetInputIf(nodes_, state_->wipe_base_fps_name);
        if (state_->wipeChainStaysRunning()) {
            const auto tick = armWipeChain(wipe_file);
            state_->wipe_first_ns = av::Timestamp(tick, {state_->fps_den, state_->fps_num}).timestamp({1, 1000000000});
        } else {
            nodes_->group(state_->wipe_group_name)->stopNodesAndWait();
            const auto previous = edgeLastTsIfExists(nodes_, edgeNameAt(nodes_, state_->wipe_selector_name, "src", 1));
            state_->wipe_first_ns = previous.isValid() ? previous.timestamp({1, 1000000000}) + 1 : 0;
            startWipeDecode(wipe_file);
        }
        setNodeObject(state_->wipe_otm_name, "outputs", Parameters(3u));
        setNodeObject(state_->wipe_selector_name, "active", Parameters(0));
    } catch (...) {
        abortTransition(state_->transition_generation);
        throw;
    }
}

int MixerOrchestrator::selectWipe(const std::vector<const av::VideoFrame*>& frames, int active, int64_t last_ns) {
    if (state_->transition_mode != MixerState::TransitionMode::Wipe) return 0;
    const auto pts = [&](int input) {
        const auto* frame = frames.at(input);
        return frame && frame->isValid() && frame->pts().isValid()
            ? frame->pts().timestamp({1, 1000000000}) : int64_t(0);
    };
    if (state_->take_phase == MixerState::TakePhase::WipeReady) {
        const auto overlay = pts(1);
        if (state_->destination_ready && overlay > 0 && overlay >= state_->wipe_first_ns) {
            state_->take_start_ns = overlay;
            state_->take_phase = MixerState::TakePhase::Wipe;
            return 1;
        }
        return 0;
    }
    const auto direct = pts(0);
    if (state_->wipe_switched && direct > last_ns &&
        direct >= state_->take_start_ns + state_->take_duration_ns) {
        completeTake(direct);
        setNodeObject(state_->wipe_otm_name, "outputs", Parameters(1u));
        retireWipeChain();
        return 0;
    }
    return active;
}

void MixerOrchestrator::warmupWipe(const std::string& wipe_file, int64_t timeout_ms) {
    std::string overlay_edge_name;
    av::Timestamp overlay_initial_ts = NOTS;
    uint64_t generation;
    const int64_t t0 = wallclock.pts();
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->wipe_otm_name.empty() || state_->wipe_selector_name.empty() ||
            state_->wipe_group_name.empty() || state_->wipe_input_node_name.empty())
            throw Error("mixer: wipe warm-up requires the wipe subgraph (see mixer.init)");
        if (state_->wipeChainStaysRunning())
            throw Error("mixer: wipe warm-up is for decoding per take; with the clip cache the wipe "
                        "chain stays running and the clips are preloaded instead");
        if (state_->transition_mode.load() != MixerState::TransitionMode::Idle)
            throw Error("mixer: cannot warm up the wipe during a transition");
        generation = state_->transition_generation.load();
        overlay_edge_name = edgeNameAt(nodes_, state_->wipe_selector_name, "src", 1);
        overlay_initial_ts = edgeLastTsIfExists(nodes_, overlay_edge_name);
        nodes_->group(state_->wipe_group_name)->stopNodesAndWait();
        resetInputIf(nodes_, state_->wipe_base_fps_name);
        startWipeDecode(wipe_file);
        // Feed the overlay's program input as a real wipe would; the selector
        // stays on the direct branch so nothing of this reaches the output.
        setNodeObject(state_->wipe_otm_name, "outputs", Parameters(3u));
    }
    WipeReadyResult ready = waitForWipeOverlayReady(nodes_, overlay_edge_name, overlay_initial_ts, 0,
                                                    state_, generation, timeout_ms);
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        setNodeObject(state_->wipe_otm_name, "outputs", Parameters(1u));
        stopGroup(state_->wipe_group_name);
        flushWipeEdges();
    }
    logstream << "mixer wipe warm-up: file=" << wipe_file << (ready.ready ? " ready" : " NOT ready")
              << " after " << (wallclock.pts() - t0) << "ms (overlay waited " << ready.waited_ms << "ms)";
    if (!ready.ready)
        throw Error("mixer: wipe warm-up did not produce an overlay frame within " + std::to_string(timeout_ms) + "ms");
}


} // namespace avp::mixer
