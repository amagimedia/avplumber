#include "internal.hpp"
#include "../primitives/ClipScrub.hpp"

namespace avp::mixer {

void MixerOrchestrator::wipe(const std::string& scene_name, const std::string& wipe_file, double duration_sec) {
    if (!std::isfinite(duration_sec) || duration_sec <= 0) throw Error("mixer: invalid wipe duration");
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (!state_->scenes.count(scene_name)) throw Error("mixer: unknown scene: " + scene_name);
    if (!state_->wipeSubgraph()) throw Error("mixer: wipe requires the wipe subgraph");
    if (actOnTransitionOnAir(scene_name, MixerState::TransitionMode::Wipe)) return;
    armWipe(scene_name, wipe_file, duration_sec);
}

void MixerOrchestrator::armWipe(const std::string& scene_name, const std::string& wipe_file,
                                std::optional<double> duration_sec) {
    try {
        beginTake(scene_name, MixerState::TransitionMode::Wipe);
        state_->take_received_ns = monotonicNs();
        state_->take_duration_ns = int64_t(duration_sec.value_or(0) * 1000000000);
        state_->take_deadline_ns = monotonicNs() + kWipeReadyTimeoutMs * 1000000;
        state_->take_phase = MixerState::TakePhase::WipeReady;
        resetInputIf(nodes_, state_->wipe_base_fps_name);
        if (state_->wipeChainStaysRunning()) {
            // The clip follows the take's schedule, as a mix's alpha does: the lever's held at its
            // first frame until the lever moves it, AUTO's running from the clip's first tick, so
            // AUTO pressed again holds and resumes a wipe as it does a mix.
            const int64_t now_ms = monotonicNs() / 1000000;
            FadeRequest held{now_ms, 1.0, !state_->pgm_is_slot_a};
            held.time_floor_ms = held.time_ceiling_ms = now_ms;
            state_->scheduleTake(held, true);
            const Parameters player = armWipeChain(wipe_file, rampTimesNs(held));
            state_->wipe_first_ns = av::Timestamp(player.at("start_tick").get<int64_t>(),
                                                  {state_->fps_den, state_->fps_num}).timestamp({1, 1000000000});
            state_->wipe_scrub_delay_ns = player.at("scrub_delay_ns").get<int64_t>();
            if (!duration_sec) {
                // AUTO runs the lever's wipe on at the clip's own rate.
                state_->take_duration_ns = player.at("length_ns").get<int64_t>();
                state_->take_manual = true;
            } else {
                // The player pushes the first tick scrub_delay after it begins, long after this change.
                FadeRequest run{state_->wipe_first_ns / 1000000, *duration_sec, !state_->pgm_is_slot_a};
                run.time_floor_ms = run.start_ms;
                state_->scheduleTake(run, false);
                applyTakeSchedule();
            }
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
    // A cut pressed during the wipe keeps the overlay on air until the cut lands (completeTake).
    if (state_->transition_mode != MixerState::TransitionMode::Wipe) return state_->wipe_on_air ? active : 0;
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
            state_->wipe_on_air = true;
            if (state_->take_request) {
                // The keys going with the lever's wipe switch where its background does, both ways.
                // Its clip followed the lever below the switch point (leverTo) and goes on to the
                // lever's position from a tick the program selector has not decided yet.
                startTakeClock(overlay, state_->take_duration_ns, FadeCurve::Linear, kWipeSwitchAt);
                if (state_->take_manual && state_->take_lever > kLeverWipeArmedAt) moveLever(state_->take_lever, std::nullopt);
                else clampTakeClock();
            }
            return 1;
        }
        return 0;
    }
    // The wipe's progress for the status, as fade.cpp reports a mix's: at the newest direct frame;
    // the lever's (or AUTO running it on) on its schedule. That frame's tick may be out already: the
    // overlay's frame of a tick usually comes first (below).
    const auto direct = pts(0);
    if (direct >= last_ns && direct > 0 && (state_->take_request || state_->take_duration_ns > 0))
        state_->take_progress = state_->take_request ? rampAt(state_->take_schedule, direct)
            : std::clamp(double(direct - state_->take_start_ns) / double(state_->take_duration_ns), 0.0, 1.0);
    // The overlay's frame of a tick can reach this selector before the direct frame of that tick
    // on every tick (2026-10-15 on the L4 show: wipes completed 3 to 8 s late, when timing jitter
    // let a direct frame through first). The equal tick completes the take without being emitted.
    // The lever's end lands on the frame its ramp reaches, AUTO's where its schedule ends (never
    // while held), or at the end of the clip it decodes per take.
    const bool ran_out = !state_->take_manual && (state_->take_request
        ? rampAt(state_->take_schedule, direct) >= 1.0
        : direct >= state_->take_start_ns + state_->take_duration_ns);
    if (direct > 0 && direct >= last_ns &&
        endTakeAt(direct, state_->wipe_switched && state_->destination_ready, ran_out))
        return direct == last_ns ? -1 : 0;   // completeTake retired the overlay
    return active;
}

void MixerOrchestrator::warmupWipe(const std::string& wipe_file, int64_t timeout_ms) {
    std::string overlay_edge_name;
    av::Timestamp overlay_initial_ts = NOTS;
    uint64_t generation;
    const int64_t t0 = wallclock.pts();
    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (!state_->wipeSubgraph())
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
    WipeReadyResult ready = waitForWipeOverlayReady(nodes_, overlay_edge_name, overlay_initial_ts,
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
