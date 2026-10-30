// MixerOrchestrator core: construction, node/edge adapters, transition
// interruption and abort, and status. Scene loading and takes live in the
// sibling scene/cut/fade/wipe.cpp files.
#include "internal.hpp"
#include "../../nodes/clip_cache/ClipCache.hpp"

namespace avp::mixer {

MixerOrchestrator::MixerOrchestrator(std::shared_ptr<NodeManager> nodes, std::shared_ptr<MixerState> state)
    : MixerOrchestrator(std::make_shared<MixerGraph>(nodes), std::move(state)) {}

MixerOrchestrator::MixerOrchestrator(std::shared_ptr<MixerGraph> nodes, std::shared_ptr<MixerState> state)
    : nodes_(std::move(nodes)), state_(std::move(state)) {}

void MixerOrchestrator::setNodeObject(const std::string& node_name, const std::string& key, const Parameters& value) {
    try {
        auto node = nodes_->node(node_name)->node();
        auto control = std::dynamic_pointer_cast<IInputsObjects>(node);
        if (!control) throw Error("node has no runtime controls");
        control->setObject(key, value);
    } catch (const std::exception& e) {
        throw Error("mixer: set " + node_name + "." + key + " failed: " + e.what());
    }
}

void MixerOrchestrator::publishRuntimeObject(const std::string& node_name,
                                             const std::string& key,
                                             const Parameters& value) {
    if (!setNodeObjectIfCreated(nodes_, node_name, key, value)) {
        logstream << "mixer: queued " << node_name << "." << key
                  << " for node not created yet";
    }
}

void MixerOrchestrator::setNodeParam(const std::string& node_name, const std::string& param, const std::string& value) {
    auto node = nodes_->node(node_name);
    auto& params = node->parameters();
    params[param] = value;
}

void MixerOrchestrator::startGroup(const std::string& group_name) {
    nodes_->group(group_name)->startNodes();
}

void MixerOrchestrator::stopGroup(const std::string& group_name) {
    nodes_->group(group_name)->stopNodes();
}

void MixerOrchestrator::flushWipeEdges() {
    for (const auto& name : state_->wipe_flush_edges) {
        auto edge = nodes_->edges()->findAny(name);
        if (!edge)
            continue;

        int occupied = edge->occupied();
        if (occupied <= 0)
            continue;

        auto active_consumer = workingConsumerForEdge(nodes_, name);
        if (active_consumer) {
            logstream << "mixer: leaving active wipe edge " << name
                      << " unflushed (" << occupied << " queued, consumer="
                      << active_consumer->name() << ")";
            continue;
        }

        logstream << "mixer: flushing stale wipe edge " << name
                  << " (" << occupied << " queued)";
        edge->clear();
    }
}

Parameters MixerOrchestrator::armWipeChain(const std::string& wipe_file, std::optional<RampNs> scrub) {
    // The compositor applies these on its render thread, possibly one loop iteration
    // each, and activating an input without prewarm resets it, threshold included. So:
    // prewarm first, which keeps the activation from touching the reset; then the reset,
    // after which the compositor rejects everything stamped before now, which is all the
    // previous take could have left in its edges; then the clip, whose first frame is
    // stamped on the next tick, so nothing of this take is rejected. Program frames of
    // the current deadline are rejected too: the wipe starts one tick later than the
    // in-flight picture, never on a stale one.
    setNodeObject(state_->wipe_overlay_name, "prewarm_inputs", Parameters(3u));
    resetInputIf(nodes_, state_->wipe_overlay_name);
    setNodeObject(state_->wipe_overlay_name, "active_inputs", Parameters(3u));
    if (scrub) {
        Parameters arm = scrubSchedule(*scrub);
        arm["path"] = wipe_file;
        setNodeObject(state_->wipe_input_node_name, "scrub", arm);
    } else {
        setNodeObject(state_->wipe_input_node_name, "play", Parameters(wipe_file));
    }
    return nodes_->node(state_->wipe_input_node_name)->getObject("status");
}

void MixerOrchestrator::startWipeDecode(const std::string& wipe_file) {
    nodes_->node(state_->wipe_input_node_name)->stop(true);
    setNodeParam(state_->wipe_input_node_name, "url", wipe_file);
    flushWipeEdges();
    startGroup(state_->wipe_group_name);
}

void MixerOrchestrator::retireWipeChain() {
    if (!state_->wipeChainStaysRunning()) {
        stopGroup(state_->wipe_group_name);
        return;
    }
    // The clip first: no frame is pushed after its stop returns. Then the compositor:
    // no active input ends the composites, and no prewarm input parks its thread.
    setNodeObject(state_->wipe_input_node_name, "stop", Parameters(nullptr));
    setNodeObject(state_->wipe_overlay_name, "active_inputs", Parameters(0u));
    setNodeObject(state_->wipe_overlay_name, "prewarm_inputs", Parameters(0u));
}

void MixerOrchestrator::ensureIdle() const {
    auto mode = state_->transition_mode.load();
    if (mode != MixerState::TransitionMode::Idle)
        throw Error("mixer: transition already in progress");
}

void MixerOrchestrator::interruptTransition(Interruption why) {
    if (state_->cut_latency) state_->cut_latency->timing.cancel();
    if (state_->transition_mode == MixerState::TransitionMode::Idle) return;
    const auto previous_mode = state_->transition_mode.load();
    const bool was_on_air = state_->blend_on_air || state_->wipe_on_air;
    const auto generation = ++state_->transition_generation;
    TransitionGuard guard([&] { abortTransition(generation); });
    // Back to the program as it was before the transition, like a switcher's transition cut
    // back to its initial state. A take replaces only one that is not on air yet (takes on air
    // go to actOnTransitionOnAir), so for it nothing visible changes.
    dropTransition(previous_mode);
    if (why == Interruption::Replaced) state_->forgetTake(); else state_->clearTakePreview();
    state_->transition_mode = MixerState::TransitionMode::Idle;
    guard.release();
    logstream << "mixer: interrupted transition; " << (was_on_air ? "back to the program picture"
                                                                  : "nothing was on air yet");
}

void MixerOrchestrator::dropTransition(MixerState::TransitionMode dropped) {
    auto cleanup = [](auto action) {
        try { action(); }
        catch (const std::exception& e) {
            logstream << "mixer: transition cleanup failed: " << e.what();
        }
    };
    cleanup([&] { restoreProgramRouting(dropped); });
    if ((dropped == MixerState::TransitionMode::Wipe || state_->wipe_on_air) && !state_->wipe_group_name.empty())
        // A resident chain parks at once. Group management retires a per-take decoder
        // independently: waiting here would add teardown time to every correction.
        cleanup([&] { retireWipeChain(); });
    state_->blend_on_air = state_->wipe_on_air = state_->take_paused = false;
    state_->take_request.reset();
    endTakeClock(TakeClock::End::Cancelled, 0);   // keys going with it return to their state
    // A take the lever cancelled leaves it engaged where it is: moving it on starts the next.
    if (state_->take_lever_end == LeverEnd::None) state_->lever.release();
    state_->take_manual = false;
    state_->take_lever_end = LeverEnd::None;
    state_->take_progress = 0;
    // The program slot's routing was restored: the other slot is cold now.
    state_->pvw_slot_scene.clear();
}

void MixerOrchestrator::restoreProgramRouting(MixerState::TransitionMode dropped) {
    // A cut that had not landed left program untouched, unless it was completing a transition
    // on air; everything else showed something that the program picture now replaces.
    const bool picture_changed = dropped != MixerState::TransitionMode::Cut ||
                                 state_->blend_on_air || state_->wipe_on_air;
    switchProgramSelector(state_->pgm_is_slot_a);
    applyPostTransitionRouting(state_->pgm_is_slot_a, state_->pgmScene(), picture_changed);
    applySceneControls(state_->pgmScene());
    for (const auto& [name, key, value] : std::vector<std::tuple<std::string, std::string, int>>{
            {state_->wipe_selector_name, "active", 0}, {state_->wipe_otm_name, "outputs", 1}}) {
        if (name.empty()) continue;
        setNodeObject(name, key, Parameters(value));
    }
}

void MixerOrchestrator::abortTransition(uint64_t generation) noexcept {
    // All callers hold the control mutex. A cancelled worker must never undo
    // the routing of its replacement transition.
    if (state_->transition_generation != generation) return;
    if (state_->cut_latency) state_->cut_latency->timing.cancel("failed");
    ++state_->transition_generation;
    state_->take_lever_end = LeverEnd::None;   // a failed take ends without the lever
    dropTransition(state_->transition_mode.load());
    state_->clearTakePreview();
    state_->transition_mode = MixerState::TransitionMode::Idle;
}

void MixerOrchestrator::finishTransition(bool new_pgm_is_slot_a, std::string new_pgm_scene, int64_t effective_ns) {
    state_->completeTransition(new_pgm_is_slot_a, std::move(new_pgm_scene), effective_ns);
    logstream << "mixer: program " << state_->pgm_scene_name << " on slot " << (new_pgm_is_slot_a ? 'A' : 'B')
              << ", preview " << (state_->preview.pvw.empty() ? "cleared" : state_->preview.pvw)
              << " from pts_ns=" << effective_ns;
}

int64_t MixerOrchestrator::mixChangeNs() const {
    const int64_t shown_ns = selectorOutputNs();
    return (shown_ns ? shown_ns : monotonicNs()) + 2 * av_rescale(1000000000, state_->fps_den, state_->fps_num);
}

int64_t MixerOrchestrator::selectorOutputNs() const {
    const auto emitted = edgeLastTsIfExists(nodes_, firstDstEdgeName(nodes_, state_->source_switcher_name));
    return emitted.isValid() ? emitted.timestamp({1, 1000000000}) : 0;
}

std::vector<std::string> MixerOrchestrator::sceneNames() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    std::vector<std::string> names;
    names.reserve(state_->scenes.size());
    for (const auto& [name, _] : state_->scenes)
        names.push_back(name);
    std::sort(names.begin(), names.end());
    return names;
}

Parameters MixerOrchestrator::status() const {
    std::unique_lock<std::mutex> lock(state_->mutex);
    Parameters s;
    s["pgm_scene"] = state_->pgm_scene_name;
    s["pvw_slot_scene"] = state_->pvw_slot_scene;
    s["preview_followers"] = state_->preview_followers.load();
    s["pgm_slot"] = state_->pgm_is_slot_a ? "A" : "B";
    s["now_pts_ms"] = wallclock.pts();
    s["cut_latency"] = state_->cut_latency ? state_->cut_latency->status() : Parameters(nullptr);
    {
        // The preview shown, and every AUX multiview's PVW change latency next to the program's
        // (mixer_pvw_follow).
        std::lock_guard<std::mutex> feed(state_->preview_mutex);
        s["pvw_scene"] = state_->preview.pvw;
        s["pvw_revision"] = state_->preview.revision;
        s["pvw_latency"] = Parameters(state_->preview_follow_samples);
    }
    s["prewarm_cut_scenes"] = state_->prewarm_cut_scenes;
    s["prewarm_source_mask"] = toParameters(state_->prewarm_source_mask);
    auto mode = state_->transition_mode.load();
    switch (mode) {
        case MixerState::TransitionMode::Idle: s["transition"] = "idle"; break;
        case MixerState::TransitionMode::Cut: s["transition"] = "cut"; break;
        case MixerState::TransitionMode::Crossfade: s["transition"] = "crossfade"; break;
        case MixerState::TransitionMode::Wipe: s["transition"] = "wipe"; break;
    }
    s["transition_on_air"] = state_->blend_on_air || state_->wipe_on_air;
    s["transition_paused"] = state_->take_paused && !state_->take_manual;
    s["transition_manual"] = state_->take_manual;
    s["transition_progress"] = state_->take_progress;
    s["transition_duration_ms"] = state_->take_duration_ns / 1000000;
    s["lever"] = {{"position", state_->lever.position()}, {"origin", state_->lever.origin()},
                  {"engaged", state_->lever.engaged()}};
    // Stall counters of both slot compositors, which take turns on program. Each
    // publishes them every 60 frames; one that is restarting is skipped, not waited for.
    const std::pair<const char*, std::string> slots[] = {
        {"A", state_->slot_a.compositor_name}, {"B", state_->slot_b.compositor_name}};
    const std::string wipe_cache_store = state_->wipe_cache_store;
    lock.unlock();
    // Read from the shared store, not the clip_cache node: the store lives for the
    // life of the instance whether or not the player group has been started.
    if (!wipe_cache_store.empty())
        s["wipe_cache"] = InstanceSharedObjects<avp::clipcache::ClipCache>::get(
            nodes_->instanceData(), wipe_cache_store)->status();
    Parameters playout = Parameters::object();
    for (const auto& [slot, name] : slots) {
        Parameters node_status;
        const auto node = nodes_->node_if_exists(name);
        if (!node || !node->getObjectTry("status", node_status) || !node_status.contains("playout")) continue;
        auto& counters = node_status["playout"];
        playout[slot] = {{"frames", counters["frames"]}, {"repeats", counters["repeats"]},
                         {"missed_deadlines", counters["missed_deadlines"]}};
        if (counters.contains("timing")) playout[slot]["timing"] = counters["timing"];
    }
    s["playout"] = playout;
    return s;
}

void MixerOrchestrator::enableCutMeasurements(const std::string& mixer_name, const std::string& encoder_name) {
    std::lock_guard<std::mutex> lock(state_->mutex);
    auto selector = std::dynamic_pointer_cast<avp::mixer::CutLatencyObserver>(nodes_->node(state_->source_switcher_name)->node());
    auto encoder = std::dynamic_pointer_cast<avp::mixer::CutLatencyObserver>(nodes_->node(encoder_name)->node());
    if (!selector || !encoder || nodes_->node(encoder_name)->parameters().value("type", std::string()) != "enc_video")
        throw Error("mixer.measurements requires created source_switcher and enc_video nodes");
    if (state_->cut_latency) {
        if (state_->cut_latency->encoder_name != encoder_name)
            throw Error("mixer.measurements is already bound to another encoder");
        return;
    }
    if (selector->cutLatencyProbe() || encoder->cutLatencyProbe())
        throw Error("mixer.measurements node already belongs to another probe");
    auto probe = std::make_shared<avp::mixer::CutLatencyProbe>(mixer_name, encoder_name);
    selector->setCutLatencyProbe(probe);
    encoder->setCutLatencyProbe(probe);
    state_->cut_latency = std::move(probe);
}

}  // namespace avp::mixer
