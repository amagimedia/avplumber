// MixerOrchestrator core: construction, node/edge adapters, transition
// interruption and abort, and status. Scene loading and takes live in the
// sibling scene/cut/fade/wipe.cpp files.
#include "internal.hpp"
#include "../../nodes/clip_cache/ClipCache.hpp"

namespace avp::mixer {

namespace {
// The hold releases on the first frame newer than the selector's `emitted` (ns) output; without
// one, on a frame stamped after now, two frames away behind the playout delay.
int64_t releaseAfter(int64_t emitted_ns) {
    return emitted_ns ? emitted_ns + 1 : wallclock.pts() * 1000000;
}
}

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

int64_t MixerOrchestrator::armWipeChain(const std::string& wipe_file) {
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
    setNodeObject(state_->wipe_input_node_name, "play", Parameters(wipe_file));
    return nodes_->node(state_->wipe_input_node_name)->getObject("status").at("start_tick").get<int64_t>();
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
    // A crossfade's blended picture exists only inside the transition compositor,
    // so it has to be frozen to survive the interruption. A wipe's does not: the
    // output carries the wipe graphic, and freezing that would paint the graphic
    // into the program, where the next wipe would composite over it and the two
    // would stack. In every other mode the program slot keeps rendering, so the
    // direct branch restored below is already the right picture.
    const bool freeze = previous_mode == MixerState::TransitionMode::Crossfade;
    auto snapshot = outputSnapshot();
    if (freeze) {
        std::lock_guard<std::mutex> lock(snapshot->mutex);
        if (!snapshot->output_connected)
            throw Error("mixer: interruption requires mixer_snapshot output and slot nodes");
        snapshot->frames.capture(state_->pgmSourceSwitcherIndex());
    }
    const auto generation = ++state_->transition_generation;
    TransitionGuard guard([&] { abortTransition(generation); });
    restoreProgramRouting(previous_mode);
    if (previous_mode == MixerState::TransitionMode::Wipe && !state_->wipe_group_name.empty()) {
        // A resident chain parks at once. Group management retires a per-take decoder
        // independently: waiting here would add teardown time to every correction,
        // including a hard cut.
        retireWipeChain();
    }
    // The program slot's routing was restored above: the other slot is cold now.
    state_->pvw_slot_scene.clear();
    if (why == Interruption::Replaced) state_->forgetTake(); else state_->clearTakePreview();
    state_->transition_mode = MixerState::TransitionMode::Idle;
    {
        // Read after the selector was switched: everything it has emitted so far, from the old
        // program or the dropped transition, is older than the release and cannot end the hold
        // even when still in flight to the output, and the first frame of the new selection does.
        const auto release_pts = releaseAfter(selectorOutputNs());
        std::lock_guard<std::mutex> lock(snapshot->mutex);
        // Without a freeze, drop any substitution an earlier interruption left in the slot.
        if (!freeze) snapshot->frames.finish();
        snapshot->frames.arm(release_pts, freeze);
    }
    guard.release();
    logstream << "mixer: interrupted transition; " << (freeze ? "retained the blended picture"
                                                             : "returned to the program picture");
}

void MixerOrchestrator::restoreProgramRouting(MixerState::TransitionMode dropped) {
    bool picture_changed = dropped != MixerState::TransitionMode::Cut;
    if (!picture_changed) {
        // A cut that interrupted a crossfade froze its blend; dropping this cut
        // too puts the live program back on air.
        auto snapshot = outputSnapshot();
        std::lock_guard<std::mutex> lock(snapshot->mutex);
        picture_changed = snapshot->frames.holding() ||
            snapshot->frames.replaces(state_->pgmSourceSwitcherIndex());
    }
    switchProgramSelector(state_->pgm_is_slot_a);
    applyPostTransitionRouting(state_->pgm_is_slot_a, state_->pgm_scene_name, picture_changed);
    applySceneControls(state_->scenes.at(state_->pgm_scene_name));
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
    const auto mode = state_->transition_mode.load();
    auto cleanup = [](auto action) {
        try { action(); }
        catch (const std::exception& e) {
            logstream << "mixer: transition abort cleanup failed: " << e.what();
        }
    };
    cleanup([&] { restoreProgramRouting(mode); });
    if (mode == MixerState::TransitionMode::Wipe && !state_->wipe_group_name.empty())
        cleanup([&] { retireWipeChain(); });
    // Remove slot substitution even when the target never produced a frame.
    // The output gate still waits for a fresh program frame before releasing.
    cleanup([&] { finishSnapshot(); });
    state_->pvw_slot_scene.clear();
    state_->clearTakePreview();
    state_->transition_mode = MixerState::TransitionMode::Idle;
}

void MixerOrchestrator::finishTransition(bool new_pgm_is_slot_a, std::string new_pgm_scene, int64_t effective_ns) {
    state_->completeTransition(new_pgm_is_slot_a, std::move(new_pgm_scene), effective_ns);
    logstream << "mixer: program " << state_->pgm_scene_name << " on slot " << (new_pgm_is_slot_a ? 'A' : 'B')
              << ", preview " << (state_->preview.pvw.empty() ? "cleared" : state_->preview.pvw)
              << " from pts_ns=" << effective_ns;
}

std::shared_ptr<OutputSnapshot> MixerOrchestrator::outputSnapshot() const {
    return InstanceSharedObjects<OutputSnapshot>::get(
        nodes_->instanceData(), state_->source_switcher_name + "_snapshot");
}

void MixerOrchestrator::finishSnapshot(int64_t emitted) {
    auto snapshot = outputSnapshot();
    std::lock_guard<std::mutex> lock(snapshot->mutex);
    snapshot->frames.finish();
    snapshot->frames.arm(releaseAfter(emitted), false);
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
