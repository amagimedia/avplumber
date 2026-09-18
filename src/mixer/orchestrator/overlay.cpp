// Overlay branch enable/disable with a monotonic-PTS handover on the selector.
#include "internal.hpp"

namespace avp::mixer {

namespace {
constexpr int kOverlayDirectInput = 0;
constexpr int kOverlayCompositedInput = 1;
}

void MixerOrchestrator::setOverlayEnabled(bool enabled, int64_t ready_timeout_ms) {
    std::string source_otm_name;
    std::string overlay_otm_name;
    std::string selector_name;
    std::string candidate_edge_name;
    std::string selector_output_edge_name;
    av::Timestamp visible_ts = NOTS;
    av::Timestamp initial_candidate_ts = NOTS;
    uint64_t generation = 0;
    int64_t timeout_ms = 0;
    int64_t poll_ms = 0;
    const int candidate_input = enabled ? kOverlayCompositedInput : kOverlayDirectInput;

    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (state_->overlay_source_otm_name.empty() ||
            state_->overlay_otm_name.empty() ||
            state_->overlay_selector_name.empty()) {
            throw Error("mixer.overlay: overlay nodes are not configured");
        }
        source_otm_name = state_->overlay_source_otm_name;
        overlay_otm_name = state_->overlay_otm_name;
        selector_name = state_->overlay_selector_name;
        timeout_ms = ready_timeout_ms >= 0 ? ready_timeout_ms : state_->overlay_ready_timeout_ms;
        poll_ms = state_->overlay_ready_poll_ms;
        generation = ++state_->overlay_generation;

        selector_output_edge_name = firstDstEdgeName(nodes_, selector_name);
        candidate_edge_name = edgeNameAt(nodes_, selector_name, "src", candidate_input);
        visible_ts = edgeLastTsIfExists(nodes_, selector_output_edge_name);
        initial_candidate_ts = edgeLastTsIfExists(nodes_, candidate_edge_name);

        if (candidate_edge_name.empty()) {
            throw Error("mixer.overlay: selector " + selector_name + " does not expose input " +
                        std::to_string(candidate_input));
        }

        if (!setNodeObjectIfCreated(nodes_, selector_name, "drop_non_monotonic", Parameters(true))) {
            logstream << "mixer.overlay: queued " << selector_name
                      << ".drop_non_monotonic for node not created yet";
        }

        if (enabled) {
            publishRuntimeObject(selector_name, "active", Parameters(kOverlayDirectInput));
            publishRuntimeObject(source_otm_name, "outputs", Parameters(1u));
            publishRuntimeObject(overlay_otm_name, "outputs", Parameters(3u));
        } else {
            // Keep both legs fed until the direct leg has caught up with the last
            // visible frame. The selector flips only after the wait below.
            publishRuntimeObject(overlay_otm_name, "outputs", Parameters(3u));
        }

        logstream << "mixer.overlay: armed " << (enabled ? "enable" : "disable")
                  << " selector=" << selector_name
                  << " candidate_edge=" << candidate_edge_name
                  << " visible_ts=" << visible_ts
                  << " initial_candidate_ts=" << initial_candidate_ts
                  << " timeout_ms=" << timeout_ms;
    }

    OverlayReadyResult ready = waitForOverlayBranchReady(
        nodes_, state_, generation, candidate_edge_name, initial_candidate_ts,
        visible_ts, timeout_ms, poll_ms);
    if (ready.cancelled) {
        logstream << "mixer.overlay: " << (enabled ? "enable" : "disable")
                  << " superseded before visible switch";
        return;
    }

    {
        std::lock_guard<std::mutex> lock(state_->mutex);
        if (!overlayCommandCurrent(state_, generation)) {
            logstream << "mixer.overlay: " << (enabled ? "enable" : "disable")
                      << " superseded before finalizing";
            return;
        }

        if (!ready.ready) {
            publishRuntimeObject(selector_name, "active", Parameters(kOverlayDirectInput));
            publishRuntimeObject(overlay_otm_name, "outputs", Parameters(1u));
            publishRuntimeObject(source_otm_name, "outputs", Parameters(0u));
            state_->overlay_enabled = false;
            std::ostringstream msg;
            msg << "mixer.overlay: " << (enabled ? "overlay" : "direct")
                << " branch did not reach monotonic PTS before timeout; edge="
                << candidate_edge_name << " last_ts=" << ready.ready_ts;
            throw Error(msg.str());
        }

        publishRuntimeObject(selector_name, "active", Parameters(candidate_input));
        if (!enabled) {
            publishRuntimeObject(overlay_otm_name, "outputs", Parameters(1u));
            publishRuntimeObject(source_otm_name, "outputs", Parameters(0u));
        }
        state_->overlay_enabled = enabled;
        logstream << "mixer.overlay: " << (enabled ? "enabled" : "disabled")
                  << " waited_ms=" << ready.waited_ms
                  << " ready_ts=" << ready.ready_ts
                  << " visible_ts=" << visible_ts;
    }
}

}  // namespace avp::mixer
