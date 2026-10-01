// The preview after a take: swapped with the scene that left program (OBS's default, always
// on), or cleared when the take keeps the program scene; the PVW slot is cold either way, and
// every change wakes the followers on the feed's own lock, published before the take's routing
// and once per take.
#include "SharedTimeline.hpp"
#include "mixer/primitives/MixerState.hpp"
#include <atomic>
#include <cassert>
#include <thread>

int main() {
    using namespace avp::mixer;
    MixerState state;
    state.pgm_scene_name = "a";
    state.pvw_slot_scene = "b";   // armed
    state.pvw_scene_name = "b";
    state.transition_mode = MixerState::TransitionMode::Cut;
    state.take_received_ns = 1000;   // the cut command's receipt, for the followers' latency
    const auto revision = state.preview.revision;

    // A follower waits on the revision with the feed's lock alone; the take wakes it right
    // after the selector switch, while the take still holds the mixer's mutex for its routing.
    std::atomic<bool> woken{false};
    MixerState::PreviewChange seen;
    std::thread follower([&] {
        std::unique_lock<std::mutex> lock(state.preview_mutex);
        state.preview_changed.wait(lock, [&] { return state.preview.revision != revision; });
        seen = state.preview;
        woken = true;
    });

    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.publishTakePreview("b", 123456789);
        follower.join();   // woken under this lock: the routing has not happened yet
        assert(woken && state.take_preview_published);
        assert(state.pgm_scene_name == "a" && state.pvw_slot_scene == "b");   // the take is not complete
        state.completeTransition(false, "b", 123456789);   // publishes nothing more
    }
    assert(seen.revision == revision + 1 && seen.pvw == "a" && seen.pgm == "b" && seen.kind == "cut");
    assert(seen.effective_ns == 123456789 && seen.published_ns > 0 && seen.received_ns == 1000);
    assert(state.preview.revision == revision + 1);   // one revision per take
    assert(state.take_received_ns == 0 && !state.take_preview_published);   // handed over once
    assert(state.pgm_scene_name == "b" && !state.pgm_is_slot_a);
    assert(state.pvw_scene_name == "a");          // shown: the scene that left program
    assert(state.pvw_slot_scene.empty());         // but the slot is cold: a take of "a" reloads it
    assert(state.transition_mode == MixerState::TransitionMode::Idle);

    // A take that did not publish (a wipe's switch happens under the wipe, untimed) publishes
    // at completion; taking the program scene again swaps nothing: there is no other scene.
    state.transition_mode = MixerState::TransitionMode::Wipe;
    state.completeTransition(true, "b", 0);
    assert(state.pvw_scene_name.empty() && state.pgm_is_slot_a);
    assert(state.preview.revision == revision + 2 && state.preview.effective_ns == 0);
    assert(state.preview.pvw.empty() && state.preview.pgm == "b" && state.preview.kind == "wipe");
    assert(state.preview.received_ns == 0 && state.transition_mode == MixerState::TransitionMode::Idle);

    // A fade that did not publish swaps at completion too: the scene it left is previewed.
    state.pvw_slot_scene = "c";
    state.transition_mode = MixerState::TransitionMode::Crossfade;
    state.completeTransition(false, "c", 5);
    assert(state.pgm_scene_name == "c" && state.pvw_scene_name == "b" && state.pvw_slot_scene.empty());
    assert(state.preview.revision == revision + 3 && state.preview.pvw == "b" && state.preview.pgm == "c");
    assert(state.preview.effective_ns == 5 && state.preview.kind == "fade");

    // An explicit preview publishes at once (effective 0), untimed, beside the current program.
    state.publishPreview("a", 0);
    assert(state.pvw_scene_name == "a" && state.pgm_scene_name == "c");
    assert(state.preview.revision == revision + 4 && state.preview.pvw == "a" && state.preview.pgm == "c");
    assert(state.preview.effective_ns == 0 && state.preview.received_ns == 0 && state.preview.kind.empty());

    // A dropped take clears the preview and forgets its receipt and its publish.
    state.take_received_ns = 7;
    state.take_preview_published = true;
    state.clearTakePreview();
    assert(state.pvw_scene_name.empty() && state.preview.revision == revision + 5 && state.preview.kind.empty());
    assert(state.take_received_ns == 0 && !state.take_preview_published);

    // A take replaced by another forgets the same but keeps the preview shown, publishing
    // nothing: the replacing take's switch publishes the next change.
    state.publishPreview("a", 0);
    state.take_received_ns = 9;
    state.take_preview_published = true;
    state.forgetTake();
    assert(state.pvw_scene_name == "a" && state.preview.revision == revision + 6 && state.preview.pvw == "a");
    assert(state.take_received_ns == 0 && !state.take_preview_published);

    // A follower's sample is the mixer's to report (mixer.status pvw_latency), under the feed's lock.
    {
        std::lock_guard<std::mutex> lock(state.preview_mutex);
        state.preview_follow_samples["aux_mv_pvw"] = {{"pvw_minus_pgm_ms", 16.7}};
        assert(Parameters(state.preview_follow_samples).at("aux_mv_pvw").at("pvw_minus_pgm_ms") == 16.7);
    }
}
