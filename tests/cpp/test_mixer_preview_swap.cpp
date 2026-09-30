// The preview after a take: swapped with the scene that left program (OBS's default), or
// cleared; the PVW slot is cold either way, and every change wakes the followers.
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
    const auto revision = state.pvw_revision;

    // A follower waits on the revision; the completed take wakes it.
    std::atomic<bool> woken{false};
    std::thread follower([&] {
        std::unique_lock<std::mutex> lock(state.mutex);
        state.preview_changed.wait(lock, [&] { return state.pvw_revision != revision; });
        woken = true;
    });

    {
        std::lock_guard<std::mutex> lock(state.mutex);
        state.completeTransition(false, "b", 123456789);
    }
    follower.join();
    assert(woken);
    assert(state.pgm_scene_name == "b" && !state.pgm_is_slot_a);
    assert(state.pvw_scene_name == "a");          // shown: the scene that left program
    assert(state.pvw_slot_scene.empty());         // but the slot is cold: a take of "a" reloads it
    assert(state.pvw_revision == revision + 1);
    assert(state.pvw_effective_ns == 123456789);
    assert(state.pvw_published_ns > 0);
    assert(state.transition_mode == MixerState::TransitionMode::Idle);

    // Taking the program scene again swaps nothing: there is no other scene to preview.
    state.completeTransition(true, "b", 0);
    assert(state.pvw_scene_name.empty() && state.pgm_is_slot_a && state.pvw_effective_ns == 0);
    assert(state.pvw_revision == revision + 2);

    // Swap off: a take clears the preview, as it always did.
    state.swap_preview = false;
    state.pvw_slot_scene = "c";
    state.completeTransition(false, "c", 5);
    assert(state.pgm_scene_name == "c" && state.pvw_scene_name.empty() && state.pvw_slot_scene.empty());
    assert(state.pvw_revision == revision + 3);

    // An explicit preview publishes at once (effective 0) and leaves the program alone.
    state.publishPreview("a", 0);
    assert(state.pvw_scene_name == "a" && state.pgm_scene_name == "c" && state.pvw_effective_ns == 0);
    assert(state.pvw_revision == revision + 4);
}
