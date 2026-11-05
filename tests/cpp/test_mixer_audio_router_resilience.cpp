// A missing or restarting audio_router node (named by mixer.init's audio_router_node) must not
// abort a take: completeTake/applyPostTransitionRouting (scene.cpp) and the per-frame fade/wipe
// progress (fade.cpp, wipe.cpp) treat it as a best-effort AFV side signal, the same way the
// encoder keyframe trigger beside it already does. Before the fix, MixerGraph::node threw
// "Mixer node no longer exists" straight out of emitTransitionEnded/emitTransitionProgress,
// which selectFrame's try/catch (fade.cpp) caught by aborting the whole transition.
#include "graph_mgmt.hpp"
#define private public
#include "mixer/orchestrator/MixerOrchestrator.hpp"
#undef private
#include "mixer/MixerGraph.hpp"
#include <cassert>
#include <functional>

using namespace avp::mixer;

namespace {

bool throws(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

}

int main() {
    const auto manager = std::make_shared<NodeManager>();   // no nodes created: audio_router_node
                                                              // names a node that will never exist
    auto state = std::make_shared<MixerState>();
    state->audio_router_node_name = "missing_audio_router";
    MixerOrchestrator control(std::make_shared<MixerGraph>(manager), state);

    SceneDefinition scene;
    scene.sources["cam1"] = Parameters();

    // emitTransitionEnded (scene.cpp, fired by completeTake before the post_otm/active_inputs
    // flips) must swallow the missing node's error, not let it escape.
    assert(!throws([&] { control.emitTransitionEnded(scene, 1000); }));

    // emitTransitionProgress (fade.cpp, fired on every program frame of a fade/dip/wipe) is the
    // same contract, reached on a different path (selectProgram/selectWipe per tick).
    state->pgm_scene_name = "a";
    state->scenes["a"] = scene;
    state->pvw_slot_scene = "b";
    state->scenes["b"] = scene;
    assert(!throws([&] { control.emitTransitionProgress("fade", 0.5, 2000); }));

    // Sanity: with the router node actually missing, MixerGraph::node still throws for anyone
    // calling it directly -- the fix is in emitTransitionEnded/Progress catching it, not in the
    // node lookup becoming lenient.
    MixerGraph graph(manager);
    assert(throws([&] { graph.node("missing_audio_router"); }));
}
