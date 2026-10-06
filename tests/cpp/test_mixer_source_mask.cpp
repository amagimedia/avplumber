#include "output_mask.hpp"
#include "mixer/routing.hpp"
#include <cassert>
#include <limits>

int main() {
    using namespace avp::mixer;
    MixerState state;
    SceneDefinition scene;
    SourceMask expected;
    for (int index : {0, 31, 32, 44, 63, 64, 95, 127, 128, 150, 191}) {
        auto name = std::to_string(index);
        state.sources[name].input_index = index;
        scene.sources[name] = {};
        expected.set(index);
    }
    assert(state.computeActiveInputsMask(scene) == expected);

    // Prewarm forces both slot bits on, for the highest pad as well.
    state.prewarm_source_mask = SourceMask().set(191);
    assert(state.sourceOutputMask(state.sources.at("191"), 0) == 3);
    assert(state.sourceOutputMask(state.sources.at("63"), 0) == 0);
    assert(state.sourceOutputMask(state.sources.at("32"), 1) == 1);

    // Wire form: a number while it fits in 64 bits, a bit string above that.
    const SourceMask low = SourceMask().set(0).set(63);
    assert(toParameters(low).is_number());
    assert(parseSourceMask(toParameters(low)) == low);
    assert(toParameters(expected).is_string());
    assert(parseSourceMask(toParameters(expected)) == expected);
    assert(parseSourceMask(Parameters(std::string(150, '1'))).test(149));
    assert(!parseSourceMask(Parameters(std::string(150, '1'))).test(150));
    // A 64-bit number still means pads 0..63, as older mixers sent it.
    const SourceMask all_low(std::numeric_limits<uint64_t>::max());
    assert(parseSourceMask(Parameters(std::numeric_limits<uint64_t>::max())) == all_low);
    assert(all_low.test(63) && !all_low.test(64));
    bool rejected = false;
    try { parseSourceMask(Parameters(std::string(kSourceMaskBits + 1, '1'))); }
    catch (const Error&) { rejected = true; }
    assert(rejected);

    // The narrow parser other nodes use (one_to_many outputs) rejects what it cannot hold.
    assert(parseBitmask(Parameters(std::string(32, '1'))) == std::numeric_limits<uint32_t>::max());
    rejected = false;
    try { parseBitmask(Parameters(std::string(33, '1'))); }
    catch (const Error&) { rejected = true; }
    assert(rejected);
    // Sparse layer order follows source index, not name or insertion order.
    SceneDefinition sparse, other;
    sparse.sources["191"] = {{"dst_x", 80}, {"z", 4}};
    sparse.sources["32"] = {{"dst_x", 0}, {"z", 4}};
    other.sources["63"] = Parameters::object();
    const auto layers = compositorLayersFromScene(state, sparse);
    assert(layers.size() == 2);
    assert(layers[0]["input"] == 32 && layers[1]["input"] == 191);
    assert(layers[1]["dst_x"] == 80 && layers[0]["z"] == 4);
    assert(sourceOutputsForScenes(state, "32", state.sources.at("32"), &sparse, &other) == 1);
    assert(sourceOutputsForScenes(state, "63", state.sources.at("63"), &sparse, &other) == 2);
    assert(sourceOutputsForScenes(state, "32", state.sources.at("32"), &sparse, &sparse) == 3);
    assert(sourceOutputsForScenes(state, "32", state.sources.at("32"), nullptr, &sparse) == 2);
    assert(sourceOutputsForScenes(state, "32", state.sources.at("32"), nullptr, nullptr) == 0);
    assert(sourceOutputsForScenes(state, "191", state.sources.at("191"), nullptr, nullptr) == 3);

    // Publishing both slots preserves unrelated router outputs and clears retired slots.
    auto& routed = state.sources["routed"];
    routed.routed = true; routed.router_node_name = "router";
    routed.route_output_a = 0; routed.route_output_b = 2;
    state.router_output_counts["router"] = 3;
    state.router_routes["router"] = {8, 77, 9};
    sparse.sources["routed"] = Parameters::object(); sparse.routes["routed"] = 64;
    auto tables = currentRouterTables(state);
    setRoutedSlotInTables(state, tables, true, &sparse);
    setRoutedSlotInTables(state, tables, false, nullptr);
    assert((tables.at("router") == std::vector<int>{64, 77, -1}));

}
