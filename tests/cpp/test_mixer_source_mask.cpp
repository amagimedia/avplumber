#include "SharedTimeline.hpp"
#include "mixer/primitives/MixerState.hpp"
#include <cassert>
#include <limits>

int main() {
    using namespace avp::mixer;
    MixerState state;
    SceneDefinition scene;
    uint64_t expected = 0;
    for (int index : {0, 31, 32, 44, 63}) {
        auto name = std::to_string(index);
        state.sources[name].input_index = index;
        scene.sources[name] = {};
        expected |= uint64_t{1} << index;
    }
    assert(state.computeActiveInputsMask(scene) == expected);
    state.prewarm_source_mask = uint64_t{1} << 63;
    assert(state.sourceOutputMask(state.sources.at("63"), 0) == 3);
    assert(state.sourceOutputMask(state.sources.at("31"), 0) == 0);
    assert(state.sourceOutputMask(state.sources.at("32"), 1) == 1);

    assert(parseBitmask<uint64_t>(Parameters(expected)) == expected);
    std::string bits(64, '0');
    bits[0] = bits[32] = bits[63] = '1';
    assert(parseBitmask<uint64_t>(Parameters(bits)) ==
           ((uint64_t{1} << 63) | (uint64_t{1} << 32) | 1));
    assert(parseBitmask<uint64_t>(Parameters(std::string(64, '1'))) ==
           std::numeric_limits<uint64_t>::max());
    assert(parseBitmask(Parameters(std::string(32, '1'))) ==
           std::numeric_limits<uint32_t>::max());
    bool rejected = false;
    try { parseBitmask<uint64_t>(Parameters(std::string(65, '1'))); }
    catch (const Error&) { rejected = true; }
    assert(rejected);
}
