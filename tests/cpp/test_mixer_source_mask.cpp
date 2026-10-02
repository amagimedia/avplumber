#include "SharedTimeline.hpp"
#include "mixer/primitives/MixerState.hpp"
#include <cassert>
#include <limits>

int main() {
    using namespace avp::mixer;
    MixerState state;
    SceneDefinition scene;
    SourceMask expected;
    for (int index : {0, 31, 32, 44, 63, 64, 95, 127, 128, 200, 255}) {
        auto name = std::to_string(index);
        state.sources[name].input_index = index;
        scene.sources[name] = {};
        expected.set(index);
    }
    assert(state.computeActiveInputsMask(scene) == expected);

    // Prewarm forces both slot bits on, for the highest pad as well.
    state.prewarm_source_mask = SourceMask().set(255);
    assert(state.sourceOutputMask(state.sources.at("255"), 0) == 3);
    assert(state.sourceOutputMask(state.sources.at("63"), 0) == 0);
    assert(state.sourceOutputMask(state.sources.at("32"), 1) == 1);

    // Wire form: a number while it fits in 64 bits, a bit string above that.
    const SourceMask low = SourceMask().set(0).set(63);
    assert(toParameters(low).is_number());
    assert(parseSourceMask(toParameters(low)) == low);
    assert(toParameters(expected).is_string());
    assert(parseSourceMask(toParameters(expected)) == expected);
    assert(parseSourceMask(Parameters(std::string(200, '1'))).test(199));
    assert(!parseSourceMask(Parameters(std::string(200, '1'))).test(200));
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
}
