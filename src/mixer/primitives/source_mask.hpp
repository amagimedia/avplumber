#pragma once
// One bit per compositor pad. A show's sources are the pads of every slot
// compositor, so this width is the hard limit on sources per mixer.
//
// It was a uint64_t, which capped a show at 64 sources. The control protocol
// already had a wider form: parseBitmask accepts a least-significant-bit-first
// string of '0'/'1', which is what toParameters writes into JSON once a mask no
// longer fits in a 64-bit number.
#include "../../util.hpp"

#include <algorithm>
#include <bitset>
#include <cstdint>
#include <string>

namespace avp::mixer {

// 193: 192 sources and the PGM pad of an aux bus that draws the program.
constexpr int kSourceMaskBits = 193;   // pyplumber/mixer/config.py MAX_SOURCES mirrors it
using SourceMask = std::bitset<kSourceMaskBits>;

/// A mask as the control protocol carries it: a number while it fits in 64 bits
/// (what every show under 65 sources sends), a bit string above that, index i
/// for pad i, trailing zeros trimmed.
inline Parameters toParameters(const SourceMask& mask) {
    if ((mask >> 64).none()) return Parameters(static_cast<uint64_t>(mask.to_ullong()));
    std::string bits = mask.to_string();
    std::reverse(bits.begin(), bits.end());
    return Parameters(bits.substr(0, bits.find_last_of('1') + 1));
}

/// Read either form. Rejects a bit string longer than the mask.
inline SourceMask parseSourceMask(const Parameters& value) {
    if (!value.is_string()) return SourceMask(value.get<uint64_t>());
    const auto s = value.get<std::string>();
    if (s.size() > (size_t)kSourceMaskBits)
        throw Error("bitmask string exceeds mask width");
    SourceMask mask;
    for (size_t i = 0; i < s.size(); ++i)
        if (s[i] == '1') mask.set(i);
    return mask;
}

}  // namespace avp::mixer
