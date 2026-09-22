#pragma once
// One bit per compositor pad. A show's sources are the pads of every slot
// compositor, so this width is the hard limit on sources per mixer.
//
// It was a uint64_t, which capped a show at 64 sources. Two words carry 128.
// The control protocol already had a wider form: parseBitmask accepts a
// least-significant-bit-first string of '0'/'1', which is what this type
// writes into JSON once a mask no longer fits in a 64-bit number.
#include "../../util.hpp"

#include <cstdint>
#include <string>

namespace avp::mixer {

struct SourceMask {
    static constexpr int kBits = 128;
    static constexpr int kWordBits = 64;

    uint64_t w[2] = {0, 0};   // w[0] holds pads 0..63, w[1] pads 64..127

    static constexpr SourceMask bit(int index) {
        SourceMask m;
        m.w[index / kWordBits] = uint64_t{1} << (index % kWordBits);
        return m;
    }
    static constexpr SourceMask all() { return SourceMask{{~uint64_t{0}, ~uint64_t{0}}}; }

    constexpr bool test(int index) const {
        return (w[index / kWordBits] >> (index % kWordBits)) & 1u;
    }
    constexpr void set(int index) { w[index / kWordBits] |= uint64_t{1} << (index % kWordBits); }

    constexpr bool any() const { return w[0] || w[1]; }
    constexpr explicit operator bool() const { return any(); }

    constexpr SourceMask operator|(const SourceMask& o) const { return {{w[0] | o.w[0], w[1] | o.w[1]}}; }
    constexpr SourceMask operator&(const SourceMask& o) const { return {{w[0] & o.w[0], w[1] & o.w[1]}}; }
    constexpr SourceMask operator~() const { return {{~w[0], ~w[1]}}; }
    SourceMask& operator|=(const SourceMask& o) { w[0] |= o.w[0]; w[1] |= o.w[1]; return *this; }
    constexpr bool operator==(const SourceMask& o) const { return w[0] == o.w[0] && w[1] == o.w[1]; }
    constexpr bool operator!=(const SourceMask& o) const { return !(*this == o); }

    /// True while every set bit fits in a JSON number.
    constexpr bool fitsInWord() const { return w[1] == 0; }
    /// The bit string parseBitmask reads: index i is pad i, trailing zeros trimmed.
    std::string toBitString() const {
        std::string s(kBits, '0');
        for (int i = 0; i < kBits; ++i)
            if (test(i)) s[i] = '1';
        const auto last = s.find_last_of('1');
        return last == std::string::npos ? std::string("0") : s.substr(0, last + 1);
    }
};

/// A mask as the control protocol carries it: a number while it fits in 64 bits
/// (what every show under 65 sources sends), a bit string above that.
inline Parameters toParameters(const SourceMask& mask) {
    if (mask.fitsInWord()) return Parameters(mask.w[0]);
    return Parameters(mask.toBitString());
}

/// Read either form. Rejects a bit string longer than the mask.
inline SourceMask parseSourceMask(const Parameters& value) {
    SourceMask mask;
    if (value.is_string()) {
        const auto s = value.get<std::string>();
        if (s.size() > (size_t)SourceMask::kBits)
            throw Error("bitmask string exceeds mask width");
        for (size_t i = 0; i < s.size(); ++i)
            if (s[i] == '1') mask.set((int)i);
        return mask;
    }
    mask.w[0] = value.get<uint64_t>();
    return mask;
}

}  // namespace avp::mixer
