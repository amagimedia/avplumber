#pragma once

#include <cstdint>
#include <stdexcept>

namespace avp::mixer {

// Integer frame indices keep fractional rates on one common monotonic grid.
class FrameRate {
    int64_t numerator_;
    int64_t denominator_;

    static int64_t floorDivide(__int128 value, int64_t divisor) {
        const auto quotient = value / divisor;
        return static_cast<int64_t>(quotient - (value % divisor < 0));
    }

public:
    FrameRate(int64_t numerator, int64_t denominator)
        : numerator_(numerator), denominator_(denominator) {
        if (numerator <= 0 || denominator <= 0 ||
            denominator > INT64_MAX / 1000000000 ||
            numerator > denominator * 1000000000)
            throw std::invalid_argument("invalid mixer frame rate");
    }

    int64_t time(int64_t index) const {
        return floorDivide(static_cast<__int128>(index) * denominator_ * 1000000000,
                           numerator_);
    }

    int64_t nearestIndex(int64_t time_ns) const {
        const int64_t period_numerator = denominator_ * 1000000000;
        // A rational timestamp truncated to ns represents [ns, ns+1).
        // Keep a half-tick boundary in that interval on the upper tick;
        // otherwise 60 Hz at a half-tick phase alternates slots every 3 frames.
        return floorDivide((static_cast<__int128>(time_ns) + 1) * numerator_ - 1 +
                           period_numerator / 2, period_numerator);
    }

    int64_t atOrBefore(int64_t time_ns) const {
        return floorDivide((static_cast<__int128>(time_ns) + 1) * numerator_ - 1,
                           denominator_ * 1000000000);
    }
};

} // namespace avp::mixer
