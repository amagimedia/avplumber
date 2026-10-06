#pragma once

#include <algorithm>
#include <chrono>
#include <limits>
#include <optional>
#include <cstdint>

namespace avp::mixer {

/// A steady_clock instant on the monotonic ns scale monotonicNs() reads.
inline int64_t monotonicNs(std::chrono::steady_clock::time_point at) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(at.time_since_epoch()).count();
}

inline int64_t monotonicNs() { return monotonicNs(std::chrono::steady_clock::now()); }

inline int waitMilliseconds(std::optional<int64_t> deadline, int64_t now) {
    if (!deadline) return 10;
    if (*deadline <= now) return 1;
    const int64_t remaining = *deadline - now;
    return static_cast<int>(std::min<int64_t>(
        remaining / 1000000 + (remaining % 1000000 != 0), std::numeric_limits<int>::max()));
}

} // namespace avp::mixer
