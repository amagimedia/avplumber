#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

namespace avp::mixer {

/// Counts durations in buckets that are fractions of the output frame period, so "how many
/// ticks fitted their slot" reads off directly. A duration falls into the first bucket whose
/// bound it does not exceed; the last bucket has no bound. The last kRecent durations are kept
/// for percentiles, which buckets this wide cannot give.
class TickTimes {
public:
    static constexpr int kBuckets = 9;
    static constexpr size_t kRecent = 600;   // 20 s at 30 fps, 10 s at 60
    /// Upper bounds of all buckets but the last, in quarters of the period: 1/4 ... 3 periods.
    static constexpr std::array<int, kBuckets - 1> kBoundQuarters = {1, 2, 3, 4, 5, 6, 8, 12};

    explicit TickTimes(int64_t period_ns = 0) : period_ns_(period_ns) {}

    int bucket(int64_t ns) const {
        for (int i = 0; i < kBuckets - 1; ++i)
            if (ns * 4 <= period_ns_ * kBoundQuarters[i]) return i;
        return kBuckets - 1;
    }

    void add(int64_t ns) {
        ++counts_[bucket(ns)];
        ++count_;
        sum_ns_ += ns;
        if (ns > max_ns_) max_ns_ = ns;
        recent_[(count_ - 1) % kRecent] = ns;
    }

    /// How many durations recentQuantileNs() covers: the last kRecent, or all before that.
    size_t recentCount() const { return static_cast<size_t>(std::min<uint64_t>(count_, kRecent)); }
    /// The duration a fraction `q` (0 to 1) of the recent ones do not exceed, the one at index
    /// floor(q * n) when sorted; 0 before any.
    int64_t recentQuantileNs(double q) const {
        const size_t n = recentCount();
        if (!n) return 0;
        std::array<int64_t, kRecent> sorted = recent_;
        const auto k = sorted.begin() + std::min(n - 1, static_cast<size_t>(q * static_cast<double>(n)));
        std::nth_element(sorted.begin(), k, sorted.begin() + n);
        return *k;
    }

    int64_t periodNs() const { return period_ns_; }
    const std::array<uint64_t, kBuckets> &counts() const { return counts_; }
    uint64_t count() const { return count_; }
    int64_t sumNs() const { return sum_ns_; }
    int64_t maxNs() const { return max_ns_; }

private:
    int64_t period_ns_;
    std::array<uint64_t, kBuckets> counts_{};
    uint64_t count_ = 0;
    int64_t sum_ns_ = 0, max_ns_ = 0;
    std::array<int64_t, kRecent> recent_{};
};

} // namespace avp::mixer
