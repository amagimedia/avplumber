#pragma once

#ifdef __linux__
#include <sys/timex.h>
#endif
#include <time.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// The process's clocks stay on CLOCK_MONOTONIC: it runs at the NTP-disciplined rate, and nothing
// in the graph has to change domain. What differs between two hosts is only where that clock's
// zero lies (boot time). clockOriginNs() is that difference to a shared scale, sampled once:
//
//     show time = CLOCK_MONOTONIC + clockOriginNs()
//     show time = CLOCK_TAI - kShowEpoch                 (at the moment of the sample)
//
// so two hosts whose system clocks agree turn a monotonic instant into the same show time.
// Off (zero) unless AVP_CLOCK_ORIGIN=tai, which leaves every existing number as it was.
namespace avp {

// 2026-01-01T00:00:00 on the CLOCK_TAI scale of a host whose kernel TAI offset is zero (as on
// GCE, where leap seconds are smeared and chrony sets no offset). Keeps show times small.
constexpr int64_t kShowEpochSeconds = 1767225600;

namespace clock_origin_detail {

inline int64_t readNs(clockid_t id) {
    struct timespec ts;
    clock_gettime(id, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}

// CLOCK_TAI minus CLOCK_MONOTONIC, from the tightest of a few bracketed reads.
inline int64_t taiMinusMonotonicNs() {
    int64_t best = 0, best_width = INT64_MAX;
    for (int i = 0; i < 9; ++i) {
        const int64_t before = readNs(CLOCK_MONOTONIC);
#ifdef CLOCK_TAI
        const int64_t tai = readNs(CLOCK_TAI);
#else
        const int64_t tai = readNs(CLOCK_REALTIME);
#endif
        const int64_t after = readNs(CLOCK_MONOTONIC);
        if (after - before < best_width) {
            best_width = after - before;
            best = tai - (before + (after - before) / 2);
        }
    }
    return best;
}

inline bool enabled() {
    const char *mode = std::getenv("AVP_CLOCK_ORIGIN");
    return mode && std::strcmp(mode, "tai") == 0;
}

} // namespace clock_origin_detail

inline bool clockOriginEnabled() {
    static const bool on = clock_origin_detail::enabled();
    return on;
}

/// Show time minus CLOCK_MONOTONIC in ns; constant for the life of the process. Zero when off.
inline int64_t clockOriginNs() {
    static const int64_t origin = clockOriginEnabled()
        ? clock_origin_detail::taiMinusMonotonicNs() - kShowEpochSeconds * 1000000000
        : 0;
    return origin;
}

/// How far the system clock has moved against the monotonic clock since the sample: a step of
/// the system clock (chrony makestep, a manual set) shows here, and so would a suspended VM.
/// Slewing does not: both clocks follow it.
inline int64_t clockOriginDriftNs() {
    if (!clockOriginEnabled()) return 0;
    return clock_origin_detail::taiMinusMonotonicNs() - kShowEpochSeconds * 1000000000 - clockOriginNs();
}

struct KernelClockState {
    bool synchronized = false;  // the kernel's own flag (STA_UNSYNC clear), what timedatectl shows
    int tai_offset = 0;         // seconds CLOCK_TAI runs ahead of CLOCK_REALTIME
    int64_t est_error_us = 0;
    int64_t max_error_us = 0;
};

inline KernelClockState kernelClockState() {
    KernelClockState state;
#ifdef __linux__
    struct timex tx;
    std::memset(&tx, 0, sizeof tx);   // modes = 0: a read, allowed without privileges
    if (adjtimex(&tx) < 0) return state;
    state.synchronized = !(tx.status & STA_UNSYNC);
    state.tai_offset = tx.tai;
    state.est_error_us = tx.esterror;
    state.max_error_us = tx.maxerror;
#endif
    return state;
}

/// A small ring of (key, monotonic ns) pairs a stage of the graph leaves behind when
/// AVP_SYNC_TRACE=1, read through node.object.get <node> sync_trace: which timestamp passed the
/// stage when. Two hosts' rings joined on the key give the stage's offset between them.
class SyncTrace {
public:
    struct Entry { int64_t key; int64_t at_ns; std::vector<int64_t> detail; };

    static bool enabled() {
        static const bool on = [] {
            const char *value = std::getenv("AVP_SYNC_TRACE");
            return value && std::strcmp(value, "1") == 0;
        }();
        return on;
    }

    void add(int64_t key, int64_t at_ns, std::vector<int64_t> detail = {}) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto &slot = ring_[next_++ % ring_.size()];
        slot.key = key;
        slot.at_ns = at_ns;
        slot.detail = std::move(detail);
    }

    /// Oldest first.
    std::vector<Entry> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<Entry> out;
        const size_t count = next_ < ring_.size() ? next_ : ring_.size();
        out.reserve(count);
        for (size_t i = next_ - count; i < next_; ++i) out.push_back(ring_[i % ring_.size()]);
        return out;
    }

private:
    mutable std::mutex mutex_;
    std::array<Entry, 512> ring_;
    size_t next_ = 0;
};

/// Traces of nodes that cannot answer node.object.get, by the label their "sync_trace"
/// parameter gives; read with the sync.trace command.
inline std::shared_ptr<SyncTrace> namedSyncTrace(const std::string &label, bool create = true) {
    static std::mutex mutex;
    static std::map<std::string, std::shared_ptr<SyncTrace>> traces;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = traces.find(label);
    if (it != traces.end()) return it->second;
    if (!create) return nullptr;
    return traces[label] = std::make_shared<SyncTrace>();
}

inline int64_t monotonicNowNs() { return clock_origin_detail::readNs(CLOCK_MONOTONIC); }

} // namespace avp
