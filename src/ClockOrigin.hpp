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
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <libavutil/rational.h>
}

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

/// The unit two hosts exchange realtime offsets in (realtime.team.sync, a realtime node's "sync"):
/// 1/120000 s. Every rate a show runs at is a whole number of these per frame (25: 4800,
/// 29.97: 4004, 30: 4000, 50: 2400, 59.94: 2002, 60: 2000, 23.976: 5005), and so is a 1024-sample
/// 48 kHz audio frame (2560), so the number crosses hosts exactly whatever unit each team counts in.
/// Nanoseconds would not do: a 29.97 frame is 33366666.67 ns, and a rounded offset can move a frame
/// that sits on a half-tick boundary (60 fps into 30, 59.94 into 29.97) to the other tick on one host.
constexpr AVRational kShareTimeBase{1, 120000};

/// A show's frame rate as written for host sync: a whole rate ("30") or an exact NTSC rate
/// ("30000/1001"). A decimal is refused: av_parse_ratio would read "29.97" as 2997/100, a different
/// rate whose ticks drift from the program's, so two hosts would slot frames a tick apart.
inline AVRational parseShowRate(const std::string &text) {
    const auto slash = text.find('/');
    const std::string num_text = text.substr(0, slash), den_text = slash == std::string::npos ? "1" : text.substr(slash + 1);
    auto digits = [](const std::string &s) { return !s.empty() && s.size() <= 9 && s.find_first_not_of("0123456789") == std::string::npos; };
    if (!digits(num_text) || !digits(den_text))
        throw std::invalid_argument("show rate \"" + text + "\" must be a whole rate (\"30\") or an exact NTSC rate (\"30000/1001\")");
    AVRational rate{std::stoi(num_text), std::stoi(den_text)};
    if (rate.num <= 0 || rate.den <= 0) throw std::invalid_argument("show rate \"" + text + "\" must be positive");
    av_reduce(&rate.num, &rate.den, rate.num, rate.den, INT32_MAX);
    if (rate.den != 1 && rate.den != 1001)
        throw std::invalid_argument("show rate \"" + text + "\" must be a whole rate or N/1001 (\"30000/1001\", \"60000/1001\")");
    return rate;
}

/// Whole ticks at *rate* between this host's monotonic zero and the show epoch: show tick =
/// local tick + this. Zero when the clock origin is off.
inline int64_t showZeroTicks(AVRational rate) {
    const __int128 origin = clockOriginNs();
    const __int128 per_tick_ns = static_cast<__int128>(rate.den) * 1000000000;
    __int128 ticks = origin * rate.num / per_tick_ns;
    if ((origin * rate.num) % per_tick_ns < 0) --ticks;
    return static_cast<int64_t>(ticks);
}

/// What two hosts of one show must agree on, per avplumber instance (InstanceSharedObjects
/// <avp::ShowSync>, id "show_sync"): the rate show ticks count at and the program's playout
/// latency. realtime.team.sync folds the hosts' clock difference in ticks of its rate, mixer
/// at_show_tick turns show ticks into the program's, outputs stamp show time: all must use the
/// program's rate, or two hosts put a frame in different ticks (a sync given "30" for a
/// 30000/1001 program). mixer.init registers the program's rate; a different one is refused.
class ShowSync {
    mutable std::mutex mutex_;
    AVRational rate_{0, 0};
    std::string rate_by_;
    int64_t latency_ns_ = -1;
public:
    void registerRate(AVRational rate, const std::string &who) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (rate_.den == 0) {
            rate_ = rate;
            rate_by_ = who;
            return;
        }
        if (av_cmp_q(rate_, rate) != 0)
            throw std::invalid_argument(who + " runs at " + std::to_string(rate.num) + "/" + std::to_string(rate.den) +
                                        " fps but the show ticks at " + std::to_string(rate_.num) + "/" +
                                        std::to_string(rate_.den) + " (" + rate_by_ +
                                        "); host sync, at_show_tick and show-time outputs need the program's rate");
    }
    /// {0, 0} until registered.
    AVRational rate() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return rate_;
    }
    /// The program compositor's playout latency; -1 until registered. Two hosts with different
    /// latencies show every frame that many ns apart, so host sync compares it.
    void registerLatency(int64_t ns) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (latency_ns_ < 0) latency_ns_ = ns;
    }
    int64_t latencyNs() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return latency_ns_;
    }
};

} // namespace avp
