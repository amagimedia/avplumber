#pragma once
#include <atomic>
#include <mutex>
#include <optional>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include "instance_shared.hpp"
#include "avutils.hpp"
#include "ClockOrigin.hpp"

class RealTimeTeam: public InstanceShared<RealTimeTeam>, public IFlushAndSeek, public ILinkableTeam<RealTimeTeam> {
protected:
    std::atomic<AVTS> offset_{AV_NOPTS_VALUE};
    std::atomic<AVTS> user_delay_{0};  // user-configurable delay in timebase units
    // Set from outside (realtime.team.set_offset): the team then keeps this offset whatever
    // arrives when, so that a second host given the matching number releases the same frame at
    // the same instant. A late frame is dropped and an early one waits; nothing re-anchors.
    std::atomic<AVTS> fixed_offset_{AV_NOPTS_VALUE};
    // Host sync (syncHosts): the show this team is synced for, and one frame of its rate in the
    // team's units.
    std::shared_ptr<avp::ShowSync> show_;
    std::atomic<AVTS> sync_tick_{0};
public:
    /// A source's own correction on top of the fixed offset, after its timestamps jumped (an
    /// encoder restart, a playlist's next item): its frames continue one step after the last one
    /// before the jump, later by whole seconds of show frames if they arrived after that. The
    /// source's video node (master) computes it; its audio node adopts it by generation, so the
    /// pair keeps its lip sync. Team units.
    /// `manual`: set by realtime.team.source_shift, applied by every node of the source at once.
    struct SourceShift { AVTS shift = 0; uint64_t generation = 0; bool manual = false; };
protected:
    std::mutex shifts_mutex_;
    std::map<std::string, SourceShift> shifts_;
    std::atomic<int64_t> clock_checked_ns_{0};
    std::atomic<int64_t> clock_drift_ns_{0};
    std::mutex busy_;
    std::unique_lock<decltype(busy_)> getLock() {
        return std::unique_lock<decltype(busy_)>(busy_);
    }
    AVRational timebase_ = {0, 0};
    std::atomic_bool flushing_ = false;
    std::weak_ptr<IPlaybackControl> earliest_stream_;
    bool first_ = true;

    std::mutex seek_mutex_;
    std::vector<std::weak_ptr<IFlushAndSeek>> seek_targets_;
public:
    void checkTimeBase(AVRational tb) {
        auto lock = getLock();
        if (timebase_.num==0 && timebase_.den==0) {
            timebase_ = tb;
        } else {
            if ((tb.num != timebase_.num) || (tb.den != timebase_.den)) {
                throw Error("all realtime nodes in a team must have the same timebase (tick_source)");
            }
        }
    }
    AVTS updateOffsetNonRecursive(AVTS local_offset) {
        auto lock = getLock();
        if (hasFixedOffset()) return fixedOffset();
        AVTS offset = offset_.load(std::memory_order_relaxed);
        // std::memory_order_relaxed because mutexed anyway
        if (offset == AV_NOPTS_VALUE) {
            offset_.store(local_offset, std::memory_order_relaxed);
            return local_offset;
        }
        // we want to synchronize to the smallest offset because it ensures that sufficient data is buffered
        // for smooth playback of all streams
        if (local_offset < offset) {
            logstream << "realtime team changing offset by " << (local_offset-offset);
            offset_.store(local_offset, std::memory_order_relaxed);
            return local_offset;
        } else {
            logstream << "realtime team ignoring offset diff " << (local_offset-offset);
            return offset;
        }
    }
    AVTS updateOffset(AVTS local_offset) {
        AVTS offset = updateOffsetNonRecursive(local_offset);
        {
            for (auto team: getLinkedTeams()) {
                offset = team->updateOffsetNonRecursive(offset);
            }
        }
        offset_.store(offset, std::memory_order_relaxed);
        return offset;
    }
    void resetNonRecursive() {
        auto lock = getLock();
        if (hasFixedOffset()) return;
        offset_.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
        first_ = true;
        logstream << "realtime team reset";
    }
    void reset() {
        resetNonRecursive();
        {
            for (auto team: getLinkedTeams()) {
                team->resetNonRecursive();
            }
        }
    }
    void setFirstNonRecursive(bool value) {
        auto lock = getLock();
        first_ = value;
    }
    void setFirst(bool value) {
        setFirstNonRecursive(value);
        {
            for (auto team: getLinkedTeams()) {
                team->setFirstNonRecursive(value);
            }
        }
    }
    bool isFirstNonRecursive() {
        auto lock = getLock();
        return first_;
    }
    bool isFirst() {
        if (!isFirstNonRecursive()) {
            return false;
        }
        {
            for (auto team: getLinkedTeams()) {
                if (!team->isFirstNonRecursive()) {
                    return false;
                }
            }
        }
        return true;
    }
    AVTS getOffset(AVTS local_offset = AV_NOPTS_VALUE) {
        if (hasFixedOffset()) return fixedOffset();
        AVTS r = offset_.load(std::memory_order_acquire);
        if ((local_offset != AV_NOPTS_VALUE) && (r != AV_NOPTS_VALUE)) {
            if (r < local_offset) {
                logstream << "getting offset from team diff " << (r-local_offset);
            } else if (r > local_offset) {
                logstream << "STRANGE: local offset smaller than team offset by " << (r-local_offset);
            }
        }
        //return r!=AV_NOPTS_VALUE ? r : local_offset;
        return r;
    }
    void startFlushing() {
        flushing_ = true;
    }
    void stopFlushing() {
        flushing_ = false;
    }
    bool isFlushing() {
        return flushing_;
    }
    bool hasFixedOffset() const {
        return fixed_offset_.load(std::memory_order_acquire) != AV_NOPTS_VALUE;
    }
    AVTS fixedOffset() const {
        return fixed_offset_.load(std::memory_order_acquire);
    }
    void setFixedOffset(AVTS offset) {
        {
            std::lock_guard<std::mutex> lock(shifts_mutex_);
            shifts_.clear();
        }
        fixed_offset_.store(offset, std::memory_order_release);
        offset_.store(offset, std::memory_order_release);
        logstream << "realtime team offset fixed at " << offset;
    }
    /// A fixed offset is only as good as the system clock it was derived from: a step of that
    /// clock after start (the monotonic clock does not follow one) leaves this host apart from
    /// the other by the step. Checked every few seconds by whichever member runs; reported, not
    /// corrected, since correcting means a jump in the picture.
    void watchClock() {
        if (!avp::clockOriginEnabled()) return;
        const int64_t now = avp::monotonicNowNs();
        int64_t last = clock_checked_ns_.load(std::memory_order_relaxed);
        if (now - last < 5000000000LL || !clock_checked_ns_.compare_exchange_strong(last, now)) return;
        const int64_t drift = avp::clockOriginDriftNs();
        clock_drift_ns_.store(drift, std::memory_order_relaxed);
        if (drift > 1000000 || drift < -1000000)
            logstream << "ERROR: the system clock moved by " << drift / 1000 << " us against the monotonic clock "
                      << "since start; this host's fixed realtime offset is off by that much";
    }
    int64_t clockDriftNs() const { return clock_drift_ns_.load(std::memory_order_relaxed); }
    /// Team units <-> the shared unit two hosts exchange (avp::kShareTimeBase, 1/120000 s);
    /// exact or refused.
    AVTS toShare(AVTS team_units) {
        const AVRational tb = getTimebase();
        const __int128 num = static_cast<__int128>(avp::kShareTimeBase.den) * tb.num,
                       den = static_cast<__int128>(avp::kShareTimeBase.num) * tb.den;
        const __int128 v = static_cast<__int128>(team_units) * num;
        if (den == 0 || v % den != 0)
            throw Error("realtime sync: the team's time base " + std::to_string(tb.num) + "/" + std::to_string(tb.den) +
                        " does not convert exactly to the shared 1/120000; give its realtime nodes time_base 1/120000");
        return static_cast<AVTS>(v / den);
    }
    AVTS fromShare(AVTS share_units) {
        const AVRational tb = getTimebase();
        const __int128 num = static_cast<__int128>(avp::kShareTimeBase.den) * tb.num,
                       den = static_cast<__int128>(avp::kShareTimeBase.num) * tb.den;
        const __int128 v = static_cast<__int128>(share_units) * den;
        if (num == 0 || v % num != 0)
            throw Error("realtime sync: a shared value is not a whole number of the team's time base " +
                        std::to_string(tb.num) + "/" + std::to_string(tb.den) + "; give its realtime nodes time_base 1/120000");
        return static_cast<AVTS>(v / num);
    }
    /// Make this host release the team's frames in step with another host (realtime.team.sync,
    /// or a realtime node's "sync" parameter). *fps* is the program's rate (avp::parseShowRate:
    /// whole or N/1001), registered in *show*; *shared* and the result are in avp::kShareTimeBase
    /// (1/120000 s) whatever the team counts in. Without `shared` the team keeps the offset it found
    /// on arrival and fixes it; with it, the offset is fixed to match. See the command for the
    /// reasoning; checkPeer compares what else must match.
    AVTS syncHosts(AVRational fps, std::optional<AVTS> shared, std::shared_ptr<avp::ShowSync> show) {
        if (fps.num <= 0 || fps.den <= 0 || (fps.den != 1 && fps.den != 1001))
            throw Error("realtime sync: fps must be a whole rate or N/1001, such as 30000/1001");
        if (!show) throw Error("realtime sync: no show to register the rate in");
        if (!avp::clockOriginEnabled()) throw Error("realtime sync needs AVP_CLOCK_ORIGIN=tai in the environment");
        const auto kernel = avp::kernelClockState();
        if (!kernel.synchronized) throw Error("realtime sync: the kernel reports the system clock unsynchronised");
        const char *expected_tai = std::getenv("AVP_TAI_OFFSET");
        const int tai_offset = expected_tai ? std::atoi(expected_tai) : 0;
        if (kernel.tai_offset != tai_offset)
            throw Error("realtime sync: kernel TAI offset is " + std::to_string(kernel.tai_offset) + " s, expected " +
                        std::to_string(tai_offset) + " (AVP_TAI_OFFSET); hosts that disagree would be that far apart");
        const AVRational tb = getTimebase();
        if (tb.num <= 0 || tb.den <= 0) throw Error("realtime sync: no realtime node has joined the team");
        // One frame must be a whole number of the team's units, checked before anything that
        // depends on when this host booted: otherwise the fold below is whole on some hosts and
        // not on others. 1/120000 (pyplumber's native-rate time base) is whole at every show rate.
        const __int128 tick_num = static_cast<__int128>(fps.den) * tb.den, tick_den = static_cast<__int128>(fps.num) * tb.num;
        if (tick_num % tick_den != 0)
            throw Error("realtime sync: one frame at " + std::to_string(fps.num) + "/" + std::to_string(fps.den) +
                        " fps is not a whole number of the team's time base " + std::to_string(tb.num) + "/" +
                        std::to_string(tb.den) + "; give its realtime nodes time_base 1/120000");
        const AVTS tick = static_cast<AVTS>(tick_num / tick_den);
        try {
            show->registerRate(fps, "realtime.team.sync");
        } catch (const std::exception &e) {
            throw Error(std::string("realtime sync: ") + e.what());
        }
        // Whole ticks between this host's monotonic zero and the show epoch, as team units.
        const AVTS fold = avp::showZeroTicks(fps) * tick;
        AVTS fixed, result;
        if (shared) {
            fixed = fromShare(*shared) + fold;
            result = *shared;
        } else {
            const AVTS local = offset_.load(std::memory_order_acquire);
            if (local == AV_NOPTS_VALUE) throw Error("realtime sync: the team has not released a frame yet");
            fixed = local;
            result = toShare(local - fold);
        }
        show_ = show;
        sync_tick_.store(tick, std::memory_order_release);
        setFixedOffset(fixed);
        return result;
    }
    /// What the other host must match, as the leader prints it after its offset:
    /// "fps=30000/1001 latency_us=66733 delay=0" (latency only once a program compositor exists,
    /// delay in 1/120000 s).
    std::string syncTerms(const avp::ShowSync &show, AVRational fps) {
        const AVRational tb = getTimebase();
        if (tb.num <= 0 || tb.den <= 0) throw Error("realtime sync: no realtime node has joined the team");
        std::string terms = "fps=" + std::to_string(fps.num) + "/" + std::to_string(fps.den);
        if (show.latencyNs() >= 0) terms += " latency_us=" + std::to_string(show.latencyNs() / 1000);
        return terms + " delay=" + std::to_string(toShare(getUserDelayTS()));
    }
    std::string syncTerms() {
        if (!show_) throw Error("realtime sync: the team is not synced");
        return syncTerms(*show_, show_->rate());
    }
    /// Refuse a sync whose other host differs in what decides where a frame lands: the program
    /// rate, its playout latency, the team delay. *peer* holds the leader's syncTerms; a term only
    /// one of the two has is a difference too, unless *unset_here_ok* (a graph built from a script
    /// syncs its realtime nodes before the program compositor that registers the latency exists).
    void checkPeer(const std::map<std::string, std::string> &peer, const avp::ShowSync &show, AVRational fps,
                   bool unset_here_ok = false) {
        std::map<std::string, std::string> own;
        std::istringstream terms(syncTerms(show, fps));
        for (std::string term; terms >> term;) {
            const auto eq = term.find('=');
            own[term.substr(0, eq)] = term.substr(eq + 1);
        }
        for (const auto &key : {"fps", "latency_us", "delay"}) {
            const auto mine = own.find(key), theirs = peer.find(key);
            if (theirs == peer.end() && mine == own.end()) continue;
            if (mine == own.end() && unset_here_ok) continue;
            if (theirs == peer.end() || mine == own.end() || theirs->second != mine->second)
                throw Error(std::string("realtime sync: ") + key + " differs between the hosts: here " +
                            (mine == own.end() ? "unset" : mine->second) + ", on the other host " +
                            (theirs == peer.end() ? "unset" : theirs->second));
        }
    }
    AVTS syncTick() const { return sync_tick_.load(std::memory_order_acquire); }
    SourceShift sourceShift(const std::string &source) {
        std::lock_guard<std::mutex> lock(shifts_mutex_);
        auto it = shifts_.find(source);
        return it == shifts_.end() ? SourceShift{} : it->second;
    }
    /// A new correction for *source*; returns its generation.
    uint64_t publishSourceShift(const std::string &source, AVTS shift, bool manual = false) {
        std::lock_guard<std::mutex> lock(shifts_mutex_);
        auto &entry = shifts_[source];
        entry.shift = shift;
        entry.manual = manual;
        return ++entry.generation;
    }
    /// Every source's correction in 1/120000 s, as the other host is given it.
    std::map<std::string, AVTS> sourceShiftsShared() {
        std::map<std::string, SourceShift> copy;
        {
            std::lock_guard<std::mutex> lock(shifts_mutex_);
            copy = shifts_;
        }
        std::map<std::string, AVTS> out;
        for (const auto &[source, entry] : copy) out[source] = toShare(entry.shift);
        return out;
    }
    void clearFixedOffset() {
        {
            std::lock_guard<std::mutex> lock(shifts_mutex_);
            shifts_.clear();
        }
        sync_tick_.store(0, std::memory_order_release);
        fixed_offset_.store(AV_NOPTS_VALUE, std::memory_order_release);
        logstream << "realtime team offset no longer fixed";
    }
    AVTS getUserDelayTS() const {
        return user_delay_.load(std::memory_order_acquire);
    }
    float getUserDelay() const {
        AVTS delay_ts = user_delay_.load(std::memory_order_acquire);
        if (timebase_.num == 0 || timebase_.den == 0) {
            // Timebase not set yet, return 0
            return 0.0f;
        }
        return (float)delay_ts * (float)timebase_.num / (float)timebase_.den;
    }
    void setUserDelayNonRecursive(float delay_sec) {
        if (timebase_.num == 0 || timebase_.den == 0) {
            throw Error("Cannot set team delay: timebase not initialized yet");
        }
        AVTS delay_ts = AVTS(delay_sec * (float)timebase_.den / (float)timebase_.num + 0.5f);
        user_delay_.store(delay_ts, std::memory_order_release);
    }
    void setUserDelay(float delay_sec) {
        // The delay is part of host sync (checkPeer): changed on one host after it, the hosts
        // drift apart by the change with nothing to say so.
        if (hasFixedOffset())
            throw Error("realtime team delay is part of host sync: clear the offset, set the delay on both hosts, sync again");
        setUserDelayNonRecursive(delay_sec);
        {
            for (auto team: getLinkedTeams()) {
                team->setUserDelayNonRecursive(delay_sec);
            }
        }
    }
    AVRational getTimebase() {
        auto lock = getLock();
        return timebase_;
    }
    size_t getSeekTargetsCount() {
        std::unique_lock<decltype(seek_mutex_)> lock(seek_mutex_);
        size_t count = 0;
        for (const auto& weak_target : seek_targets_) {
            if (!weak_target.expired()) {
                count++;
            }
        }
        return count;
    }

    std::shared_ptr<IPlaybackControl> getEarliestStream() {
        auto earliest_stream = earliest_stream_.lock();
        if (earliest_stream) {
           return earliest_stream;
        }

        av::Timestamp earliest_ts;

        for (const auto& weak_target : seek_targets_) {
            auto target = weak_target.lock();
            if (target) {
                auto pNode = std::dynamic_pointer_cast<Node>(target);
                if (pNode) {
                    std::shared_ptr<IPlaybackControl> streams_in = pNode->sourceEdge()->findNodeUp<IPlaybackControl>();
                    if (streams_in) {
                        StreamTarget start_target = StreamTarget::from_timestamp({2000, {1, 1000}});
                        if (streams_in->convertStreamTarget(start_target, StreamTarget::ETargetType::tt_SyncTime)) {
                            if (earliest_ts.isNoPts() || start_target.ts < earliest_ts) {
                                earliest_ts = start_target.ts;
                                earliest_stream_ = streams_in;
                            }
                        }
                    }
                }
            }
        }

        return earliest_stream_.lock();
    }

    void teamLinked(std::shared_ptr<RealTimeTeam> team, bool synchronize) override {
        if (synchronize) {
            flushAndSeek(StreamTarget::now());
        }
        if (!hasFixedOffset()) offset_ = AV_NOPTS_VALUE;
    }

    virtual void flushAndSeek(StreamTarget seek_target) override {
        StreamTarget target = seek_target;
        {
            std::unique_lock<decltype(seek_mutex_)> lock(seek_mutex_);
            int64_t current_wallclock = -1;
            int input_idx = -1;

            auto streams_in = getEarliestStream();

            for (int i = 0; i < seek_targets_.size(); ++i) {
                auto node = seek_targets_[i].lock();
                if (node) {
                    auto p_time = std::dynamic_pointer_cast<IFrameTimestamp>(node);
                    if (p_time) {
                        current_wallclock = p_time->getCurrentFrameWallclock();
                    }
                    if (current_wallclock < 0)
                        continue;

                    input_idx = i;

                    if (streams_in) {
                        if (seek_target.isRelative()) {
                            target.type = StreamTarget::ETargetType::tt_Wallclock;
                            target.ts = av::Timestamp(current_wallclock, {1, 1000});
                            if (seek_target.isFrameRelative()) {
                                if (seek_target.frame_number != 0) {
                                    streams_in->offsetStreamTargetByFrames(target, seek_target.frame_number);
                                }
                            }
                            if (seek_target.isTimestampRelative()) {
                                target.ts = addTS(seek_target.ts, av::Timestamp(current_wallclock, {1, 1000}));
                            }
                        }
                        if (seek_target.isTimestamp()) {
                            streams_in->convertStreamTarget(target, StreamTarget::ETargetType::tt_SyncTime);
                        }
                    }
                    break;
                }
            }

            if ((input_idx < 0) && seek_target.isRelative()) {
                throw Error("Relative seek not possible. Can not determine current time.");
            }

            flushAndSeekNonRecursive(target);
        }
    }

    void flushAndSeekNonRecursive(StreamTarget target) {
        std::vector<std::shared_ptr<IFlushAndSeek>> seek_targets_shared;

        for (auto t: seek_targets_) {
            auto target_shared = t.lock();
            if (target_shared) {
                seek_targets_shared.push_back(target_shared);
            }
        }
        {
            for (auto team: getLinkedTeams()) {
                std::unique_lock<decltype(team->seek_mutex_)> team_seek_mutex(team->seek_mutex_);
                for (auto t: team->seek_targets_) {
                    auto target_shared = t.lock();
                    if (target_shared) {
                        seek_targets_shared.push_back(target_shared);
                    }
                }
            }
        }

        for (auto t: seek_targets_shared) {
            t->flushAndSeek_start(target);
        }
        for (auto t: seek_targets_shared) {
            t->flushAndSeek(target);
        }
        for (auto t: seek_targets_shared) {
            t->flushAndSeek_finish(target);
        }
        for (auto t: seek_targets_shared) {
            t->flushAndSeek_complete(target);
        }
    }
    void addSeekTarget(std::weak_ptr<IFlushAndSeek> target) {
        std::unique_lock<decltype(seek_mutex_)> lock(seek_mutex_);
        seek_targets_.push_back(target);
        earliest_stream_.reset();
    }
};
