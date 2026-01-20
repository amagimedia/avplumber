#pragma once
#include <atomic>
#include <mutex>
#include "instance_shared.hpp"
#include "avutils.hpp"

class RealTimeTeam: public InstanceShared<RealTimeTeam>, public IFlushAndSeek, public ILinkableTeam<RealTimeTeam> {
protected:
    std::atomic<AVTS> offset_{AV_NOPTS_VALUE};
    std::atomic<AVTS> user_delay_{0};  // user-configurable delay in timebase units
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
        setUserDelayNonRecursive(delay_sec);
        {
            for (auto team: getLinkedTeams()) {
                team->setUserDelayNonRecursive(delay_sec);
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
        offset_ = AV_NOPTS_VALUE;
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
