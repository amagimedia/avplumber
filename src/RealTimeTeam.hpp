#pragma once
#include <atomic>
#include <mutex>
#include "instance_shared.hpp"
#include "avutils.hpp"

class RealTimeTeam: public InstanceShared<RealTimeTeam>, public IFlushAndSeek, public ILinkableTeam<RealTimeTeam> {
protected:
    std::atomic<AVTS> offset_{AV_NOPTS_VALUE};
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
        for (auto team: linked_teams_) {
            offset = team->updateOffsetNonRecursive(offset);
        }
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
        for (auto team: linked_teams_) {
            team->resetNonRecursive();
        }
    }
    void setFirstNonRecursive(bool value) {
        auto lock = getLock();
        first_ = value;
    }
    void setFirst(bool value) {
        setFirstNonRecursive(value);
        for (auto team: linked_teams_) {
            team->setFirstNonRecursive(value);
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
        for (auto team: linked_teams_) {
            if (!team->isFirstNonRecursive()) {
                return false;
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

    virtual void flushAndSeek(StreamTarget seek_target) override {
        flushAndSeekNonRecursive(seek_target);
        for (auto team: linked_teams_) {
            team->flushAndSeekNonRecursive(seek_target);
        }
    }

    void flushAndSeekNonRecursive(StreamTarget seek_target) {
        std::unique_lock<decltype(seek_mutex_)>(seek_mutex_);
        StreamTarget target = seek_target;
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

        for (int i = 0; i < seek_targets_.size(); ++i) {
            auto node = seek_targets_[i].lock();
            if (node) {
                node->flushAndSeek_start(target);
            }
        }
        for (int i = 0; i < seek_targets_.size(); ++i) {
            auto node = seek_targets_[i].lock();
            if (node) {
                node->flushAndSeek(target);
            }
        }
        for (int i = 0; i < seek_targets_.size(); ++i) {
            auto node = seek_targets_[i].lock();
            if (node) {
                node->flushAndSeek_finish(target);
            }
        }
        for (int i = 0; i < seek_targets_.size(); ++i) {
            auto node = seek_targets_[i].lock();
            if (node) {
                node->flushAndSeek_complete(target);
            }
        }
    }
    void addSeekTarget(std::weak_ptr<IFlushAndSeek> target) {
        std::unique_lock<decltype(seek_mutex_)>(seek_mutex_);
        seek_targets_.push_back(target);
        earliest_stream_.reset();
    }
};
