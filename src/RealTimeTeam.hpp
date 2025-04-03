#pragma once
#include <atomic>
#include <mutex>
#include "instance_shared.hpp"
#include "avutils.hpp"

class RealTimeTeam: public InstanceShared<RealTimeTeam>, public IFlushAndSeek {
protected:
    std::atomic<AVTS> offset_{AV_NOPTS_VALUE};
    std::mutex busy_;
    std::unique_lock<decltype(busy_)> getLock() {
        return std::unique_lock<decltype(busy_)>(busy_);
    }
    AVRational timebase_ = {0, 0};
    std::atomic_bool flushing_ = false;

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
    AVTS updateOffset(AVTS local_offset) {
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
    void reset() {
        auto lock = getLock();
        offset_.store(AV_NOPTS_VALUE, std::memory_order_relaxed);
        logstream << "realtime team reset";
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
    virtual void flushAndSeek(StreamTarget seek_target) override {
        std::unique_lock<decltype(seek_mutex_)>(seek_mutex_);
        StreamTarget target = seek_target;
        int64_t current_wallclock = -1;
        int input_idx = -1;

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

                if (target.isFrameRelative()) {
                    target.type = StreamTarget::ETargetType::tt_Wallclock;
                    target.ts = av::Timestamp(current_wallclock, {1, 1000});

                    if (target.frame_number != 0) {
                        auto pNode = std::dynamic_pointer_cast<Node>(node);
                        if (pNode) {
                            std::shared_ptr<IPlaybackControl> streams_in = pNode->sourceEdge()->findNodeUp<IPlaybackControl>();
                            if (streams_in) {
                                streams_in->offsetStreamTargetByFrames(target, target.frame_number);
                            }
                        }
                    }
                }
                if (target.isTimestampRelative()) {
                    target.type = StreamTarget::ETargetType::tt_Wallclock;
                    target.ts = addTS(target.ts, av::Timestamp(current_wallclock, {1, 1000}));
                }
                break;
            }
        }

        if ((input_idx < 0) && target.isRelative()) {
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
    }
    void addSeekTarget(std::weak_ptr<IFlushAndSeek> target) {
        std::unique_lock<decltype(seek_mutex_)>(seek_mutex_);
        seek_targets_.push_back(target);
    }
};
