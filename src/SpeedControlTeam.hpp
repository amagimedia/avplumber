#pragma once
#include <mutex>
#include "avutils.hpp"
#include "instance_shared.hpp"

class SpeedControlTeam: public InstanceShared<SpeedControlTeam> {
protected:
    std::mutex mutex_;
    float speed_ = 1;
    float inv_speed_ = 1;
    av::Timestamp last_pts_ = NOTS;
    av::Timestamp last_sync_ = NOTS;
    av::Timestamp shift_ = {0, {1, 1}};
    av::Timestamp previous_shift_ = {0, {1, 1}};
    std::vector<std::weak_ptr<ISpeed>> nodes_;
    std::weak_ptr<IFlushAndSeek> sync_obj_;
    std::unique_lock<decltype(mutex_)> getLock() {
        return std::unique_lock<decltype(mutex_)>(mutex_);
    }
public:
    void setSpeed(float speed) {
        bool direction_changed = false;
        {
            auto lock = getLock();

            if (!last_sync_.isNoPts()) {
                av::Timestamp elapsed = addTS(last_pts_, negateTS(last_sync_));
                av::Timestamp scaled = { AVTS(std::round(double(elapsed.timestamp()) * double(inv_speed_))), elapsed.timebase() };
                previous_shift_ = shift_;
                shift_ = addTS(shift_, last_sync_, scaled, negateTS(last_pts_));
            }

            direction_changed = std::signbit(speed_) != std::signbit(speed);
            speed_ = speed;
            inv_speed_ = 1.0f/speed;
            last_sync_ = last_pts_;

            if (direction_changed) {
                logstream << "playback direction changed to " << ((speed < 0) ? "backward" : "forward");
                for (auto n: nodes_) {
                    auto node = n.lock();
                    if (node) {
                        auto pNode = std::dynamic_pointer_cast<Node>(node);
                        // set input playback direction
                        if (pNode) {
                            std::shared_ptr<IPlaybackControl> streams_in = pNode->sourceEdge()->findNodeUp<IPlaybackControl>();
                            if (streams_in) {
                                streams_in->setPlaybackDirection(speed > 0 ? IPlaybackControl::EPlaybackDirection::pd_Forward : IPlaybackControl::EPlaybackDirection::pd_Backward);
                            }
                        }
                    }
                }
            }
        }

        if (direction_changed) {
            // flush queues
            auto obj = sync_obj_.lock();
            if (obj) {
                obj->flushAndSeek(StreamTarget::now());
            }
        }

        // notify about speed change
        for (auto n: nodes_) {
            auto node = n.lock();
            if (node) {
                node->speedChanged();
            }
        }
    }
    float getSpeed() {
        auto lock = getLock();
        return speed_;
    }
    void setLastPTS(av::Timestamp pts) {
        auto lock = getLock();
        last_pts_ = pts;
        if (last_sync_.isNoPts()) {
            last_sync_ = last_pts_;
        }
    }
    av::Timestamp scalePTS(av::Timestamp pts, bool forbid_changed_speed) {
        auto lock = getLock();
        if (forbid_changed_speed && (speed_ != 1)) {
            return NOTS;
        }
        if (last_sync_.isNoPts()) {
            return pts;
        }
        av::Timestamp elapsed = addTS(pts, negateTS(last_sync_));
        av::Timestamp scaled = { AVTS(std::round(double(elapsed.timestamp()) * double(inv_speed_))), elapsed.timebase() };
        return rescaleTS(addTS(last_sync_, scaled, shift_), pts.timebase());
    }
    void addNode(std::weak_ptr<ISpeed> node) {
        auto lock = getLock();
        nodes_.push_back(node);
    }
    void setSyncObj(std::weak_ptr<IFlushAndSeek> obj) {
        auto lock = getLock();
        sync_obj_ = obj;
    }
    void setLastSync(av::Timestamp s) {
        shift_ = previous_shift_;
        last_sync_ = s;
    }
    void reset() {
        logstream << "speed team reset";
        //last_pts_ = NOTS;
        //last_sync_ = NOTS;
    }
};
