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
    std::vector<std::weak_ptr<IFlushAndSeek>> nodes_;
    std::weak_ptr<IFlushAndSeek> sync_obj_;
    std::unique_lock<decltype(mutex_)> getLock() {
        return std::unique_lock<decltype(mutex_)>(mutex_);
    }
public:
    void setSpeed(float speed) {
        auto lock = getLock();

        if (!last_sync_.isNoPts()) {
            av::Timestamp elapsed = addTS(last_pts_, negateTS(last_sync_));
            av::Timestamp scaled = { AVTS(std::round(double(elapsed.timestamp()) * double(inv_speed_))), elapsed.timebase() };
            shift_ = addTS(shift_, last_sync_, scaled, negateTS(last_pts_));
        }

        bool direction_changed = std::signbit(speed_) != std::signbit(speed);
        speed_ = speed;
        inv_speed_ = 1.0f/speed;
        last_sync_ = last_pts_;

        if (direction_changed) {
            for (auto n: nodes_) {
                auto node = n.lock();
                if (node) {
                    auto pNode = std::dynamic_pointer_cast<Node>(node);
                    // set input playback direction
                    if (pNode) {
                        std::shared_ptr<IStreamsInput> streams_in = pNode->sourceEdge()->findNodeUp<IStreamsInput>();
                        if (streams_in) {
                            streams_in->setPlaybackDirection(speed > 0 ? IStreamsInput::EPlaybackDirection::pd_Forward : IStreamsInput::EPlaybackDirection::pd_Backward);
                        }
                    }
                }
            }
            // seek to current frame
            auto obj = sync_obj_.lock();
            if (obj) {
                obj->flushAndSeek(StreamTarget::from_timestamp(last_pts_));
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
    void addNode(std::weak_ptr<IFlushAndSeek> node) {
        auto lock = getLock();
        nodes_.push_back(node);
    }
    void setSyncObj(std::weak_ptr<IFlushAndSeek> obj) {
        auto lock = getLock();
        sync_obj_ = obj;
    }
};
