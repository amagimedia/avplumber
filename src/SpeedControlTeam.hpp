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
    std::unique_lock<decltype(mutex_)> getLock() {
        return std::unique_lock<decltype(mutex_)>(mutex_);
    }
public:
    void setSpeed(float speed) {
        auto lock = getLock();
        speed_ = speed;
        inv_speed_ = 1.0f/speed;
        last_sync_ = last_pts_;
    }
    float getSpeed() {
        auto lock = getLock();
        return speed_;
    }
    void setLastPTS(av::Timestamp pts) {
        auto lock = getLock();
        last_pts_ = pts;
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
        return rescaleTS(addTS(last_sync_, scaled), pts.timebase());
    }
};
