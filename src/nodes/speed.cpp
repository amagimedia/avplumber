#include "node_common.hpp"
#include "../SpeedControlTeam.hpp"

template<typename T> class Speed: public NodeSISO<T, T>, public NonBlockingNode<Speed<T>> {
protected:
    std::shared_ptr<SpeedControlTeam> team_;
    av::Rational timebase_ {0, 0};
    bool discard_when_speed_changed_ = false;
public:
    using NodeSISO<T, T>::NodeSISO;
    virtual void processNonBlocking(EventLoop& evl, bool ticks) override {
        T* dataptr = this->source_->peek(0);
        if (dataptr==nullptr) {
            // no data available in queue
            if (!ticks) {
                // retry when we have packet in source queue
                this->processWhenSignalled(this->edgeSource()->edge()->producedEvent());
            }
            // if ticks==true, processNonBlocking will be called automatically with next tick
            // no need to schedule it
            return;
        }
        T &frame = *dataptr;

        av::Timestamp orig_pts = frame.pts();
        av::Timestamp in_pts = orig_pts;
        if (timebase_.getNumerator() && timebase_.getDenominator()) {
            in_pts = rescaleTS(in_pts, timebase_);
        }
        av::Timestamp out_pts = team_->scalePTS(in_pts, discard_when_speed_changed_);
        if (out_pts.isValid()) {
            frame.setTimeBase(av::Rational());
            frame.setPts(out_pts);
        }

        // put it in the sink queue:
        if (out_pts.isNoPts() || this->sink_->put(frame, true)) {
            // put returned true, success, remove this packet from the source queue
            this->source_->pop();
            team_->setLastPTS(orig_pts);
            if (!ticks) {
                // process next packet
                this->yieldAndProcess();
            }
        } else {
            // put returned false, no space in queue
            frame.setTimeBase(av::Rational());
            frame.setPts(orig_pts);
            if (!ticks) {
                // retry when we have space in sink
                this->processWhenSignalled(this->edgeSink()->edge()->consumedEvent());
            }
        }
    }
    static std::shared_ptr<Speed> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto r = NodeSISO<T, T>::template createCommon<Speed<T>>(edges, params);
        if (params.count("timebase")) {
            r->timebase_ = parseRatio(params["timebase"]);
        }
        std::string team = "default";
        if (params.count("team")) {
            team = params["team"];
        }
        if (params.count("discard_when_speed_changed")) {
            r->discard_when_speed_changed_ = params["discard_when_speed_changed"];
        }
        r->team_ = InstanceSharedObjects<SpeedControlTeam>::get(nci.instance, team);
        return r;
    }
};

class VideoSpeed: public Speed<av::VideoFrame> {};
class AudioSpeed: public Speed<av::AudioSamples> {};

DECLNODE(speed_video, VideoSpeed);
DECLNODE(speed_audio, AudioSpeed);
