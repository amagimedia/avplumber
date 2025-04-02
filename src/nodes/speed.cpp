#include "node_common.hpp"
#include "../SpeedControlTeam.hpp"
#include "../RealTimeTeam.hpp"
#include "../graph_base.hpp"

template<typename T> class Speed: public NodeSISO<T, T>, public NonBlockingNode<Speed<T>>, public ISpeed {
protected:
    std::shared_ptr<SpeedControlTeam> team_;
    av::Rational timebase_ {0, 0};
    std::weak_ptr<NodeWrapper> sync_node_;
    bool discard_when_speed_changed_ = false;
public:
    using NodeSISO<T, T>::NodeSISO;
    virtual void processNonBlocking(EventLoop& evl, bool ticks) override {
        bool process_next;
        do {
            process_next = false;
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
            
            if (!out_pts.isNoPts()) {
                // put it in the sink queue:
                if (this->sink_->put(frame, true)) {
                    // put returned true, success, remove this packet from the source queue
                    this->source_->pop();
                    team_->setLastPTS(orig_pts);
                    if (!ticks) {
                        // process next packet
                        this->yieldAndProcess();
                    } else {
                        process_next = true;
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
        } while (process_next);
    }
    void speedChanged() override {
        this->lockProcessing();
        auto n = sync_node_.lock();
        if (n) {
            auto node = n->node();
            if (node) {
                auto nsi = std::dynamic_pointer_cast<NodeSingleInput<T>>(node);
                if (nsi) {
                    std::shared_ptr<EdgeBase> edge = nsi->sourceEdge();
                    while (edge) {
                        auto prod = edge->producer().lock();
                        logstream << prod->name_ << " E " << edge << " T " << this;
                        if (prod.get() == this) {
                            logstream << "DONE!";
                            break;
                        }
                        if (prod) {
                            edge = prod->sourceEdge();
                        } else {
                            break;
                        }
                    }
                }
            }
            
            // pause & lock all nodes
            
        }
        this->unlockProcessing();
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
        r->team_->addNode(r);
        if (params.count("speed")) {
            double speed = params["speed"].get<double>();
            r->team_->setSpeed(speed);
        }
        if (params.count("sync_team")) {
            std::shared_ptr<RealTimeTeam> sync_team = InstanceSharedObjects<RealTimeTeam>::get(nci.instance, params["sync_team"]);
            if (sync_team) {
                r->team_->setSyncObj(sync_team);
            }
        }
        if (params.count("sync_node")) {
             r->sync_node_ = nci.nodes.node(params["sync_node"]);
        }
        return r;
    }
};

class VideoSpeed: public Speed<av::VideoFrame> {};
class AudioSpeed: public Speed<av::AudioSamples> {};

DECLNODE(speed_video, VideoSpeed);
DECLNODE(speed_audio, AudioSpeed);
