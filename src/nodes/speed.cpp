#include "node_common.hpp"
#include "../SpeedControlTeam.hpp"
#include "../RealTimeTeam.hpp"
#include "../graph_base.hpp"

template<typename T> class Speed: public NodeSISO<T, T>, public NonBlockingNode<Speed<T>>, public ISpeed, public IReturnsObjects {
protected:
    std::shared_ptr<SpeedControlTeam> team_;
    av::Rational timebase_ {0, 0};
    std::weak_ptr<NodeWrapper> sync_node_;
    bool sync_node_tick_ = false;
    bool discard_when_speed_changed_ = false;
    std::list<std::tuple<int64_t, av::Timestamp>> scaled_pts_;
public:
    using NodeSISO<T, T>::NodeSISO;

    bool rescaleFrameTS(T* frame) {
        av::Timestamp orig_pts = frame->pts();
        av::Timestamp in_pts = orig_pts;
        if (timebase_.getNumerator() && timebase_.getDenominator()) {
            in_pts = rescaleTS(in_pts, timebase_);
        }
        av::Timestamp out_pts = team_->scalePTS(in_pts, discard_when_speed_changed_);

        if (frame->raw()->sample_rate != 0) {
            // Try to read sample_rate from frame metadata, if present
            auto sample_rate_entry = av_dict_get(frame->raw()->metadata, "sample_rate", nullptr, 0);
            if (sample_rate_entry) {
                int orig_sample_rate = std::atoi(sample_rate_entry->value);
                frame->raw()->sample_rate = int(float(orig_sample_rate) * team_->getSpeed() + 0.5);
            }
        }

        if (out_pts.isValid()) {
            frame->setTimeBase(av::Rational());
            frame->setPts(out_pts);
            return true;
        }
        return false;
    }

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
            int orig_sample_rate = frame.raw()->sample_rate;

            if (!orig_pts.isNoPts()) {
                if (!rescaleFrameTS(&frame)) {
                    // frame dropped
                    this->source_->pop();
                    if (!ticks) {
                        // process next packet
                        this->yieldAndProcess();
                    } else {
                        process_next = true;
                    }
                    return;
                }
                // put it in the sink queue:
                if (this->sink_->put(frame, true)) {
                    logstream << "SSS/put = " << frame.pts().timestamp({1, 1000});
                    // store orifinal frame PTS
                    auto frame_ts = av_dict_get(frame.raw()->metadata, "frame_ts", nullptr, 0);
                    if (frame_ts) {
                        if (scaled_pts_.size() > 8)
                            scaled_pts_.pop_front();
                        int64_t f_ts = std::atoll(frame_ts->value);
                        scaled_pts_.emplace_back(std::make_tuple(f_ts, orig_pts));
                    }
 
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
                    if (frame.raw()) {
                        frame.raw()->sample_rate = orig_sample_rate;
                    }
                    if (!ticks) {
                        // retry when we have space in sink
                        this->processWhenSignalled(this->edgeSink()->edge()->consumedEvent());
                    }
                }
            } else if (isEofMarker(frame)) {
                // EOF frame
                if (this->sink_->put(frame, true)) {
                    logstream << "SSS/eof = " << frame.pts().timestamp({1, 1000});
                    this->source_->pop();
                    if (!ticks) {
                        // process next packet
                        this->yieldAndProcess();
                    } else {
                        process_next = true;
                    }
                } else {
                    // put returned false, no space in queue
                    if (!ticks) {
                        // retry when we have space in sink
                        this->processWhenSignalled(this->edgeSink()->edge()->consumedEvent());
                    }
                }
            }
        } while (process_next);
    }
    Parameters getObject(const std::string name) override {
        if (name == "info") {
            Parameters res;
            res["speed"] = team_->getSpeed();
            return res;
        }

        throw Error("Unknown object to get");
    }
    void speedChanged() override {
        auto n = sync_node_.lock();
        if (!n)
            return;
        auto node = n->node();
        if (!node)
            return;
        auto nsi = std::dynamic_pointer_cast<NodeSingleInput<T>>(node);
        if (!nsi)
            return;
        auto ifs = std::dynamic_pointer_cast<IFrameTimestamp>(node);
        if (!ifs)
            return;
        int64_t current_wc = ifs->getCurrentFrameTimestamp();
        if (current_wc < 0)
            return;

        av::Timestamp current_ts;
        this->lockProcessing();
        for (auto& p: scaled_pts_) {
            if (std::get<0>(p) == current_wc) {
                current_ts = std::get<1>(p);
                break;
            }
        }
        if (!current_ts.isValid()) {
            logstream << "unable to rescale frames";
            this->unlockProcessing();
            return;
        }

        nsi->lockProcessing();

        team_->setLastSync(current_ts);
        auto edge = std::dynamic_pointer_cast<Edge<T>>(nsi->sourceEdge());
        // pause all nodes from sync-node to current node
        while (edge) {
            T* p = edge->peek();
            if (p) {
                auto frame_wc = av_dict_get(p->raw()->metadata, "frame_ts", nullptr, 0);
                if (frame_wc) {
                    int64_t f_wc = atoll(frame_wc->value);
                    // rescale PTS again with current speed
                    for (const auto& it: scaled_pts_) {
                        if (f_wc == std::get<0>(it)) {
                            // found original PTS
                            p->setTimeBase(av::Rational());
                            p->setPts(std::get<1>(it));
                            if (!rescaleFrameTS(p)) {
                                // drop frame
                                edge->pop();
                            }
                            team_->setLastPTS(std::get<1>(it));
                            break;
                        }
                    }
                }
            }
            auto prod = edge->producer().lock();
            if (prod.get() == this) {
                break;
            }
            if (prod) {
                prod->lockProcessing();
                edge = std::dynamic_pointer_cast<Edge<T>>(prod->sourceEdge());
            } else {
                break;
            }
        }
        edge = std::dynamic_pointer_cast<Edge<T>>(nsi->sourceEdge());

        auto p_reset = std::dynamic_pointer_cast<IInputReset>(node);
        if (p_reset) {
            p_reset->resetInput();
        }

        // resume everything
        while (edge) {
            auto prod = edge->producer().lock();
            auto p_reset = std::dynamic_pointer_cast<IInputReset>(prod);
            if (p_reset) {
                p_reset->resetInput();
            }
            if (prod.get() == this) {
                break;
            }
            if (prod) {
                prod->unlockProcessing();
                edge = std::dynamic_pointer_cast<Edge<T>>(prod->sourceEdge());
            } else {
                break;
            }
        }
        nsi->unlockProcessing();
        this->unlockProcessing();

        if (!sync_node_tick_) {
            auto nnbi = std::dynamic_pointer_cast<NonBlockingNodeBase>(node);
            if (nnbi) {
                nnbi->doExecute();
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
            auto n = nci.nodes.node(params["sync_node"]);
            auto p = n->parameters();
            r->sync_node_ = n;
            r->sync_node_tick_ = p.count("tick_source") && (p["tick_source"].get<std::string>() == "obs");
        }
        return r;
    }
};

class VideoSpeed: public Speed<av::VideoFrame> {};
class AudioSpeed: public Speed<av::AudioSamples> {};

DECLNODE(speed_video, VideoSpeed);
DECLNODE(speed_audio, AudioSpeed);
