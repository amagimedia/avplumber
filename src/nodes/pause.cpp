#include "node_common.hpp"
#include "../graph_interfaces.hpp"
#include "../PauseControlTeam.hpp"
#include "../RealTimeTeam.hpp"

template<typename T> class Pause: public NodeSISO<T, T>, public NonBlockingNode<Pause<T>>, public IInputReset {
protected:
    std::shared_ptr<PauseControlTeam> team_;
    Event wake_paused_;
    std::atomic_bool pass_single_ {false};
public:
    using NodeSISO<T, T>::NodeSISO;
    virtual void processNonBlocking(EventLoop& evl, bool ticks) override {
        if (team_->isPaused() && !pass_single_) {
            logstream << "PAUSE peek & dropping!";
            auto p = this->source_->peek(0); // need to call this to consume packets when in flushing state
            if (p) {
                //if (p->streamIndex() == 0) {
                    logstream << "PAUSE peek & dropping!: " << p->streamIndex() << "   " << rescaleTS(p->pts(), {1, 1000});
                //}
            }
            if (!ticks) {
                this->processWhenSignalled(wake_paused_);
            }
            return;
        }

        // TODO DRY: all NonBlockingNodes
        bool process_next;
        do {
            process_next = false;
            T* dataptr = this->source_->peek(0);
            if (dataptr==nullptr) {
                logstream << "PAUSE data not available!";
                // no data available in queue
                if (!ticks) {
                    // retry when we have packet in source queue
                    this->processWhenSignalled(this->edgeSource()->edge()->producedEvent());
                }
                // if ticks==true, processNonBlocking will be called automatically with next tick
                // no need to schedule it
                return;
            }
            
            T &pkt = *dataptr;

            if (pkt.streamIndex() == 0) {
                logstream << "Video pause frame RECV: " << rescaleTS(pkt.pts(), {1,1000});
            }


            // put it in the sink queue:
            if (pkt.streamIndex() == 0) {
                logstream << "Video pause frame PUT: " << rescaleTS(pkt.pts(), {1,1000});
            }
            if (this->sink_->put(pkt, true)) {
                // put returned true, success, remove this packet from the source queue
                av::Timestamp pkt_ts = pkt.pts();
                this->source_->pop();
                pass_single_ = false;
                if (!ticks) {
                    // process next packet
                    this->yieldAndProcess();
                    team_->checkPause(pkt_ts);
                } else {
                    process_next = !team_->checkPause(pkt_ts);
                }
            } else {
                // put returned false, no space in queue
                if (!ticks) {
                    // retry when we have space in sink
                    this->processWhenSignalled(this->edgeSink()->edge()->consumedEvent());
                }
            }
        } while (process_next);
    }
    virtual void resetInput() override {
        logstream << "PAUSE " << this->name_ << " reset INPUT";
        pass_single_ = true;
        wake_paused_.signal();
    }
    static std::shared_ptr<Pause> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto r = NodeSISO<T, T>::template createCommon<Pause<T>>(edges, params);
        std::string team = "default";
        if (params.count("team")) {
            team = params["team"];
        }
        r->team_ = InstanceSharedObjects<PauseControlTeam>::get(nci.instance, team);
        r->team_->addNode(std::weak_ptr<IInputReset>(r));
        if (params.count("paused")) {
            if (params["paused"].get<bool>()) {
                r->team_->pause();
                r->pass_single_ = true;
            }
        }

        if (edges.exists<av::VideoFrame>(params["src"])) {
            auto in_edge = edges.find<av::VideoFrame>(params["src"]);
            std::weak_ptr<IPlaybackControl> streams_in = in_edge->findNodeUp<IPlaybackControl>();
            r->team_->setPlaybackNode(streams_in);
            if (params.count("sync_team")) {
                std::shared_ptr<RealTimeTeam> sync_team = InstanceSharedObjects<RealTimeTeam>::get(nci.instance, params["sync_team"]);
                if (sync_team) {
                    r->team_->addSyncObj(sync_team);
                }
            }
        }
        return r;
    }
};

DECLNODE_ATD(pause, Pause);