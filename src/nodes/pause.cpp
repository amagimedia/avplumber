#include "node_common.hpp"
#include "../PauseControlTeam.hpp"

template<typename T> class Pause: public NodeSISO<T, T>, public NonBlockingNode<Pause<T>>, public IInputReset {
protected:
    std::shared_ptr<PauseControlTeam> team_;
    Event wake_paused_;
    std::atomic_bool pass_single_ {false};
public:
    using NodeSISO<T, T>::NodeSISO;
    virtual void processNonBlocking(EventLoop& evl, bool ticks) override {
        if (team_->isPaused() && !pass_single_) {
            this->source_->peek(0); // need to call this to consume packets when in flushing state
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

            // put it in the sink queue:
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
        return r;
    }
};

DECLNODE_ATD(pause, Pause);