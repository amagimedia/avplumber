#include "../node_common.hpp"

template <typename T> class DelayGenerator: public NodeSISO<T, T> { // TODO IFlushable
protected:
    struct Item {
        AVTS emit_time;
        T data;
    };
    AVTS delay_;
    std::list<Item> delay_line_;
public:
    using NodeSISO<T, T>::NodeSISO;
    virtual void process() {
        AVTS time_to_wait = delay_line_.empty() ? -1 : std::max(0L, delay_line_.front().emit_time - wallclock.pts());
        bool have_input = this->source_->peek(time_to_wait);
        if (have_input) {
            T data = this->source_->get();
            delay_line_.push_back({wallclock.pts() + delay_, data});
        }
        if (delay_line_.empty()) {
            logstream << "BUG: delay_line_ empty (should never happen)";
            return;
        }
        bool can_emit = wallclock.pts() >= delay_line_.front().emit_time;
        if (can_emit) {
            this->sink_->put(delay_line_.front().data);
            delay_line_.pop_front();
        }
    }
    static std::shared_ptr<DelayGenerator> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        std::shared_ptr<DelayGenerator> r = NodeSISO<T, T>::template createCommon<DelayGenerator>(edges, params);
        r->delay_ = wallclock.secondsToAVTS(params["delay"]);
        return r;
    }
};

DECLNODE_ATD(delaygen, DelayGenerator);
