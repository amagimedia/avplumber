#include "node_common.hpp"
#include <functional>

template<typename T> class SourceSwitcher: public NodeMultiInput<T>, public NodeSingleOutput<T> {
protected:
    struct SourceState {
        AVTS last_packet_time = AV_NOPTS_VALUE; // wallclock, milliseconds
        int timeout_ms = -1;
        std::function<void()> on_in;
        std::function<void()> on_out;
    };
    std::vector<SourceState> source_states_;
    int current_source_ = 0;

public:
    using NodeSingleOutput<T>::NodeSingleOutput;

    virtual void start() override {
        AVTS now = wallclock.pts();
        for (SourceState &ss: source_states_) {
            ss.last_packet_time = now;
        }
    }
    virtual void process() override {
        SourceState *cur_src = &source_states_[current_source_];

        AVTS now = wallclock.pts();
        int remaining = cur_src->last_packet_time + cur_src->timeout_ms - now;
        if (remaining < 0) remaining = 0;
        int srci = this->findSourceWithData(remaining);
        now = wallclock.pts();

        if ((srci < 0) || ((cur_src->timeout_ms >= 0) && (now - cur_src->last_packet_time > cur_src->timeout_ms))) {
            // findSourceWithData timed out, or
            // current source timed out (findSourceWithData may return >= 0 when other source is ready)
            int new_source = (current_source_+1) % this->source_edges_.size();
            logstream << "switching source: " << current_source_ << " -> " << new_source;
            if (cur_src->on_out) {
                cur_src->on_out();
            }
            current_source_ = new_source;
            cur_src = &source_states_[current_source_];
            if (cur_src->on_in) {
                cur_src->on_in();
            }
        }

        if (srci < 0) {
            // no source with data available
            return;
        }
        source_states_[srci] = wallclock.pts();
        if (srci == current_source_) {
            T* dataptr = this->source_edges_[srci]->peek();
            T &data = &dataptr;
            this->sink_->put(data);
            this->source_edges_[srci]->pop();
        } else {
            this->source_edges_[srci]->pop();
        }
    }

    static std::shared_ptr<SourceSwitcher> create(NodeCreationInfo &nci) {
        std::shared_ptr<Edge<av::Packet>> out_edge = nci.edges.find<av::Packet>(nci.params["dst"]);
        std::shared_ptr<SourceSwitcher> r = std::make_shared<SourceSwitcher>(make_unique<EdgeSink<av::Packet>>(out_edge));
        r->createSourcesFromParameters(nci.edges, nci.params);
        r->source_states_.resize(r->source_edges_->size());

        const Parameters &src_params = nci.params["src_params"];
        auto srcs = jsonToStringList(nci.params["src"]);
        if (!src_params.is_object()) {
            throw Error("Dictionary { \"key\": { ... object ... } [ , ... ] } excepted in field \"src_params\"");
        }
        
        for (json::const_iterator it = src_params.begin(); it != src_params.end(); ++it) {
            std::string key = it.key();
            int i = 0;
            int srci = -1;
            for (std::string &srckey: srcs) {
                if (key==srckey) {
                    srci = i;
                    break;
                }
                i++;
            }
            if (srci < 0) {
                throw Error("unknown key " + key + " in src_params, no matching source queue (src) found");
            }
            SourceState *src = &r->source_states_[srci];
            const Parameters &par = it.value();
            if (par.count("timeout")) {
                src->timeout_ms = par["timeout"].get<float>() * 1000;
            }
            if (par.count("group")) {
                std::weak_ptr<NodeGroup> wgroup = nci.nodes.group(par["group"]);
                src->on_in = [wgroup]() {
                    std::shared_ptr<NodeGroup> group = wgroup.lock();
                    if (!group) return;
                    group->startNodes();
                };
                src->on_out = [wgroup]() {
                    std::shared_ptr<NodeGroup> group = wgroup.lock();
                    if (!group) return;
                    group->stopNodes();
                };
            }
        }
        
        return r;
    }
};

DECLNODE_ATD(source_switcher, SourceSwitcher);