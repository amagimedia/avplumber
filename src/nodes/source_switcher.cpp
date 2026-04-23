#include "node_common.hpp"
#include "../SharedTimeline.hpp"

template <typename T>
class MixerSourceSwitcher : public NodeMultiInput<T>, public NodeSingleOutput<T>,
                            public TimelineReader, public IInputsObjects {
    std::atomic<int> active_input_{0};
    std::string label_;
    int last_selected_input_ = -1;
    av::Timestamp last_output_pts_ = NOTS;

public:
    using NodeSingleOutput<T>::NodeSingleOutput;

    virtual void process() override {
        int srci = this->findSourceWithData(0);
        if (srci < 0) {
            this->waitForInput();
            return;
        }

        for (int i = 0; i < (int)this->source_edges_.size(); i++) {
            T* data = this->source_edges_[i]->peek();
            if (!data) continue;
            int active = this->template tlGet<int>("active", data->pts(), active_input_.load(std::memory_order_relaxed));
            if (i == active) {
                if (label_ == "wipe_sel") {
                    if (i != last_selected_input_) {
                        logstream << "source_switcher[" << label_ << "]: selected input " << i
                                  << " at pts " << data->pts();
                    }
                    if (last_output_pts_.isValid() && data->pts() < last_output_pts_) {
                        logstream << "source_switcher[" << label_ << "]: output pts went backwards "
                                  << last_output_pts_ << " -> " << data->pts() << " on input " << i;
                    }
                    last_selected_input_ = i;
                    last_output_pts_ = data->pts();
                }
                this->sink_->put(*data);
            }
            this->source_edges_[i]->pop();
        }
    }

    void setObject(const std::string key, const Parameters& value) override {
        if (key == "active")
            active_input_.store(value.get<int>(), std::memory_order_relaxed);
    }

    static std::shared_ptr<MixerSourceSwitcher> create(NodeCreationInfo& nci) {
        std::shared_ptr<Edge<T>> out_edge = nci.edges.find<T>(nci.params["dst"]);
        auto r = std::make_shared<MixerSourceSwitcher>(make_unique<EdgeSink<T>>(out_edge));
        r->createSourcesFromParameters(nci.edges, nci.params);
        out_edge->setProducer(r);
        r->initTimeline(nci);
        if (nci.params.count("name"))
            r->label_ = nci.params["name"].get<std::string>();
        else if (nci.params.count("dst"))
            r->label_ = nci.params["dst"].get<std::string>();
        else
            r->label_ = "<unnamed>";
        if (nci.params.count("active"))
            r->active_input_.store(nci.params["active"].get<int>(), std::memory_order_relaxed);
        return r;
    }
};

DECLNODE_ATD(source_switcher, MixerSourceSwitcher);
