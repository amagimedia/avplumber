#include "node_common.hpp"
#include "../SharedTimeline.hpp"

template <typename T>
class MixerSourceSwitcher : public NodeMultiInput<T>, public NodeSingleOutput<T>,
                            public TimelineReader, public IInputsObjects {
    std::atomic<int> active_input_{0};

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
            int active = this->tlGet<int>("active", data->pts(), active_input_.load(std::memory_order_relaxed));
            if (i == active)
                this->sink_->put(*data);
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
        if (nci.params.count("active"))
            r->active_input_.store(nci.params["active"].get<int>(), std::memory_order_relaxed);
        return r;
    }
};

DECLNODE_ATD(source_switcher, MixerSourceSwitcher);
