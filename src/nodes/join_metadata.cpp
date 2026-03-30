#include "node_common.hpp"

template <typename T> class JoinMetadata: public NodeMultiInput<T>, public NodeSingleOutput<T> {
private:
    static constexpr int primary_index_ = 0;
    static constexpr int auxiliary_index_ = 1;

    static bool isUsable(const T& data) {
        return (!data.isNull()) && data.isComplete() && data.pts().isValid();
    }

public:
    using NodeSingleOutput<T>::NodeSingleOutput;

    virtual void process() override {
        T* primary = this->source_edges_[primary_index_]->peek();
        T* auxiliary = this->source_edges_[auxiliary_index_]->peek();

        // Keep synchronization mux-like: do not infer "missing" from empty queues.
        if (primary == nullptr || auxiliary == nullptr) {
            this->waitForInput();
            return;
        }

        if (!isUsable(*primary)) {
            this->sink_->put(*primary);
            this->source_edges_[primary_index_]->pop();
            return;
        }
        if (!isUsable(*auxiliary)) {
            this->source_edges_[auxiliary_index_]->pop();
            return;
        }

        av::Timestamp ts_primary = primary->pts();
        av::Timestamp ts_auxiliary = auxiliary->pts();

        if (ts_primary == ts_auxiliary) {
            av_dict_copy(&primary->raw()->metadata, auxiliary->raw()->metadata, 0);
            this->sink_->put(*primary);
            this->source_edges_[primary_index_]->pop();
            this->source_edges_[auxiliary_index_]->pop();
        } else if (ts_primary < ts_auxiliary) {
            this->sink_->put(*primary);
            this->source_edges_[primary_index_]->pop();
        } else {
            this->source_edges_[auxiliary_index_]->pop();
        }
    }

    static std::shared_ptr<JoinMetadata> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;

        auto src_names = jsonToStringList(params["src"]);
        if (src_names.size() != 2) {
            throw Error("join_metadata requires exactly 2 inputs in src: [primary, auxiliary]");
        }

        std::shared_ptr<Edge<T>> out_edge = edges.find<T>(params["dst"]);
        auto result = std::make_shared<JoinMetadata>(make_unique<EdgeSink<T>>(out_edge));
        result->createSourcesFromParameters(edges, params);
        out_edge->setProducer(result);
        return result;
    }
};

DECLNODE_ATD_RAW(join_metadata, JoinMetadata);
