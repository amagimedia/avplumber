#include "soft_join_common.hpp"

// soft_join_metadata_frame:
// Hold-last join of auxiliary MetadataFrame streams onto a MetadataFrame primary.
// Primary PTS drives emission; empty aux queues do not block.
// Alias: soft_join_metadata

class SoftJoinMetadataFrame: public NodeMultiInput<MetadataFrame>,
                             public NodeSingleOutput<MetadataFrame>,
                             public ReportsFinishByFlag {
private:
    static constexpr int primary_index_ = 0;
    static constexpr const char* kNodeName = "soft_join_metadata_frame";

    bool primary_eof_ = false;
    std::vector<soft_join::AuxState> aux_;
    double max_age_sec_ = -1.0;

    void consumeEofMarkers() {
        if (!primary_eof_) {
            MetadataFrame* primary = this->source_edges_[primary_index_]->peek();
            if (primary && isEofMarker(*primary)) {
                this->source_edges_[primary_index_]->pop();
                primary_eof_ = true;
            }
        }
        for (size_t i = 0; i < aux_.size(); ++i) {
            if (aux_[i].eof) continue;
            MetadataFrame* frame = this->source_edges_[primary_index_ + 1 + i]->peek();
            if (frame && isEofMarker(*frame)) {
                this->source_edges_[primary_index_ + 1 + i]->pop();
                aux_[i].eof = true;
            }
        }
    }

    std::vector<std::shared_ptr<Edge<MetadataFrame>>> auxEdges() const {
        std::vector<std::shared_ptr<Edge<MetadataFrame>>> edges;
        edges.reserve(aux_.size());
        for (size_t i = 0; i < aux_.size(); ++i) {
            edges.push_back(this->source_edges_[primary_index_ + 1 + i]);
        }
        return edges;
    }

    MetadataFrame mergeIntoPrimary(const MetadataFrame& primary) const {
        Parameters md = primary.metadata();
        soft_join::mergeHeldIntoParameters(md, aux_);
        return MetadataFrame(primary.pts(), std::move(md));
    }

public:
    using NodeSingleOutput<MetadataFrame>::NodeSingleOutput;

    bool consumeEofIfPresent() override {
        return false;
    }

    virtual void process() override {
        consumeEofMarkers();

        if (this->stopping_) {
            this->finished_ = true;
            return;
        }

        if (primary_eof_ && soft_join::allAuxEof(aux_)) {
            this->sink_->put(MetadataFrame());
            this->finished_ = true;
            return;
        }

        if (primary_eof_) {
            bool any_present = false;
            for (size_t i = 0; i < aux_.size(); ++i) {
                if (aux_[i].eof) continue;
                MetadataFrame* frame = this->source_edges_[primary_index_ + 1 + i]->peek();
                if (frame == nullptr) continue;
                any_present = true;
                if (isEofMarker(*frame)) {
                    this->source_edges_[primary_index_ + 1 + i]->pop();
                    aux_[i].eof = true;
                } else {
                    this->source_edges_[primary_index_ + 1 + i]->pop();
                }
            }
            if (!any_present && !soft_join::allAuxEof(aux_)) {
                this->waitForInput();
            }
            return;
        }

        MetadataFrame* primary = this->source_edges_[primary_index_]->peek();
        if (primary == nullptr) {
            this->waitForInput();
            return;
        }
        if (isEofMarker(*primary)) {
            consumeEofMarkers();
            return;
        }

        if (!soft_join::isUsableMetadata(*primary)) {
            this->sink_->put(*primary);
            this->source_edges_[primary_index_]->pop();
            return;
        }

        auto edges = auxEdges();
        soft_join::drainAuxForPrimary(edges, aux_, primary->pts(), max_age_sec_);
        this->sink_->put(mergeIntoPrimary(*primary));
        this->source_edges_[primary_index_]->pop();
    }

    static std::shared_ptr<SoftJoinMetadataFrame> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;

        auto src_names = jsonToStringList(params["src"]);
        if (src_names.size() < 2) {
            throw Error(std::string(kNodeName) +
                        " requires src: [primary, aux...] with at least one auxiliary");
        }
        const size_t aux_count = src_names.size() - 1;
        auto into_keys = soft_join::parseIntoKeys(params, aux_count, kNodeName);
        const double max_age = soft_join::parseMaxAge(params, kNodeName);

        std::shared_ptr<Edge<MetadataFrame>> out_edge = edges.find<MetadataFrame>(params["dst"]);
        auto result = std::make_shared<SoftJoinMetadataFrame>(make_unique<EdgeSink<MetadataFrame>>(out_edge));
        result->auto_eof_ = false;
        result->max_age_sec_ = max_age;
        result->aux_.resize(aux_count);
        for (size_t i = 0; i < aux_count; ++i) {
            result->aux_[i].into = into_keys[i];
        }
        result->createSourcesFromParameters(edges, params);
        out_edge->setProducer(result);
        return result;
    }
};

DECLNODE(soft_join_metadata_frame, SoftJoinMetadataFrame);
DECLNODE_ALIAS(soft_join_metadata, SoftJoinMetadataFrame);
