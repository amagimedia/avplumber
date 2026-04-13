#include "node_common.hpp"

template <typename T> class JoinMetadata: public NodeMultiInput<T>, public NodeSingleOutput<T>, public ReportsFinishByFlag {
private:
    static constexpr int primary_index_ = 0;
    static constexpr int auxiliary_index_ = 1;
    bool primary_eof_ = false;
    bool auxiliary_eof_ = false;

    static bool isUsable(const T& data) {
        return (!data.isNull()) && data.isComplete() && data.pts().isValid();
    }

    void consumeEofMarkers() {
        if (!primary_eof_) {
            T* primary = this->source_edges_[primary_index_]->peek();
            if (primary && isEofMarker(*primary)) {
                this->source_edges_[primary_index_]->pop();
                primary_eof_ = true;
            }
        }
        if (!auxiliary_eof_) {
            T* auxiliary = this->source_edges_[auxiliary_index_]->peek();
            if (auxiliary && isEofMarker(*auxiliary)) {
                this->source_edges_[auxiliary_index_]->pop();
                auxiliary_eof_ = true;
            }
        }
    }

public:
    using NodeSingleOutput<T>::NodeSingleOutput;

    virtual void process() override {
        consumeEofMarkers();

        if (this->stopping_) {
            this->finished_ = true;
            return;
        }

        if (primary_eof_ && auxiliary_eof_) {
            this->sink_->put(T());
            this->finished_ = true;
            return;
        }

        T* primary = this->source_edges_[primary_index_]->peek();
        T* auxiliary = this->source_edges_[auxiliary_index_]->peek();

        if (auxiliary_eof_) {
            if (primary == nullptr) {
                this->waitForInput();
                return;
            }
            if (isEofMarker(*primary)) {
                this->source_edges_[primary_index_]->pop();
                primary_eof_ = true;
                return;
            }
            this->sink_->put(*primary);
            this->source_edges_[primary_index_]->pop();
            return;
        }

        if (primary_eof_) {
            if (auxiliary == nullptr) {
                this->waitForInput();
                return;
            }
            if (isEofMarker(*auxiliary)) {
                this->source_edges_[auxiliary_index_]->pop();
                auxiliary_eof_ = true;
                return;
            }
            this->source_edges_[auxiliary_index_]->pop();
            return;
        }

        // Keep synchronization mux-like: do not infer "missing" from empty queues
        // while both sides are still live.
        if (primary == nullptr || auxiliary == nullptr) {
            this->waitForInput();
            return;
        }

        if (isEofMarker(*primary) || isEofMarker(*auxiliary)) {
            consumeEofMarkers();
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
            // Copy side data from auxiliary to primary (for segmentation masks, etc.)
            if (auxiliary->raw()->nb_side_data > 0) {
                for (int i = 0; i < auxiliary->raw()->nb_side_data; ++i) {
                    AVFrameSideData* sd_src = auxiliary->raw()->side_data[i];
                    if (!sd_src || !sd_src->buf) continue;
                    // Only copy if not already present on primary
                    if (av_frame_get_side_data(primary->raw(), sd_src->type)) continue;
                    AVBufferRef* ref = av_buffer_ref(sd_src->buf);
                    if (!ref) continue;
                    AVFrameSideData* sd_dst = av_frame_new_side_data_from_buf(primary->raw(), sd_src->type, ref);
                    if (!sd_dst) {
                        av_buffer_unref(&ref);
                        logstream << "join_metadata: failed to copy side data type " << (int)sd_src->type;
                    }
                }
            }
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
