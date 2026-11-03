#include "soft_join_common.hpp"

// soft_join_video_frame:
// Hold-last join of auxiliary MetadataFrame streams onto a VideoFrame primary.
// Primary PTS drives emission; empty aux queues do not block.
// Held JSON is written into the frame AVDictionary (string values / JSON dumps).

class SoftJoinVideoFrame: public NodeSingleOutput<av::VideoFrame>,
                          public ReportsFinishByFlag,
                          public IStoppable {
private:
    static constexpr const char* kNodeName = "soft_join_video_frame";

    std::shared_ptr<Edge<av::VideoFrame>> primary_edge_;
    std::vector<std::shared_ptr<Edge<MetadataFrame>>> aux_edges_;
    std::vector<soft_join::AuxState> aux_;
    double max_age_sec_ = -1.0;

    Event stop_event_;
    std::unique_ptr<MultiEventWait> event_wait_;
    std::atomic_bool stopping_{false};
    bool primary_eof_ = false;

    static bool isUsablePrimary(const av::VideoFrame& frame) {
        return (!frame.isNull()) && frame.isComplete() && frame.pts().isValid();
    }

    void waitForInput(int timeout_ms = -1) {
        event_wait_->wait(timeout_ms);
    }

    void consumeEofMarkers() {
        if (!primary_eof_) {
            av::VideoFrame* primary = primary_edge_->peek();
            if (primary && isEofMarker(*primary)) {
                primary_edge_->pop();
                primary_eof_ = true;
            }
        }
        for (size_t i = 0; i < aux_.size(); ++i) {
            if (aux_[i].eof) continue;
            MetadataFrame* frame = aux_edges_[i]->peek();
            if (frame && isEofMarker(*frame)) {
                aux_edges_[i]->pop();
                aux_[i].eof = true;
            }
        }
    }

public:
    using NodeSingleOutput<av::VideoFrame>::NodeSingleOutput;

    bool consumeEofIfPresent() override {
        return false;
    }

    virtual void stop() override {
        stopping_ = true;
        stop_event_.signal();
    }

    virtual void process() override {
        consumeEofMarkers();

        if (stopping_) {
            this->finished_ = true;
            return;
        }

        if (primary_eof_ && soft_join::allAuxEof(aux_)) {
            this->sink_->put(av::VideoFrame());
            this->finished_ = true;
            return;
        }

        if (primary_eof_) {
            bool any_present = false;
            for (size_t i = 0; i < aux_.size(); ++i) {
                if (aux_[i].eof) continue;
                MetadataFrame* frame = aux_edges_[i]->peek();
                if (frame == nullptr) continue;
                any_present = true;
                if (isEofMarker(*frame)) {
                    aux_edges_[i]->pop();
                    aux_[i].eof = true;
                } else {
                    aux_edges_[i]->pop();
                }
            }
            if (!any_present && !soft_join::allAuxEof(aux_)) {
                waitForInput();
            }
            return;
        }

        av::VideoFrame* primary = primary_edge_->peek();
        if (primary == nullptr) {
            waitForInput();
            return;
        }
        if (isEofMarker(*primary)) {
            consumeEofMarkers();
            return;
        }

        if (!isUsablePrimary(*primary)) {
            this->sink_->put(*primary);
            primary_edge_->pop();
            return;
        }

        soft_join::drainAuxForPrimary(aux_edges_, aux_, primary->pts(), max_age_sec_);
        av::VideoFrame out = *primary;
        soft_join::mergeHeldIntoVideoFrame(out, aux_);
        this->sink_->put(out);
        primary_edge_->pop();
    }

    static std::shared_ptr<SoftJoinVideoFrame> create(NodeCreationInfo &nci) {
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

        auto name_it = src_names.begin();
        std::shared_ptr<Edge<av::VideoFrame>> primary_edge = edges.find<av::VideoFrame>(*name_it);
        ++name_it;

        std::vector<std::shared_ptr<Edge<MetadataFrame>>> aux_edges;
        aux_edges.reserve(aux_count);
        for (; name_it != src_names.end(); ++name_it) {
            aux_edges.push_back(edges.find<MetadataFrame>(*name_it));
        }

        std::shared_ptr<Edge<av::VideoFrame>> out_edge = edges.find<av::VideoFrame>(params["dst"]);
        auto result = std::make_shared<SoftJoinVideoFrame>(make_unique<EdgeSink<av::VideoFrame>>(out_edge));
        result->max_age_sec_ = max_age;
        result->primary_edge_ = primary_edge;
        result->aux_edges_ = std::move(aux_edges);
        result->aux_.resize(aux_count);
        for (size_t i = 0; i < aux_count; ++i) {
            result->aux_[i].into = into_keys[i];
        }

        primary_edge->setConsumer(result);
        std::vector<Event*> events;
        events.reserve(1 + aux_count + 1);
        events.push_back(&primary_edge->producedEvent());
        for (auto& edge : result->aux_edges_) {
            edge->setConsumer(result);
            events.push_back(&edge->producedEvent());
        }
        events.push_back(&result->stop_event_);
        result->event_wait_ = make_unique<MultiEventWait>(events);

        out_edge->setProducer(result);
        return result;
    }
};

DECLNODE(soft_join_video_frame, SoftJoinVideoFrame);
