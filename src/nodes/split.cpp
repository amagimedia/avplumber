#include "node_common.hpp"

template <typename T> class Split: public NodeSingleInput<T>, public NodeMultiOutput<T>, public ReportsFinishByFlag {
protected:
    bool drop_ = false;
    bool restart_with_keyframe_ = false;
    bool eof_pending_ = false;
    std::vector<bool> waiting_key_, eof_sent_;
    template<typename U> static bool isKey(const U&) { return true; }
    static bool isKey(const av::Packet& packet) { return packet.isKeyPacket(); }
    void forwardPendingEof() {
        bool complete = true;
        size_t index = 0;
        for (auto &edge: this->sink_edges_) {
            if (!eof_sent_[index]) {
                eof_sent_[index] = edge->try_enqueue(createEofMarker<T>());
                complete = complete && eof_sent_[index];
            }
            ++index;
        }
        this->finished_ = complete;
    }
public:
    using NodeSingleInput<T>::NodeSingleInput;
    virtual bool consumeEofIfPresent() override {
        if (eof_pending_) return false;
        T* data = this->source_->peek(0);
        if (data == nullptr || !isEofMarker(*data)) {
            return false;
        }
        this->source_->pop();
        onEofConsumed();
        if (!drop_) this->finished_ = true;
        return true;
    }
    virtual void onEofConsumed() override {
        if (drop_) {
            // Consume upstream EOF first. A stalled optional consumer must not
            // hold the reliable branch's split inside a blocking enqueue.
            eof_pending_ = true;
            eof_sent_.resize(this->sink_edges_.size(), false);
            forwardPendingEof();
            return;
        }
        T eof = createEofMarker<T>();
        for (auto &edge: this->sink_edges_) {
            edge->enqueue(eof);
        }
    }
    virtual void process() {
        if (eof_pending_) {
            forwardPendingEof();
            if (!this->finished_) this->sink_edges_.front()->consumedEvent().wait(10);
            return;
        }
        T* data = this->source_->peek();
        if (data==nullptr) {
            return;
        }
        if (isEofMarker(*data)) {
            this->source_->pop();
            onEofConsumed();
            if (!drop_) this->finished_ = true;
            return;
        }
        size_t index = 0;
        for (auto &edge: this->sink_edges_) {
            if (waiting_key_[index] && !isKey(*data)) {
                edge->countDrop();
                ++index;
                continue;
            }
            if (!data->isComplete()) {
                logstream << "WARNING: split putting incomplete frame into sink!";
            }
            const bool sent = EdgeSink<T>(edge).put(*data, drop_);
            waiting_key_[index++] = restart_with_keyframe_ && !sent;
        }
        this->source_->pop();
    }
    void setDrop(const bool drop) {
        drop_ = drop;
    }
    static std::shared_ptr<Split> create(NodeCreationInfo &nci) {
        EdgeManager &edges = nci.edges;
        const Parameters &params = nci.params;
        auto in_edge = edges.find<T>(params["src"]);
        auto r = std::make_shared<Split>(make_unique<EdgeSource<T>>(in_edge));
        if (params.count("drop")==1) {
            r->setDrop(params["drop"].get<bool>());
        }
        r->createSinksFromParameters(edges, params);
        r->restart_with_keyframe_ = params.value("restart_with_keyframe", false);
        r->waiting_key_.resize(r->sink_edges_.size(), false);
        in_edge->setConsumer(r);
        return r;
    }
};

DECLNODE_ATD(split, Split);
