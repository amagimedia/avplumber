#pragma once
#include "util.hpp"
#include "Event.hpp"
#include "MultiEventWait.hpp"
#include "graph_core.hpp"
#include <avcpp/dictionary.h>
#include <memory>
#include "graph_interfaces.hpp"

using namespace std::chrono_literals;

class ReportsFinishByFlag: public IReportsFinish {
protected:
    bool finished_ = false;
public:
    virtual bool finished() {
        return finished_;
    }
    void markFinished() {
        finished_ = true;
    }
    virtual ~ReportsFinishByFlag() {
    }
};

template<typename InputType> class NodeSingleInput: virtual public Node, public IStoppable, virtual public IInitAfterCreate, public IFlushAndSeek {
public:
    using SourceType = Source<InputType>;
protected:
    std::unique_ptr<SourceType> source_;
    bool auto_eof_ = true;
    void executeUpstream(std::function<void(EdgeBase&, std::shared_ptr<Node>)> cb) {
        std::shared_ptr<EdgeBase> edge = sourceEdge();
        while (edge) {
            auto node = edge->producer().lock();
            cb(*edge, node);
            if (node) {
                edge = node->sourceEdge();
            } else {
                break;
            }
        }
    }
public:
    SourceType& source() {
        return *source_;
    }
    NodeSingleInput(std::unique_ptr<SourceType> &&source): source_(std::move(source)) {
    }
    virtual void init(EdgeManager&, const Parameters&) override {
        registerInSourceEdge();
    }
    EdgeSource<InputType>* edgeSource() {
        return dynamic_cast<EdgeSource<InputType>*>(source_.get());
    }
    void stop() override {
        if (edgeSource()) {
            edgeSource()->edge()->finishConsumer();
        } else {
            throw Error("stop() called for node without edge source!");
        }
    }
    virtual std::weak_ptr<Node> sourceNode() override {
        auto src = edgeSource();
        if (src==nullptr) return std::weak_ptr<Node>();
        return src->edge()->producer();
    }
    virtual std::shared_ptr<EdgeBase> sourceEdge() override {
        auto src = edgeSource();
        if (src==nullptr) return {};
        return src->edge();
    }
    template<typename NodeType> std::shared_ptr<NodeType> findNodeUp() {
        auto src = edgeSource();
        if (src==nullptr) return nullptr;
        return src->edge()->template findNodeUp<NodeType>();
    }
    void registerInSourceEdge() {
        EdgeSource<InputType>* esrc = this->edgeSource();
        if (esrc!=nullptr) {
            esrc->edge()->setConsumer(this->shared_from_this());
        }
    }
    virtual void flushAndSeek_start(StreamTarget target) override {
        // pause all nodes processing
        pauseProcessing();
        executeUpstream([target](EdgeBase& edge, std::shared_ptr<Node> node) {
            if (node) {
                node->pauseProcessing();
            }
        });
        // start flushing
        executeUpstream([target](EdgeBase& edge, std::shared_ptr<Node> node) {
            edge.startFlushing();
            edge.finishConsumer(); // to wake up consumer that may be waiting for frame
            edge.finishProducer();
        });
    }
    virtual void flushAndSeek(StreamTarget target) override {
        // wait current node to end processing
        lockProcessing();
        unlockProcessing();
        // wait all nodes to complete its work
        executeUpstream([](EdgeBase& edge, std::shared_ptr<Node> node) {
            if (node) {
                node->lockProcessing();
                node->unlockProcessing();
            }
        });
    }
    void flushAndSeek_finish(StreamTarget target) override {
        // clear all edges & stop flushing
        executeUpstream([](EdgeBase& edge, std::shared_ptr<Node> node) {
            edge.clear();
            edge.stopFlushing();
        });
    }
    void flushAndSeek_complete(StreamTarget target) override {
        std::shared_ptr<IPlaybackControl> input = findNodeUp<IPlaybackControl>();
        IPlaybackControl::EPlaybackDirection direction = IPlaybackControl::EPlaybackDirection::pd_Forward;

        if (input) {
            input->convertStreamTarget(target, StreamTarget::ETargetType::tt_SyncTime);
            direction = input->getPlaybackDirection();
        }
        // set-up discard until & execute seek
        executeUpstream([target, direction](EdgeBase& edge, std::shared_ptr<Node> node) {
            if (!node || !node->isPausedProcessing())
                return;
            if (!target.isEmpty()) {
                if (target.ts.isValid()) {
                    std::shared_ptr<IDecoder> dec = std::dynamic_pointer_cast<IDecoder>(node);
                    if (dec) {
                        if (direction == IPlaybackControl::EPlaybackDirection::pd_Forward) {
                            dec->discardUntil(addTS(target.ts, av::Timestamp(-7, {1, 1000})));
                        } else {
                            // make sure encoder will outout valid frame (not any frame inside nvdec queue)
                            dec->discardUntil(av::Timestamp(0, {1, 1}));
                        }
                    }
                }
                std::shared_ptr<IPlaybackControl> input = std::dynamic_pointer_cast<IPlaybackControl>(node);
                if (input) {
                    input->seekAndPause(target);
                }
            }
            // reset nodes
            std::shared_ptr<IInputReset> input_reset = std::dynamic_pointer_cast<IInputReset>(node);
            if (input_reset) {
                input_reset->resetInput();
            }
        });
        // reset current node
        IInputReset* this_reset = dynamic_cast<IInputReset*>(this);
        if (this_reset) {
            this_reset->resetInput();
        }
        // resume all nodes
        executeUpstream([](EdgeBase& edge, std::shared_ptr<Node> node) {
            std::shared_ptr<IPlaybackControl> input = std::dynamic_pointer_cast<IPlaybackControl>(node);
            // resume input after seek
            if (input && node->isPausedProcessing()) {
                input->resumeAfterSeek();
            }
            if (node) {
                node->resumeProcessing();
            }
        });
        resumeProcessing();
    }
    virtual void onEofConsumed() {
        IFlushable* flushable = dynamic_cast<IFlushable*>(this);
        if (flushable) {
            flushable->flush();
        }
    }
    virtual bool consumeEofIfPresent() override {
        if (!auto_eof_) return false;
        InputType* ptr = source_->peek(0);
        if (ptr == nullptr || !isEofMarker(*ptr)) return false;
        source_->pop();
        onEofConsumed();
        ReportsFinishByFlag* finishable = dynamic_cast<ReportsFinishByFlag*>(this);
        if (finishable) {
            finishable->markFinished();
        }
        return true;
    }
    virtual ~NodeSingleInput() {
    }
};

template<typename T, typename = decltype(std::declval<T>().dts())> av::Timestamp getTS(T &frm) {
    return frm.dts();
}
template<typename T, typename...Args> av::Timestamp getTS(T &frm, Args...args) {
    return frm.pts();
}

template<typename InputType> class NodeMultiInput: virtual public Node, public IStoppable {
public:
    using SourceType = Source<InputType>;
protected:
    std::vector<std::shared_ptr<Edge<InputType>>> source_edges_;
    Event stop_event_;
    std::unique_ptr<MultiEventWait> event_wait_;
    std::atomic_bool stopping_{false};
    bool auto_eof_ = true;
    void createSourcesFromParameters(EdgeManager &edges, const Parameters &params) {
        std::list<std::string> edge_names = jsonToStringList(params["src"]);
        size_t count = edge_names.size();
        source_edges_.reserve(count);
        std::vector<Event*> events;
        events.reserve(count+1);
        for (std::string sname: edge_names) {
            std::shared_ptr<Edge<InputType>> edge = edges.find<InputType>(sname);
            source_edges_.push_back(edge);
            events.push_back(&(edge->producedEvent()));
            edge->setConsumer(this->shared_from_this());
        }
        events.push_back(&stop_event_);
        event_wait_ = make_unique<MultiEventWait>(events);
    }
    // This function MAY in some cases return -1 even with timeout_ms==-1
    // (e.g. if stop() is called)
    int findSourceWithData(int timeout_ms = -1) {
        while(true) {
            // find earliest packet (least PTS/DTS) in streams:
            av::Timestamp least_ts = NOTS;
            int least_ts_i = -1;
            for (int i=0; i<source_edges_.size(); i++) {
                auto &edge = source_edges_[i];
                InputType *data = edge->peek();
                if (data==nullptr) continue;
                av::Timestamp ts = getTS(*data);
                if ( least_ts.isNoPts() || (ts < least_ts) ) {
                    least_ts = ts;
                    least_ts_i = i;
                }
            }
            if (least_ts_i<0 && !stopping_) {
                if (!event_wait_->wait(timeout_ms)) {
                    return -1;
                }
            } else {
                return least_ts_i;
            }
        }
    }
    void waitForInput() {
        event_wait_->wait();
    }
public:
    virtual void stop() {
        stopping_ = true;
        stop_event_.signal();
    }
    virtual void onEofConsumed() {
        IFlushable* flushable = dynamic_cast<IFlushable*>(this);
        if (flushable) {
            flushable->flush();
        }
    }
    virtual bool consumeEofIfPresent() override {
        if (!auto_eof_) return false;
        for (auto& edge : source_edges_) {
            auto* ptr = edge->peek();
            if (!ptr || !isEofMarker(*ptr)) return false;
        }
        for (auto& edge : source_edges_) {
            edge->pop();
        }
        onEofConsumed();
        ReportsFinishByFlag* finishable = dynamic_cast<ReportsFinishByFlag*>(this);
        if (finishable) {
            finishable->markFinished();
        }
        return true;
    }

    virtual ~NodeMultiInput() {
    }
};

template<typename OutputType> class NodeWithOutputs: virtual public Node, public IWaitsSinksEmpty {
public:
    virtual void forEachOutput(std::function<void(Sink<OutputType>*)>) = 0;
    virtual void waitSinksEmpty() override {
        forEachOutput([](Sink<OutputType>* sink_a) {
            EdgeSink<OutputType>* sink = dynamic_cast<EdgeSink<OutputType>*>(sink_a);
            if (sink) {
                sink->edge()->waitEmpty();
            } else {
                throw Error("waitSinksEmpty called for node without edge sink!");
            }
        });
    }
    virtual void stopSinks() override {
        forEachOutput([](Sink<OutputType>* sink_a) {
            EdgeSink<OutputType>* sink = dynamic_cast<EdgeSink<OutputType>*>(sink_a);
            if (sink) {
                sink->edge()->finishProducer();
            } else {
                throw Error("stopSinks called for node without edge sink!");
            }
        });
    }
    // Enqueue an EOF marker on every output edge. Enqueues directly (like
    // stopSinks) rather than via put() because EOF markers carry no PTS.
    void emitEof() {
        OutputType eof = createEofMarker<OutputType>();
        forEachOutput([&eof](Sink<OutputType>* sink_a) {
            EdgeSink<OutputType>* sink = dynamic_cast<EdgeSink<OutputType>*>(sink_a);
            if (sink) {
                sink->edge()->enqueue(eof);
            } else {
                throw Error("emitEof called for node without edge sink!");
            }
        });
    }
    virtual void start() override {
        forEachOutput([](Sink<OutputType> *sink) {
            auto edge_sink = dynamic_cast<EdgeSink<OutputType>*>(sink);
            if (edge_sink==nullptr) return;
            if (edge_sink->edge()->consumer().expired()) {
                // the `expired` name is a bit misleading - it returns true also for not-yet-set weak_ptr
                logstream << "WARNING: one of dst queues is not connected to any other node";
            }
        });
    }
};

template<typename OutputType> class NodeSingleOutput: virtual public NodeWithOutputs<OutputType>, virtual public IInitAfterCreate {
public:
    using SinkType = Sink<OutputType>;
protected:
    std::unique_ptr<SinkType> sink_;
public:
    virtual void forEachOutput(std::function<void(Sink<OutputType>*)> cb) override {
        cb(sink_.get());
    }
    SinkType& sink() {
        return *sink_;
    }
    EdgeSink<OutputType>* edgeSink() {
        return dynamic_cast<EdgeSink<OutputType>*>(sink_.get());
    }
    virtual std::weak_ptr<Node> sinkNode() {
        EdgeSink<OutputType>* sink = edgeSink();
        if (sink==nullptr) return {};
        return sink->edge()->consumer();
    }
    void registerInSinkEdge() {
        EdgeSink<OutputType>* edst = this->edgeSink();
        if (edst!=nullptr) {
            edst->edge()->setProducer(this->shared_from_this());
        } else {
            logstream << "WARNING: Sink is not edge - couldn't register!" << std::endl;
        }
    }
    NodeSingleOutput(std::unique_ptr<SinkType> &&sink): sink_(std::move(sink)) {
    }
    virtual void init(EdgeManager&, const Parameters&) override {
        registerInSinkEdge();
    }
    virtual ~NodeSingleOutput() {
    }
};

template<typename OutputType> class NodeMultiOutput: public NodeWithOutputs<OutputType> {
public:
    using SinkType = Sink<OutputType>;
protected:
    std::vector<std::shared_ptr<Edge<OutputType>>> sink_edges_;
    void createSinksFromParameters(EdgeManager &edges, const Parameters &params) {
        std::list<std::string> edge_names = jsonToStringList(params["dst"]);
        sink_edges_.reserve(edge_names.size());
        for (const std::string &outname: edge_names) {
            auto out_edge = edges.find<OutputType>(outname);
            sink_edges_.push_back(out_edge);
            out_edge->setProducer(this->shared_from_this());
        }
    }
public:
    virtual void forEachOutput(std::function<void(SinkType*)> cb) {
        for (auto edge: sink_edges_) {
            EdgeSink<OutputType> sink(edge);
            cb(&sink);
        }
    }
};

// Single Input, Single Output Node
template<typename InputType, typename OutputType> class NodeSISO: public NodeSingleInput<InputType>, public NodeSingleOutput<OutputType> {
public:
    using SourceType = Source<InputType>;
    using SinkType = Sink<OutputType>;
    NodeSISO(std::unique_ptr<SourceType> &&source, std::unique_ptr<SinkType> &&sink):
        NodeSingleInput<InputType>(std::move(source)), NodeSingleOutput<OutputType>(std::move(sink)) {
    }
    template<typename Child, typename ... Args> static std::shared_ptr<Child> createCommon(EdgeManager &edges, const Parameters &params, Args&& ... args) {
        std::shared_ptr<Edge<InputType>> src_edge = edges.find<InputType>(params["src"]);
        std::shared_ptr<Edge<OutputType>> dst_edge = edges.find<OutputType>(params["dst"]);
        std::shared_ptr<Child> r = std::make_shared<Child>(src_edge->makeSource(), dst_edge->makeSink(), std::forward<Args>(args)...);
        return r;
    }
    virtual void init(EdgeManager &edges, const Parameters &params) {
        NodeSingleInput<InputType>::init(edges, params);
        NodeSingleOutput<OutputType>::init(edges, params);
    }
    virtual void onEofConsumed() override {
        IFlushable* flushable = dynamic_cast<IFlushable*>(this);
        if (flushable) {
            flushable->flush();
        }
        this->sink_->put(createEofMarker<OutputType>());
    }
    virtual ~NodeSISO() {
    }
};

template <typename T> class TransparentNode: public NodeSISO<T, T> {
public:
    using NodeSISO<T, T>::NodeSISO;
    virtual void process() {
        T* ptr = this->source_->peek();
        if (ptr==nullptr) return;
        if (isEofMarker(*ptr)) {
            // Leave the EOF marker in the queue so consumeEofIfPresent() forwards
            // it via NodeSISO::onEofConsumed() and the dumb-node loop exits.
            return;
        }
        T data = *ptr;
        this->sink_->put(data);
        this->source_->pop();
    }
    virtual ~TransparentNode() {
    }
};

class NodeDoesNotBuffer: public IFlushable, public ReportsFinishByFlag {
public:
    virtual void flush() {
        finished_ = true;
    }
};
