#pragma once
#include "util.hpp"
#include "Event.hpp"
#include "MultiEventWait.hpp"
#include "graph_core.hpp"
#include <avcpp/dictionary.h>
#include <memory>
#include "graph_interfaces.hpp"

template<typename InputType> class NodeSingleInput: virtual public Node, public IStoppable, virtual public IInitAfterCreate, public IFlushAndSeek {
public:
    using SourceType = Source<InputType>;
protected:
    std::unique_ptr<SourceType> source_;
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
    virtual void flushAndSeek(SeekTarget target) override {
        // start flushing:
        executeUpstream([target](EdgeBase& edge, std::shared_ptr<Node> node) {
            edge.startFlushing();
            edge.finishConsumer(); // to wake up consumer that may be waiting for frame
            std::shared_ptr<IDecoder> dec = std::dynamic_pointer_cast<IDecoder>(node);
            if (target.ts.isValid() && dec) {
                dec->discardUntil(target.ts);
            }
            std::shared_ptr<IStreamsInput> input = std::dynamic_pointer_cast<IStreamsInput>(node);
            if (input) {
                input->seekAndPause(target);
            }
            std::shared_ptr<IInputReset> input_reset = std::dynamic_pointer_cast<IInputReset>(node);
            if (input_reset) {
                input_reset->resetInput();
            }
        });
        IInputReset* this_reset = dynamic_cast<IInputReset*>(this);
        if (this_reset) {
            this_reset->resetInput();
        }
        // wait for flushed state:
        while(true) {
            bool flushed = true;
            executeUpstream([&flushed](EdgeBase& edge, std::shared_ptr<Node> node) {
                flushed &= edge.isFlushed();
            });
            if (flushed) {
                break;
            }
            wallclock.sleepms(5);
        }
        // stop flushing and resume paused input:
        executeUpstream([](EdgeBase& edge, std::shared_ptr<Node> node) {
            edge.stopFlushing();
            std::shared_ptr<IStreamsInput> input = std::dynamic_pointer_cast<IStreamsInput>(node);
            if (input) {
                input->resumeAfterSeek();
            }
        });
    }
    virtual ~NodeSingleInput() {
    }
};

template<typename T, typename = decltype(std::declval<T>().dts())> av::Timestamp getTS(T &frm) {
    return frm.dts();
}
template<typename T, typename...Args> av::Timestamp getTS(T &frm) {
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
    // This function MAY in some cases return -1
    // (e.g. if stop() is called)
    int findSourceWithData() {
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
                event_wait_->wait();
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

    virtual ~NodeMultiInput() {
    }
};

template<typename OutputType> class NodeWithOutputs: virtual public Node, public IWaitsSinksEmpty {
public:
    virtual void forEachOutput(std::function<void(Sink<OutputType>*)>) = 0;
    virtual void waitSinksEmpty() {
        forEachOutput([](Sink<OutputType>* sink_a) {
            EdgeSink<OutputType>* sink = dynamic_cast<EdgeSink<OutputType>*>(sink_a);
            if (sink) {
                sink->edge()->waitEmpty();
            } else {
                throw Error("waitSinksEmpty called for node without edge sink!");
            }
        });
    }
    virtual void stopSinks() {
        forEachOutput([](Sink<OutputType>* sink_a) {
            EdgeSink<OutputType>* sink = dynamic_cast<EdgeSink<OutputType>*>(sink_a);
            if (sink) {
                sink->edge()->finishProducer();
            } else {
                throw Error("stopSinks called for node without edge sink!");
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
    virtual ~NodeSISO() {
    }
};

template <typename T> class TransparentNode: public NodeSISO<T, T> {
public:
    using NodeSISO<T, T>::NodeSISO;
    virtual void process() {
        T* ptr = this->source_->peek();
        if (ptr!=nullptr) {
            T data = *ptr;
            this->sink_->put(data);
            this->source_->pop();
        }
    }
    virtual ~TransparentNode() {
    }
};

class ReportsFinishByFlag: public IReportsFinish {
protected:
    bool finished_ = false;
public:
    virtual bool finished() {
        return finished_;
    }
    virtual ~ReportsFinishByFlag() {
    }
};

class NodeDoesNotBuffer: public IFlushable, public ReportsFinishByFlag {
public:
    virtual void flush() {
        finished_ = true;
    }
};
