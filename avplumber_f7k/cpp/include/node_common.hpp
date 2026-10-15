#pragma once
// Include-path shim for compiling src/nodes/rescale_video.cpp on the Rust
// graph. It is not a replacement for src/nodes/node_common.hpp: the C++
// Makefile build never sees this file. Quoted includes in that .cpp
// (`../util.hpp`, `../video_parameters.hpp`) still resolve to src/.

#include "avplumber_core.h"
#include "util.hpp"

#include <avcpp/frame.h>
#include <avcpp/pixelformat.h>
#include <avcpp/sampleformat.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using std::make_unique;

class InstanceData;
extern "C" InstanceData* avp_cpp_instance(void* rust_instance);

av::Rational parseRatio(const std::string ratio);

extern "C" {
#include <libavutil/avutil.h>
#include <libavutil/frame.h>
}

inline bool isEofMarker(const av::VideoFrame& frame) {
    return frame.pts().isNoPts();
}

class IVideoFormatSource {
public:
    virtual ~IVideoFormatSource() = default;
    virtual int width() = 0;
    virtual int height() = 0;
    virtual av::PixelFormat pixelFormat() = 0;
    virtual av::PixelFormat realPixelFormat() { return pixelFormat(); }
};

class IFlushable {
public:
    virtual ~IFlushable() = default;
    virtual void flush() = 0;
};

class IInputReset {
public:
    virtual ~IInputReset() = default;
    virtual void resetInput() = 0;
};

class IFrameRateSource {
public:
    virtual ~IFrameRateSource() = default;
    virtual av::Rational frameRate() = 0;
};

class ITimeBaseSource {
public:
    virtual ~ITimeBaseSource() = default;
    virtual av::Rational timeBase() = 0;
};

class IAudioMetadataSource {
public:
    virtual ~IAudioMetadataSource() = default;
    virtual int sampleRate() = 0;
    virtual av::SampleFormat sampleFormat() = 0;
    virtual uint64_t channelLayout() = 0;
};

class IReturnsObjects {
public:
    virtual ~IReturnsObjects() = default;
    virtual Parameters getObject(const std::string) = 0;
};

class IInputsObjects {
public:
    virtual ~IInputsObjects() = default;
    virtual void setObject(const std::string, const Parameters&) = 0;
};

class Node;
class EdgeBase {
public:
    virtual ~EdgeBase() = default;
    virtual void clear() {}
    virtual std::weak_ptr<Node> producer() { return {}; }
};

class Node : public std::enable_shared_from_this<Node> {
public:
    virtual ~Node() = default;
    virtual void init(class EdgeManager&, const Parameters&) {}
    virtual void process() {}
    virtual std::weak_ptr<Node> sourceNode() { return {}; }
    virtual std::shared_ptr<EdgeBase> sourceEdge() { return {}; }
};

class IPreferredFormatReceiver {
public:
    virtual ~IPreferredFormatReceiver() = default;
    virtual void setPreferredPixelFormat(av::PixelFormat) = 0;
    virtual void setPreferredResolution(int width, int height) = 0;
};

class ReportsFinishByFlag {
protected:
    bool finished_ = false;

public:
    virtual bool finished() { return finished_; }
    void markFinished() { finished_ = true; }
    virtual ~ReportsFinishByFlag() = default;
};

class NodeDoesNotBuffer : public IFlushable, public ReportsFinishByFlag {
public:
    void flush() override { finished_ = true; }
};

// Names remembered at create(); bound on the first get/put, after node.add
// has inserted the vertex. A second bind of the same endpoint succeeds.
struct LazyBind {
    AvpNode* node = nullptr;
    std::string name;
    bool output = false;
    AvpEdge* edge = nullptr;

    void ensure() {
        if (edge) {
            return;
        }
        edge = output ? avp_node_bind_sink(node, name.c_str(), AVP_MEDIA_VIDEO, 0)
                      : avp_node_bind_source(node, name.c_str(), AVP_MEDIA_VIDEO, 0);
        if (!edge) {
            throw Error(std::string("failed to bind ") + (output ? "output " : "input ") + name);
        }
    }
};

char* avpshim_dup(const char* text);

namespace avpshim {

inline av::VideoFrame adopt_video(AVFrame* owned) {
    if (!owned) {
        throw Error("video edge delivered a null AVFrame");
    }
    // avcpp's constructor refs; it has no adopt. Free the taken pointer after
    // the ref. The first setTimeBase from {0,0} does not rescale PTS.
    const AVRational tb = owned->time_base;
    av::VideoFrame frame(owned);
    av_frame_free(&owned);
    if (!(tb.num == 0 && tb.den == 0)) {
        frame.setTimeBase(av::Rational(tb.num, tb.den));
    }
    frame.setComplete(true);
    return frame;
}

struct Holder {
    virtual ~Holder() = default;
    virtual void process() = 0;
    virtual bool finished() = 0;
    virtual int set_object(const char*, const char*, char** err) {
        if (err) {
            *err = avpshim_dup("node has no runtime controls");
        }
        return -1;
    }
    virtual int get_object(const char*, char**, char** err) {
        if (err) {
            *err = avpshim_dup("node returns no objects");
        }
        return -1;
    }
};

template <typename T, typename = void>
struct HasFinished : std::false_type {};

template <typename T>
struct HasFinished<T, std::void_t<decltype(std::declval<T&>().finished())>> : std::true_type {};

template <typename Class>
struct HolderImpl : Holder {
    std::shared_ptr<Class> node;
    void process() override { node->process(); }
    bool finished() override {
        if constexpr (HasFinished<Class>::value) {
            return node->finished();
        } else {
            return false;
        }
    }
    int set_object(const char* key, const char* json, char** err) override {
        try {
            auto control = std::dynamic_pointer_cast<IInputsObjects>(node);
            if (!control) {
                if (err) {
                    *err = avpshim_dup("node has no runtime controls");
                }
                return -1;
            }
            control->setObject(key ? key : "", Parameters::parse(json ? json : "null"));
            return 0;
        } catch (const std::exception& ex) {
            if (err) {
                *err = avpshim_dup(ex.what());
            }
            return -1;
        }
    }
    int get_object(const char* key, char** json, char** err) override {
        try {
            auto control = std::dynamic_pointer_cast<IReturnsObjects>(node);
            if (!control) {
                if (err) {
                    *err = avpshim_dup("node returns no objects");
                }
                return -1;
            }
            const std::string dumped = control->getObject(key ? key : "").dump();
            if (json) {
                *json = avpshim_dup(dumped.c_str());
            }
            return 0;
        } catch (const std::exception& ex) {
            if (err) {
                *err = avpshim_dup(ex.what());
            }
            return -1;
        }
    }
};

} // namespace avpshim

extern "C" void avpshim_destroy(void* self);
extern "C" AvpFlow avpshim_process(void* self);
extern "C" int avpshim_set_object(void* self, const char* key, const char* json, char** err);
extern "C" int avpshim_get_object(void* self, const char* key, char** json, char** err);

inline const AvpNodeVtable* avpshim_vtable() {
    static const AvpNodeVtable vtable = {
        nullptr,
        nullptr,
        avpshim_destroy,
        avpshim_process,
        nullptr,
        nullptr,
        0,
        avpshim_set_object,
        avpshim_get_object,
    };
    return &vtable;
}

template <typename T>
class Source {
public:
    virtual ~Source() = default;
    virtual T get(const int timeout_ms = -1) = 0;
    virtual bool tryGet(T& dest, const int timeout_ms = -1) = 0;
    virtual T* peek(const int timeout_ms = -1) = 0;
    virtual bool tryPeek(T& dest, const int timeout_ms = -1) = 0;
    virtual bool pop() = 0;
};

template <typename T>
class Sink {
public:
    virtual ~Sink() = default;
    virtual bool put(const T& frame, bool = false) = 0;
};

inline std::optional<av::VideoFrame> take_video(LazyBind& bind, int timeout_ms);

template <typename T>
class ShimSource : public Source<T> {
public:
    LazyBind bind;
    LazyBind* forward = nullptr;

    ShimSource(AvpNode* node, std::string name) : bind{node, std::move(name), false, nullptr} {}

    T get(const int timeout_ms) override {
        static_assert(std::is_same<T, av::VideoFrame>::value, "shim bridges video frames");
        bind.ensure();
        for (;;) {
            AvpItem item{};
            if (!avp_edge_take(bind.edge, timeout_ms, &item)) {
                // Interrupt, not EOF. A real PTS keeps isEofMarker false, and
                // an incomplete frame keeps operator bool false, so process()
                // returns without finishing and the executor can observe stop.
                av::VideoFrame idle;
                idle.raw()->pts = 0;
                return idle;
            }
            if (item.is_event) {
                if (item.event.type == AVP_EV_EOF) {
                    return av::VideoFrame();
                }
                // Spec is forwarded and not shown to the node. Flush is not
                // turned into EOF: this scaler has no state to reset.
                if (!forward) {
                    throw Error("nowhere to forward an edge event");
                }
                forward->ensure();
                avp_edge_push_event(forward->edge, &item.event);
                continue;
            }
            if (item.grain.type != AVP_MEDIA_VIDEO || item.grain.ptr == nullptr) {
                throw Error("video source received a non-video grain");
            }
            return avpshim::adopt_video(static_cast<AVFrame*>(item.grain.ptr));
        }
    }

    bool tryGet(T&, const int) override { throw Error("tryGet is not implemented"); }
    T* peek(const int timeout_ms) override {
        static_assert(std::is_same<T, av::VideoFrame>::value, "shim bridges video frames");
        if (!peeked_) {
            auto frame = take_video(bind, timeout_ms);
            if (!frame) {
                return nullptr;
            }
            peeked_ = std::move(*frame);
        }
        return &*peeked_;
    }
    bool tryPeek(T&, const int) override { throw Error("tryPeek is not implemented"); }
    bool pop() override {
        peeked_.reset();
        return true;
    }

    std::optional<T> peeked_;
};

// avp_edge_push clones, then frees the pointer it was given on accept.
// Pass a clone. The caller's VideoFrame still frees its own AVFrame.
// avcpp never writes m_timeBase into AVFrame.time_base; a drawn canvas may
// only have the time base on the AVFrame (av_frame_copy_props).
inline bool push_video(LazyBind& bind, const av::VideoFrame& frame) {
    bind.ensure();
    if (!frame.isComplete() || isEofMarker(frame)) {
        AvpEdgeEvent ev{};
        ev.type = AVP_EV_EOF;
        ev.resume_at = AVP_NOPTS;
        avp_edge_push_event(bind.edge, &ev);
        return true;
    }
    AVFrame* clone = av_frame_clone(frame.raw());
    if (!clone) {
        throw Error("av_frame_clone failed");
    }
    AVRational tb = frame.timeBase().getValue();
    if (tb.num == 0 || tb.den == 0) {
        tb = clone->time_base;
    }
    clone->time_base = tb;
    AvpGrain grain;
    grain.type = AVP_MEDIA_VIDEO;
    grain.ptr = clone;
    const AvpFlow flow = avp_edge_push(bind.edge, &grain);
    if (flow != AVP_FLOW_PUSHED) {
        av_frame_free(&clone);
        return false;
    }
    return true;
}

inline std::optional<av::VideoFrame> take_video(LazyBind& bind, int timeout_ms) {
    bind.ensure();
    for (;;) {
        AvpItem item{};
        if (!avp_edge_take(bind.edge, timeout_ms, &item)) {
            return std::nullopt;
        }
        if (item.is_event) {
            if (item.event.type == AVP_EV_EOF) {
                return av::VideoFrame();
            }
            continue;
        }
        if (item.grain.type != AVP_MEDIA_VIDEO || item.grain.ptr == nullptr) {
            throw Error("video source received a non-video grain");
        }
        return avpshim::adopt_video(static_cast<AVFrame*>(item.grain.ptr));
    }
}

template <typename T>
class ShimSink : public Sink<T> {
public:
    LazyBind bind;

    ShimSink(AvpNode* node, std::string name) : bind{node, std::move(name), true, nullptr} {}

    bool put(const T& frame, bool) override {
        static_assert(std::is_same<T, av::VideoFrame>::value, "shim bridges video frames");
        return push_video(bind, frame);
    }
};

template <typename T>
class EdgeSource;

template <typename T>
class Edge : public EdgeBase {
    AvpNode* node_;
    std::string name_;
    LazyBind incoming_;
    LazyBind outgoing_;
    std::optional<T> peeked_;
    std::weak_ptr<Node> producer_;
    std::weak_ptr<Node> consumer_;

public:
    Edge(AvpNode* node, std::string name)
        : node_(node),
          name_(std::move(name)),
          incoming_{node_, name_, false, nullptr},
          outgoing_{node_, name_, true, nullptr} {}

    void setProducer(std::weak_ptr<Node> producer) { producer_ = std::move(producer); }
    void setConsumer(std::weak_ptr<Node> consumer) { consumer_ = std::move(consumer); }
    std::weak_ptr<Node> producer() override { return producer_; }
    void clear() override {}

    // Non-blocking. NodeMultiInput polls this; a blocking peek would stall on
    // the first empty input. The frame stays here until pop().
    T* peek() { return wait_peek(0); }

    // Holds the taken frame until pop(). cuda_transform and one_to_many block
    // here through EdgeSource::peek (timeout < 0).
    T* wait_peek(int timeout_ms = -1) {
        static_assert(std::is_same<T, av::VideoFrame>::value, "shim bridges video frames");
        if (!peeked_) {
            auto frame = take_video(incoming_, timeout_ms);
            if (!frame) {
                return nullptr;
            }
            peeked_ = std::move(*frame);
        }
        return &*peeked_;
    }
    bool pop() {
        peeked_.reset();
        return true;
    }
    LazyBind& consumer_bind() { return incoming_; }
    LazyBind& producer_bind() { return outgoing_; }

    template <typename NodeType>
    std::shared_ptr<NodeType> findNodeUp() {
        return nullptr;
    }

    struct ProducedSignal {
        void signal() {}
    };
    ProducedSignal producedEvent() { return {}; }

    bool enqueue(const T& frame) {
        static_assert(std::is_same<T, av::VideoFrame>::value, "shim bridges video frames");
        return push_video(outgoing_, frame);
    }

    // Producer side. mixer_selector compares occupied() with capacity() and
    // waits on consumedEvent when the output is full.
    int occupied() {
        outgoing_.ensure();
        return avp_edge_occupied(outgoing_.edge);
    }
    int capacity() {
        outgoing_.ensure();
        const int used = avp_edge_occupied(outgoing_.edge);
        if (avp_edge_is_full(outgoing_.edge)) {
            return used > 0 ? used : 1;
        }
        return used + 1;
    }
    struct WritableWait {
        AvpEdge* edge = nullptr;
        bool wait(int timeout_ms) const {
            return edge && avp_edge_wait_writable(edge, timeout_ms);
        }
    };
    WritableWait consumedEvent() {
        outgoing_.ensure();
        return WritableWait{outgoing_.edge};
    }

    std::unique_ptr<Source<T>> makeSource() {
        return std::make_unique<ShimSource<T>>(node_, name_);
    }
    std::unique_ptr<Sink<T>> makeSink() { return std::make_unique<ShimSink<T>>(node_, name_); }
};

template <typename T>
class EdgeSource : public Source<T> {
    std::shared_ptr<Edge<T>> edge_;

public:
    LazyBind* forward = nullptr;
    explicit EdgeSource(std::shared_ptr<Edge<T>> edge) : edge_(std::move(edge)) {}
    std::shared_ptr<Edge<T>> edge() { return edge_; }
    T get(const int timeout_ms) override {
        static_assert(std::is_same<T, av::VideoFrame>::value, "shim bridges video frames");
        edge_->consumer_bind().ensure();
        for (;;) {
            AvpItem item{};
            if (!avp_edge_take(edge_->consumer_bind().edge, timeout_ms, &item)) {
                av::VideoFrame idle;
                idle.raw()->pts = 0;
                return idle;
            }
            if (item.is_event) {
                if (item.event.type == AVP_EV_EOF) {
                    return av::VideoFrame();
                }
                if (!forward) {
                    throw Error("nowhere to forward an edge event");
                }
                forward->ensure();
                avp_edge_push_event(forward->edge, &item.event);
                continue;
            }
            if (item.grain.type != AVP_MEDIA_VIDEO || item.grain.ptr == nullptr) {
                throw Error("video source received a non-video grain");
            }
            return avpshim::adopt_video(static_cast<AVFrame*>(item.grain.ptr));
        }
    }
    bool tryGet(T&, const int) override { throw Error("EdgeSource::tryGet is not implemented"); }
    T* peek(const int timeout_ms) override { return edge_->wait_peek(timeout_ms); }
    bool tryPeek(T&, const int) override { throw Error("EdgeSource::tryPeek is not implemented"); }
    bool pop() override { return edge_->pop(); }
};

template <typename T>
class EdgeSink : public Sink<T> {
    std::shared_ptr<Edge<T>> edge_;

public:
    explicit EdgeSink(std::shared_ptr<Edge<T>> edge) : edge_(std::move(edge)) {}
    std::shared_ptr<Edge<T>> edge() { return edge_; }
    bool put(const T& frame, bool) override { return edge_->enqueue(frame); }
};

class EdgeManager {
    AvpNode* node_;

public:
    explicit EdgeManager(AvpNode* node) : node_(node) {}

    template <typename T>
    std::shared_ptr<Edge<T>> find(const Parameters& name) {
        return find<T>(name.get<std::string>());
    }
    template <typename T>
    std::shared_ptr<Edge<T>> find(const std::string& name) {
        return std::make_shared<Edge<T>>(node_, name);
    }
};

struct NodeCreationInfo {
    InstanceData& instance;
    EdgeManager& edges;
    const Parameters& params;
};

template <typename InputType>
class NodeSingleInput : public virtual Node {
protected:
    std::unique_ptr<Source<InputType>> source_;

public:
    explicit NodeSingleInput(std::unique_ptr<Source<InputType>> source) : source_(std::move(source)) {}
    virtual ~NodeSingleInput() = default;
    void init(EdgeManager&, const Parameters&) override {}
    virtual void onEofConsumed() {}

    template <typename NodeType>
    std::shared_ptr<NodeType> findNodeUp() {
        return nullptr;
    }
};

template <typename OutputType>
class NodeMultiOutput : public virtual Node {
protected:
    std::vector<std::shared_ptr<Edge<OutputType>>> sink_edges_;

    void createSinksFromParameters(EdgeManager& edges, const Parameters& params) {
        for (const std::string& outname : jsonToStringList(params["dst"])) {
            auto out_edge = edges.find<OutputType>(outname);
            sink_edges_.push_back(out_edge);
            out_edge->setProducer(this->shared_from_this());
        }
    }
};

template <typename InputType>
class NodeMultiInput : public virtual Node {
protected:
    std::vector<std::shared_ptr<Edge<InputType>>> source_edges_;
    std::atomic_bool stopping_{false};
    bool auto_eof_ = true;
    struct StopEvent {
        void signal() {}
    };
    StopEvent stop_event_;

    void createSourcesFromParameters(EdgeManager& edges, const Parameters& params) {
        for (const std::string& name : jsonToStringList(params["src"])) {
            auto edge = edges.find<InputType>(name);
            source_edges_.push_back(edge);
            edge->setConsumer(this->shared_from_this());
        }
    }

    // Polls until a frame is waiting or the timeout passes. There is no
    // producedEvent on the Rust edge; mixer_selector's own wait is 20ms.
    int findSourceWithData(int timeout_ms = -1) {
        const auto start = std::chrono::steady_clock::now();
        for (;;) {
            av::Timestamp least(AV_NOPTS_VALUE, av::Rational(0, 1));
            int least_i = -1;
            for (int i = 0; i < static_cast<int>(source_edges_.size()); ++i) {
                InputType* data = source_edges_[i]->peek();
                if (!data) {
                    continue;
                }
                const av::Timestamp ts = data->pts();
                if (least.isNoPts() || ts < least) {
                    least = ts;
                    least_i = i;
                }
            }
            if (least_i >= 0 || stopping_.load(std::memory_order_relaxed)) {
                return least_i;
            }
            if (timeout_ms == 0) {
                return -1;
            }
            if (timeout_ms > 0) {
                const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start);
                if (elapsed.count() >= timeout_ms) {
                    return -1;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    void waitForInput() { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }

public:
    virtual void stop() { stopping_.store(true, std::memory_order_relaxed); }
};

template <typename OutputType>
class NodeSingleOutput {
protected:
    std::unique_ptr<Sink<OutputType>> sink_;

public:
    explicit NodeSingleOutput(std::unique_ptr<Sink<OutputType>> sink) : sink_(std::move(sink)) {}
    virtual ~NodeSingleOutput() = default;
    virtual void init(EdgeManager&, const Parameters&) {}
    EdgeSink<OutputType>* edgeSink() { return dynamic_cast<EdgeSink<OutputType>*>(sink_.get()); }
};

template <typename InputType, typename OutputType>
class NodeSISO : public NodeSingleInput<InputType>, public NodeSingleOutput<OutputType> {
public:
    using SourceType = Source<InputType>;
    using SinkType = Sink<OutputType>;

    NodeSISO(std::unique_ptr<Source<InputType>>&& source, std::unique_ptr<Sink<OutputType>>&& sink)
        : NodeSingleInput<InputType>(std::move(source)),
          NodeSingleOutput<OutputType>(std::move(sink)) {
        if (auto* src = dynamic_cast<EdgeSource<InputType>*>(this->source_.get())) {
            if (auto* dst = dynamic_cast<EdgeSink<OutputType>*>(this->sink_.get())) {
                src->forward = &dst->edge()->producer_bind();
            }
        } else if (auto* src = dynamic_cast<ShimSource<InputType>*>(this->source_.get())) {
            if (auto* dst = dynamic_cast<ShimSink<OutputType>*>(this->sink_.get())) {
                src->forward = &dst->bind;
            }
        }
    }

    template <typename Child, typename... Args>
    static std::shared_ptr<Child> createCommon(EdgeManager& edges, const Parameters& params,
                                                Args&&... args) {
        auto src_edge = edges.find<InputType>(params["src"]);
        auto dst_edge = edges.find<OutputType>(params["dst"]);
        return std::make_shared<Child>(std::make_unique<EdgeSource<InputType>>(src_edge),
                                        std::make_unique<EdgeSink<OutputType>>(dst_edge),
                                        std::forward<Args>(args)...);
    }
};

namespace avpshim {

inline void declare_pads(AvpNode* node, const Parameters& params);

template <typename Class>
AvpNode* instantiate(AvpCore* core, AvpNode* node, const char* json) {
    try {
        Parameters params = Parameters::parse(json ? json : "null");
        EdgeManager edges(node);
        InstanceData& cpp_inst = *avp_cpp_instance(core);
        NodeCreationInfo nci{cpp_inst, edges, params};
        auto obj = Class::create(nci);
        // NodeSingleOutput also has an init(); a qualified Node call is the
        // virtual one the node overrode. An unqualified call is ambiguous.
        static_cast<Node&>(*obj).init(edges, params);
        auto holder = std::make_unique<HolderImpl<Class>>();
        holder->node = std::move(obj);
        declare_pads(node, params);
        avp_node_set_impl(node, holder.release(), avpshim_vtable());
        return node;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "cpp node factory: %s\n", ex.what());
        return nullptr;
    } catch (...) {
        std::fprintf(stderr, "cpp node factory: unknown exception\n");
        return nullptr;
    }
}

inline void declare_pads(AvpNode* node, const Parameters& params) {
    // One pad keeps the name "in"/"out": rescale and cuda_transform bind every
    // edge on that side to it. Several pads are named after the edges, in
    // order, which is how bind_named pairs them and how a later bind finds them.
    const auto srcs = params.contains("src") ? jsonToStringList(params["src"]) : std::list<std::string>{};
    const auto dsts = params.contains("dst") ? jsonToStringList(params["dst"]) : std::list<std::string>{};
    if (srcs.size() <= 1) {
        avp_node_declare_pad(node, 0, "in", AVP_MEDIA_VIDEO);
    } else {
        for (const auto& name : srcs) {
            avp_node_declare_pad(node, 0, name.c_str(), AVP_MEDIA_VIDEO);
        }
    }
    if (dsts.size() <= 1) {
        avp_node_declare_pad(node, 1, "out", AVP_MEDIA_VIDEO);
    } else {
        for (const auto& name : dsts) {
            avp_node_declare_pad(node, 1, name.c_str(), AVP_MEDIA_VIDEO);
        }
    }
}

} // namespace avpshim

// The link function keeps the TU (and its static registrar) in the Rust
// binary under --gc-sections. It also queues the registrar itself, so
// registration does not depend on .init_array order.
#define DECLNODE(nodetype, classname)                                                              \
    extern "C" AvpNode* avp_factory_##nodetype(AvpCore* core, AvpNode* node, const char* json) {   \
        return avpshim::instantiate<classname>(core, node, json);                                 \
    }                                                                                              \
    extern "C" void avp_modinit_##nodetype(AvpCore* core) {                                        \
        avp_register_node_factory(core, #nodetype, avp_factory_##nodetype);                        \
    }                                                                                              \
    static struct AvpRegistrar_##nodetype {                                                        \
        AvpRegistrar_##nodetype() { avp_register_module_init(avp_modinit_##nodetype); }            \
    } avp_registrar_##nodetype;                                                                    \
    extern "C" int avp_cpp_link_##nodetype(void) {                                                 \
        avp_register_module_init(avp_modinit_##nodetype);                                          \
        return &avp_registrar_##nodetype != nullptr;                                               \
    }

// Video only. The real macro instantiates every media type and does not
// register a factory; this one registers OneToMany<av::VideoFrame>.
#define DECLNODE_ATD(nodetype, tplname)                                                            \
    using avp_atd_##nodetype = tplname<av::VideoFrame>;                                            \
    DECLNODE(nodetype, avp_atd_##nodetype)
