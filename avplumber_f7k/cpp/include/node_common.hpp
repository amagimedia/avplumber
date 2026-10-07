#pragma once
// Include-path shim for compiling src/nodes/rescale_video.cpp on the Rust
// graph. It is not a replacement for src/nodes/node_common.hpp: the C++
// Makefile build never sees this file. Quoted includes in that .cpp
// (`../util.hpp`, `../video_parameters.hpp`) still resolve to src/.

#include "avplumber_core.h"
#include "util.hpp"

#include <avcpp/frame.h>
#include <avcpp/pixelformat.h>

#include <cstdio>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

class InstanceData;
extern "C" InstanceData* avp_cpp_instance(void* rust_instance);

av::Rational parseRatio(const std::string ratio);

extern "C" {
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

class Node : public std::enable_shared_from_this<Node> {
public:
    virtual ~Node() = default;
    virtual void init(class EdgeManager&, const Parameters&) {}
    virtual void process() {}
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
};

template <typename Class>
struct HolderImpl : Holder {
    std::shared_ptr<Class> node;
    void process() override { node->process(); }
    bool finished() override { return node->finished(); }
};

} // namespace avpshim

extern "C" void avpshim_destroy(void* self);
extern "C" AvpFlow avpshim_process(void* self);

inline const AvpNodeVtable* avpshim_vtable() {
    static const AvpNodeVtable vtable = {
        nullptr, nullptr, avpshim_destroy, avpshim_process, nullptr, nullptr, 0,
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
    T* peek(const int) override { throw Error("peek is not implemented"); }
    bool tryPeek(T&, const int) override { throw Error("tryPeek is not implemented"); }
    bool pop() override { throw Error("pop is not implemented"); }
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
class Edge {
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

    // Holds the taken frame until pop(), which is how cuda_transform reads.
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
    bool enqueue(const T& frame) {
        static_assert(std::is_same<T, av::VideoFrame>::value, "shim bridges video frames");
        return push_video(outgoing_, frame);
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
    explicit EdgeSource(std::shared_ptr<Edge<T>> edge) : edge_(std::move(edge)) {}
    std::shared_ptr<Edge<T>> edge() { return edge_; }
    T get(const int) override { throw Error("EdgeSource::get is not implemented"); }
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
    bool put(const T& frame, bool) override { return edge_->enqueue(frame); }
};

class EdgeManager {
    AvpNode* node_;

public:
    explicit EdgeManager(AvpNode* node) : node_(node) {}

    template <typename T>
    std::shared_ptr<Edge<T>> find(const Parameters& name) {
        return std::make_shared<Edge<T>>(node_, name.get<std::string>());
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
};

template <typename OutputType>
class NodeSingleOutput {
protected:
    std::unique_ptr<Sink<OutputType>> sink_;

public:
    explicit NodeSingleOutput(std::unique_ptr<Sink<OutputType>> sink) : sink_(std::move(sink)) {}
    virtual ~NodeSingleOutput() = default;
};

template <typename InputType, typename OutputType>
class NodeSISO : public NodeSingleInput<InputType>, public NodeSingleOutput<OutputType> {
public:
    NodeSISO(std::unique_ptr<Source<InputType>>&& source, std::unique_ptr<Sink<OutputType>>&& sink)
        : NodeSingleInput<InputType>(std::move(source)),
          NodeSingleOutput<OutputType>(std::move(sink)) {
        auto* src = static_cast<ShimSource<InputType>*>(this->source_.get());
        auto* dst = static_cast<ShimSink<OutputType>*>(this->sink_.get());
        src->forward = &dst->bind;
    }

    template <typename Child, typename... Args>
    static std::shared_ptr<Child> createCommon(EdgeManager& edges, const Parameters& params,
                                                Args&&... args) {
        auto src_edge = edges.find<InputType>(params["src"]);
        auto dst_edge = edges.find<OutputType>(params["dst"]);
        return std::make_shared<Child>(src_edge->makeSource(), dst_edge->makeSink(),
                                        std::forward<Args>(args)...);
    }
};

namespace avpshim {

template <typename Class>
AvpNode* instantiate(AvpCore* core, AvpNode* node, const char* json) {
    try {
        Parameters params = Parameters::parse(json ? json : "null");
        EdgeManager edges(node);
        InstanceData& cpp_inst = *avp_cpp_instance(core);
        NodeCreationInfo nci{cpp_inst, edges, params};
        auto obj = Class::create(nci);
        obj->init(edges, params);
        auto holder = std::make_unique<HolderImpl<Class>>();
        holder->node = std::move(obj);
        avp_node_declare_pad(node, 0, "in", AVP_MEDIA_VIDEO);
        avp_node_declare_pad(node, 1, "out", AVP_MEDIA_VIDEO);
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
