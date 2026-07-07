# Rust Refactor — Draft Headers (`avplumber_core.h` + `node_common` shim)

> Companion to `rust_refactor_plan_v2.md` (design) and
> `rust_refactor_impl_breakdown.md` (language split). This file contains the two
> concrete M0/M2 deliverables as drafts:
>
> 1. **`avplumber_core.h`** — the C ABI the Rust core exports and both C++ and Rust
>    nodes call. Pure C, no `std::`, no C++ types cross it.
> 2. **`avplumber_node_compat.hpp`** — the C++ shim that `node_common.hpp` pulls in,
>    so existing Tier-S nodes recompile unchanged (§3 of the breakdown).
>
> Status: **draft skeleton.** avcpp method names (`.raw()`, wrap/adopt semantics)
> and a few FFmpeg accessors may need adjustment against the pinned avcpp/FFmpeg
> versions. Marked `TODO` where a decision is deferred. These are meant to be
> dropped into `src/` and compiled, then iterated.

---

## 1. `avplumber_core.h` — the C ABI

```c
/* avplumber_core.h — C ABI exported by the Rust core.
 * Consumed by C++ nodes (via the compat shim) and Rust nodes (via bindgen/manual).
 * Rules: opaque handles; explicit ownership; UTF-8 char*; JSON params as strings;
 * no C++ types; stable, append-only enums. C11.
 */
#ifndef AVPLUMBER_CORE_H
#define AVPLUMBER_CORE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ handles */
typedef struct AvpCore     AvpCore;      /* one instance (graph + scheduler)   */
typedef struct AvpNode     AvpNode;      /* a node instance (Rust- or C++-made) */
typedef struct AvpEdge     AvpEdge;      /* one edge endpoint bound to a node   */
typedef struct AvpExecutor AvpExecutor;  /* a clock domain / cooperative loop   */

/* --------------------------------------------------------------- primitives */
typedef struct { int num; int den; } AvpRational;

#define AVP_NOPTS INT64_MIN

/* -------------------------------------------------------------- media types */
typedef enum {
    AVP_MEDIA_PACKET   = 1,  /* ptr = AVPacket*                    */
    AVP_MEDIA_VIDEO    = 2,  /* ptr = AVFrame*                     */
    AVP_MEDIA_AUDIO    = 3,  /* ptr = AVFrame*                     */
    AVP_MEDIA_EGL      = 4,  /* ptr = opaque EglImageFrame* (C++)  */
    AVP_MEDIA_METADATA = 5   /* ptr = opaque MetadataFrame*  (C++) */
} AvpMediaType;

/* One media buffer crossing the boundary. Rust passes ONE reference in; the
 * receiver owns it until it either frees it or forwards it via avp_edge_push. */
typedef struct {
    AvpMediaType type;
    void*        ptr;     /* AVFrame*/AVPacket* for 1..3; opaque for 4..5 */
    uint64_t     epoch;   /* seek/restart generation; see events below   */
} AvpBuffer;

/* For opaque C++-owned media (EGL/Metadata), the C++ side registers how Rust may
 * move/own it without understanding it. AVFrame/AVPacket need no vtable (Rust uses
 * FFmpeg C functions directly). */
typedef struct {
    void    (*retain)(void* obj);
    void    (*release)(void* obj);
    int64_t (*get_pts)(void* obj);                 /* AVP_NOPTS if none      */
    void    (*get_time_base)(void* obj, AvpRational* out);
} AvpMediaVtable;

void avp_register_media_type(AvpCore*, AvpMediaType, const AvpMediaVtable*);

/* -------------------------------------------------------------- edge events */
/* In-band control on edges. Segment & Caps are STICKY: cached on the edge and
 * replayed to a (re)connecting consumer (design §3.6 — restart safety). */
typedef enum {
    AVP_EV_EOF         = 1,
    AVP_EV_FLUSH_START = 2,   /* uses .epoch */
    AVP_EV_FLUSH_STOP  = 3,   /* uses .epoch */
    AVP_EV_SEGMENT     = 4,   /* uses .segment (STICKY) */
    AVP_EV_CAPS        = 5    /* uses .caps    (STICKY) */
} AvpEventType;

typedef enum {
    AVP_TS_INPUT     = 0,
    AVP_TS_WALLCLOCK = 1,
    AVP_TS_SYNCTIME  = 2
} AvpTimestampSource;

/* Timeline definition. out_pts = base + rescale((pts - start)/rate, time_base -> tb).
 * Promotion of SpeedControlTeam::scalePTS + RealTimeTeam offset + input_rec ts. */
typedef struct {
    AvpRational        time_base;
    int64_t            base;       /* output pts assigned to `start`        */
    int64_t            start;      /* first valid source pts in this segment*/
    double             rate;       /* playback rate; <0 = reverse           */
    uint64_t           sync_group; /* streams sharing it stay co-buffered   */
    AvpTimestampSource ts_source;
} AvpSegment;

/* Sticky stream format. Complements (does not replace) IVideoFormatSource /
 * IAudioMetadataSource pull queries. */
typedef struct {
    AvpMediaType media;              /* VIDEO or AUDIO                     */
    /* video */
    int          width, height;
    int          pixel_format;       /* AVPixelFormat                     */
    AvpRational  frame_rate;
    AvpRational  sample_aspect_ratio;
    /* audio */
    int          sample_rate;
    int          sample_format;      /* AVSampleFormat                    */
    uint64_t     channel_layout;
    /* common */
    AvpRational  time_base;
} AvpCaps;

typedef struct {
    AvpEventType type;
    uint64_t     epoch;      /* FLUSH_START / FLUSH_STOP */
    AvpSegment   segment;    /* SEGMENT */
    AvpCaps      caps;       /* CAPS    */
} AvpEdgeEvent;

/* A dequeued item is either a buffer or an event (single ordered stream). */
typedef struct {
    int          is_event;   /* 0 = buffer, 1 = event */
    AvpBuffer    buffer;
    AvpEdgeEvent event;
} AvpItem;

/* --------------------------------------------------------------- edge ops */
typedef enum {
    AVP_FLOW_PUSHED       = 0,
    AVP_FLOW_DROP         = 1,
    AVP_FLOW_BACKPRESSURE = 2,   /* sink full; retry when writable      */
    AVP_FLOW_EOF          = 3,
    AVP_FLOW_ERROR        = 4
} AvpFlow;

/* Producer side. push transfers the buffer's ref into the edge on PUSHED; on
 * BACKPRESSURE/DROP the caller retains ownership. Events never drop. */
AvpFlow avp_edge_push(AvpEdge*, const AvpBuffer* buf);
void    avp_edge_push_event(AvpEdge*, const AvpEdgeEvent* ev);

/* Consumer side. peek does not transfer ownership; pop advances. On peek of a
 * buffer, ptr is borrowed until pop. To keep it past pop, ref it (av_frame_ref /
 * the media vtable retain). timeout_ms: <0 block, 0 poll, >0 bounded. */
int  avp_edge_peek(AvpEdge*, int timeout_ms, AvpItem* out);  /* 1 got, 0 none */
void avp_edge_pop(AvpEdge*);
int  avp_edge_occupied(AvpEdge*);

/* Non-blocking wakeup (replaces processWhenSignalled / consumedEvent). The core
 * re-invokes the node's vtable.poll when the edge becomes readable/writable. */
void avp_edge_notify_readable(AvpEdge*, AvpNode*);
void avp_edge_notify_writable(AvpEdge*, AvpNode*);

/* Bind a named edge to this node as source/sink of a given media type. Returns the
 * endpoint handle. capacity==0 lets the core decide (DirectEdge if co-located). */
AvpEdge* avp_node_bind_source(AvpNode*, const char* edge_name, AvpMediaType, size_t capacity);
AvpEdge* avp_node_bind_sink  (AvpNode*, const char* edge_name, AvpMediaType, size_t capacity);

/* --------------------------------------------------------- node vtable/api */
/* Nodes PULL from their edges (avp_edge_peek). process()/poll() take no item. */
typedef struct {
    void    (*start)(AvpNode*);
    void    (*stop)(AvpNode*);
    void    (*destroy)(AvpNode*);     /* release the backing node object     */

    /* Blocking nodes: run on a dedicated OS thread; loop until EOF/ERROR.
     * process() blocks internally on avp_edge_peek(timeout<0). NULL if non-blk. */
    AvpFlow (*process)(AvpNode*);

    /* Non-blocking nodes: cooperative; called when scheduled/woken. Uses
     * avp_edge_peek(timeout=0) + notify_*. NULL if blocking. */
    AvpFlow (*poll)(AvpNode*);

    /* Capability discovery. Returns a const per-interface vtable, or NULL. */
    const void* (*query_interface)(AvpNode*, uint32_t iface_id);
} AvpNodeVtable;

/* The core calls this to attach the C++/Rust object + vtable to the AvpNode. */
void  avp_node_set_impl(AvpNode*, void* self, const AvpNodeVtable*);
void* avp_node_impl(AvpNode*);
const char* avp_node_name(AvpNode*);

/* -------------------------------------------------- interface discovery */
/* Stable, append-only ids for the graph_interfaces.hpp capabilities. */
typedef enum {
    AVP_IFACE_DECODER              = 1,
    AVP_IFACE_ENCODER              = 2,
    AVP_IFACE_MUXER                = 3,
    AVP_IFACE_VIDEO_FORMAT_SOURCE  = 4,
    AVP_IFACE_AUDIO_METADATA_SOURCE= 5,
    AVP_IFACE_FRAME_RATE_SOURCE    = 6,
    AVP_IFACE_TIME_BASE_SOURCE     = 7,
    AVP_IFACE_PLAYBACK_CONTROL     = 8,
    AVP_IFACE_INPUT_RESET          = 9,
    AVP_IFACE_FRAME_NUMBER         = 10,
    AVP_IFACE_FRAME_TIMESTAMP      = 11,
    AVP_IFACE_SENTINEL             = 12,
    AVP_IFACE_PREFERRED_FORMAT_RX  = 13,
    AVP_IFACE_NEEDS_OUT_FRAME_SIZE = 14,
    AVP_IFACE_RETURNS_OBJECTS      = 15,
    AVP_IFACE_INPUTS_OBJECTS       = 16,
    AVP_IFACE_STREAMS_INPUT        = 17,
    AVP_IFACE_JACK_SINK            = 18
    /* append only */
} AvpInterfaceId;

/* Query one node, or walk upstream from an edge (replaces findNodeUp<T>). */
const void* avp_node_query_interface(AvpNode*, AvpInterfaceId);
const void* avp_find_interface_up(AvpEdge* from, AvpInterfaceId);

/* Representative per-interface C vtables. First arg is always the AvpNode whose
 * query_interface returned this vtable. The rest follow the same pattern. */
typedef struct {                                   /* AVP_IFACE_DECODER */
    const char* (*codec_name)(AvpNode*);
    const char* (*media_type_string)(AvpNode*);
    void        (*discard_until)(AvpNode*, int64_t pts, AvpRational tb);
} AvpIDecoder;

typedef struct {                                   /* AVP_IFACE_VIDEO_FORMAT_SOURCE */
    int (*width)(AvpNode*);
    int (*height)(AvpNode*);
    int (*pixel_format)(AvpNode*);                 /* AVPixelFormat */
} AvpIVideoFormatSource;

typedef struct {                                   /* AVP_IFACE_PLAYBACK_CONTROL */
    int  (*get_direction)(AvpNode*);               /* 0 fwd, 1 back */
    void (*set_direction)(AvpNode*, int dir);
    /* Target conversion / seek now flow as core-issued events; these remain for
     * nodes that still answer queries. See design §4.4 (interfaces kept). */
} AvpIPlaybackControl;

typedef struct {                                   /* AVP_IFACE_RETURNS_OBJECTS */
    /* returns an owned JSON string; caller frees with avp_string_free */
    char* (*get_object)(AvpNode*, const char* name);
} AvpIReturnsObjects;

typedef struct {                                   /* AVP_IFACE_INPUTS_OBJECTS */
    void (*set_object)(AvpNode*, const char* name, const char* json);
} AvpIInputsObjects;

void avp_string_free(char*);

/* -------------------------------------------------- factory registration */
/* Replaces DECLNODE + generate_node_list. The factory creates the impl, calls
 * avp_node_set_impl, and returns the AvpNode (core-allocated, passed in). */
typedef AvpNode* (*AvpNodeFactoryFn)(AvpCore*, AvpNode* node, const char* json_params);

void avp_register_node_factory(AvpCore*, const char* type_name, AvpNodeFactoryFn);

/* Called once at load to let a translation unit self-register its factories,
 * before any graph is built. C++ shim drives this from static registrars. */
typedef void (*AvpModuleInit)(AvpCore*);
void avp_register_module_init(AvpModuleInit);

/* ---------------------------------------------- instance-shared registry */
/* Generic string-keyed shared objects (sentinel, etc.). The 4 Teams are native
 * Rust services and do NOT use this. */
void* avp_shared_get(AvpCore*, const char* type_key, const char* name);
void  avp_shared_put(AvpCore*, const char* type_key, const char* name,
                     void* obj, const AvpMediaVtable* ownership /* retain/release */);

/* --------------------------------------------------------- core lifecycle */
/* Usually the Rust binary owns main(); these exist for the static-lib/OBS embed. */
AvpCore* avp_core_create(void);
void     avp_core_destroy(AvpCore*);
int      avp_core_exec_command(AvpCore*, const char* line, char** out_reply); /* control proto */
int      avp_core_serve_tcp(AvpCore*, uint16_t port);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* AVPLUMBER_CORE_H */
```

### Design notes on the C ABI

- **Pull, not push.** `process()`/`poll()` take no item; nodes pull via
  `avp_edge_peek`. This is faithful to today's `source_->get()` and makes the shim a
  thin wrapper. (Earlier drafts sketched a push `poll(item)`; pull is chosen so
  legacy node bodies don't invert.)
- **One item stream.** Buffers and events share one ordered queue (`AvpItem`), so
  ordering between a `Segment` and the buffers under it is intrinsic (§3.3).
- **Ownership is stated per call.** `push` takes the ref on `PUSHED`; `peek` borrows
  until `pop`. This is the whole refcount contract the shim must honor (§3.2 of the
  breakdown).
- **Interface vtables are `const` singletons per node type**; `query_interface`
  returns a pointer to a static struct whose fns re-enter the node. Only 4 shown;
  the rest are mechanical.

---

## 2. `avplumber_node_compat.hpp` — the C++ shim

`node_common.hpp` becomes a one-line include of this. Existing Tier-S nodes keep
their source (design §3).

```cpp
/* avplumber_node_compat.hpp — C++ compatibility shim over avplumber_core.h.
 * Re-implements the old node_common.hpp surface (Node, NodeSISO, Source/Sink,
 * DECLNODE*, interfaces) backed by the C ABI. Draft skeleton.
 */
#pragma once
#include "avplumber_core.h"
#include "graph_interfaces.hpp"   /* the C++ interface classes are UNCHANGED */
#include "util.hpp"               /* Error, logstream, Parameters (nlohmann)  */
#include <avcpp/packet.h>
#include <avcpp/frame.h>
#include <memory>
#include <string>

extern "C" {
#include <libavutil/frame.h>
#include <libavcodec/packet.h>
}

/* ---------------------------------------------------- buffer <-> avcpp marshalling
 * Rust delivers ONE ref per buffer. Wrapping adopts it into an avcpp object that
 * unrefs on destruction; forwarding via Sink::put moves the ref back out. */
namespace avpshim {

template<typename T> struct Media;   /* trait: media tag + wrap/unwrap for T */

template<> struct Media<av::Packet> {
    static constexpr AvpMediaType tag = AVP_MEDIA_PACKET;
    /* adopt: take ownership of an AVPacket* Rust handed us (already reffed). */
    static av::Packet wrap(void* p) {
        AVPacket* pk = static_cast<AVPacket*>(p);
        return av::Packet(pk, av::Packet::wrap_ref{});  /* TODO: exact avcpp adopt ctor */
    }
    static void* unwrap(av::Packet& x) { return x.raw(); }  /* ref moved into edge */
};
template<> struct Media<av::VideoFrame> {
    static constexpr AvpMediaType tag = AVP_MEDIA_VIDEO;
    static av::VideoFrame wrap(void* p) {
        AVFrame* f = static_cast<AVFrame*>(p);
        return av::VideoFrame(f, av::VideoFrame::wrap_ref{});  /* TODO: avcpp adopt */
    }
    static void* unwrap(av::VideoFrame& x) { return x.raw(); }
};
template<> struct Media<av::AudioSamples> {
    static constexpr AvpMediaType tag = AVP_MEDIA_AUDIO;
    static av::AudioSamples wrap(void* p) {
        AVFrame* f = static_cast<AVFrame*>(p);
        return av::AudioSamples(f, av::AudioSamples::wrap_ref{});
    }
    static void* unwrap(av::AudioSamples& x) { return x.raw(); }
};
/* EglImageFrame / MetadataFrame: opaque C++ objects; wrap = static_cast, and the
 * media vtable registered via avp_register_media_type handles retain/release. */

} /* namespace avpshim */

/* ---------------------------------------------------------------- Source<T> */
template<typename T> class Source {
    AvpEdge* edge_;
    /* holds the last-peeked adopted object so peek() can return a T* */
    std::unique_ptr<T> peeked_;
public:
    using DataType = T;
    explicit Source(AvpEdge* e): edge_(e) {}

    /* blocking get: returns next buffer, skipping/handling events (EOF surfaces as
     * an EOF-marker T so isEofMarker() still works during the transition). */
    T get(int timeout_ms = -1) {
        AvpItem it;
        while (avp_edge_peek(edge_, timeout_ms, &it)) {
            if (it.is_event) { if (handleEvent(it.event)) { avp_edge_pop(edge_);
                return makeEofMarker<T>(); } avp_edge_pop(edge_); continue; }
            T data = avpshim::Media<T>::wrap(it.buffer.ptr);  /* adopts the ref */
            avp_edge_pop(edge_);
            return data;
        }
        return T();   /* finished */
    }
    /* non-blocking peek (timeout 0). Pointer valid until pop(). */
    T* peek(int timeout_ms = 0) {
        AvpItem it;
        if (!avp_edge_peek(edge_, timeout_ms, &it)) return nullptr;
        if (it.is_event) { handleEvent(it.event); return nullptr; }
        /* ref the frame so it survives pop; adopt into peeked_ */
        peeked_ = std::make_unique<T>(avpshim::Media<T>::wrap(refOf(it.buffer)));
        return peeked_.get();
    }
    bool pop() { peeked_.reset(); avp_edge_pop(edge_); return true; }
    AvpEdge* edge() { return edge_; }
private:
    bool handleEvent(const AvpEdgeEvent&); /* flush/segment/caps -> node hooks; TODO */
    static void* refOf(const AvpBuffer&);  /* av_frame_ref / media retain; TODO */
};

/* ------------------------------------------------------------------ Sink<T> */
template<typename T> class Sink {
    AvpEdge* edge_;
public:
    using DataType = T;
    explicit Sink(AvpEdge* e): edge_(e) {}
    bool put(T data, bool drop_if_full = false) {
        AvpBuffer b { avpshim::Media<T>::tag, avpshim::Media<T>::unwrap(data), 0 /*epoch set by core*/ };
        AvpFlow f = avp_edge_push(edge_, &b);
        if (f == AVP_FLOW_PUSHED) { data.release_ownership(); /* ref moved to edge; TODO exact avcpp */ return true; }
        if (f == AVP_FLOW_BACKPRESSURE) { if (drop_if_full) return false; /* block: TODO */ return false; }
        return false;
    }
    AvpEdge* edge() { return edge_; }
};

/* -------------------------------------------------------------------- Node */
class NodeBase {
protected:
    AvpNode* handle_ = nullptr;
public:
    virtual ~NodeBase() = default;
    void bind(AvpNode* h) { handle_ = h; }
    AvpNode* handle() { return handle_; }
    virtual void process() {}                 /* blocking nodes override */
    virtual void start() {}
    virtual void stop() {}
    /* findNodeUp<IFoo> -> avp_find_interface_up + wrap back to the C++ interface */
    template<typename Iface> Iface* findNodeUp();  /* TODO: id map + trampoline */
};

template<typename InT>  class NodeSingleInput  : public virtual NodeBase {
protected: std::unique_ptr<Source<InT>> source_;
public:    Source<InT>& source() { return *source_; } };

template<typename OutT> class NodeSingleOutput : public virtual NodeBase {
protected: std::unique_ptr<Sink<OutT>> sink_;
public:    Sink<OutT>& sink() { return *sink_; } };

template<typename InT, typename OutT>
class NodeSISO : public NodeSingleInput<InT>, public NodeSingleOutput<OutT> {
public:
    /* Same signature nodes already call. Binds edges from JSON params. */
    template<typename Child, typename... Args>
    static Child* createCommon(AvpNode* node, const Parameters& p, Args&&... args) {
        auto* c = new Child(std::forward<Args>(args)...);
        c->bind(node);
        c->source_ = std::make_unique<Source<InT>>(
            avp_node_bind_source(node, p.at("src").get<std::string>().c_str(),
                                 avpshim::Media<InT>::tag, 0));
        c->sink_ = std::make_unique<Sink<OutT>>(
            avp_node_bind_sink(node, p.at("dst").get<std::string>().c_str(),
                               avpshim::Media<OutT>::tag, 0));
        return c;
    }
};

/* -------------------------------------------------- vtable + query_interface glue */
namespace avpshim {

/* Blocking process trampoline */
template<typename Child> AvpFlow proc_tramp(AvpNode* n) {
    auto* self = static_cast<Child*>(avp_node_impl(n));
    try { self->process(); } catch (std::exception& e) { logstream << e.what(); return AVP_FLOW_ERROR; }
    return AVP_FLOW_PUSHED;
}
template<typename Child> void start_tramp(AvpNode* n){ static_cast<Child*>(avp_node_impl(n))->start(); }
template<typename Child> void stop_tramp (AvpNode* n){ static_cast<Child*>(avp_node_impl(n))->stop();  }
template<typename Child> void dtor_tramp (AvpNode* n){ delete static_cast<Child*>(avp_node_impl(n));    }

/* query_interface: dynamic_cast<IFoo*>(child) -> a static C vtable per interface.
 * Shown for IDecoder; the rest via the QI_ENTRY x-macro below. */
template<typename Child> const AvpIDecoder* qi_decoder(AvpNode* n) {
    auto* d = dynamic_cast<IDecoder*>(static_cast<Child*>(avp_node_impl(n)));
    if (!d) return nullptr;
    static const AvpIDecoder vt = {
        /* codec_name */ [](AvpNode* n)->const char*{
            static thread_local std::string s;
            s = dynamic_cast<IDecoder*>(static_cast<Child*>(avp_node_impl(n)))->codecName();
            return s.c_str(); },
        /* media_type_string */ [](AvpNode* n)->const char*{
            static thread_local std::string s;
            s = dynamic_cast<IDecoder*>(static_cast<Child*>(avp_node_impl(n)))->codecMediaTypeString();
            return s.c_str(); },
        /* discard_until */ [](AvpNode* n, int64_t pts, AvpRational tb){
            dynamic_cast<IDecoder*>(static_cast<Child*>(avp_node_impl(n)))
                ->discardUntil(av::Timestamp(pts, {tb.num, tb.den})); }
    };
    return &vt;
}

template<typename Child> const void* query_interface(AvpNode* n, uint32_t id) {
    switch (id) {
        case AVP_IFACE_DECODER: return qi_decoder<Child>(n);
        /* QI_ENTRY(AVP_IFACE_VIDEO_FORMAT_SOURCE, qi_video_format<Child>) ... */
        default: return nullptr;
    }
}

template<typename Child> const AvpNodeVtable* vtable_for(bool nonblocking) {
    static const AvpNodeVtable blk = {
        start_tramp<Child>, stop_tramp<Child>, dtor_tramp<Child>,
        proc_tramp<Child>, nullptr, query_interface<Child> };
    static const AvpNodeVtable nbk = {
        start_tramp<Child>, stop_tramp<Child>, dtor_tramp<Child>,
        nullptr, /*poll*/ nullptr /*TODO*/, query_interface<Child> };
    return nonblocking ? &nbk : &blk;
}

/* Factory: parse JSON, call Child::create(nci-equivalent), attach impl+vtable. */
template<typename Child>
AvpNode* factory(AvpCore* core, AvpNode* node, const char* json) {
    Parameters params = Parameters::parse(json);
    NodeCreationInfo nci { core, node, params };        /* shim's NCI (see below) */
    Child* c = Child::create(nci);                       /* node's own create()   */
    avp_node_set_impl(node, c, vtable_for<Child>(/*nonblocking=*/false)); /* TODO detect */
    return node;
}

struct Registrar {
    Registrar(const char* type, AvpModuleInit init) { avp_register_module_init(init); (void)type; }
};

} /* namespace avpshim */

/* NodeCreationInfo the shim hands to Child::create(). Mirrors the old struct’s
 * members that nodes read (params, edges-by-name via node, instance/core). */
struct NodeCreationInfo {
    AvpCore*          core;
    AvpNode*          node;
    const Parameters& params;
    /* edges: nodes call NodeSISO::createCommon(node, params) — no EdgeManager& */
};

/* ---------------------------------------------------------------- DECLNODE */
/* Fixed-type node. Registers one factory under `nodetype`. No trailing ';'. */
#define DECLNODE(nodetype, Class)                                              \
    static void avp_modinit_##Class(AvpCore* core) {                          \
        avp_register_node_factory(core, #nodetype, &avpshim::factory<Class>); \
    }                                                                          \
    static avpshim::Registrar avp_reg_##Class(#nodetype, avp_modinit_##Class)

/* Auto-type-detect: register one factory per concrete media type. The runtime tag
 * replaces template specialization. `type = "nodetype"` picks the first matching
 * connected edge; `nodetype<av::VideoFrame>` selects explicitly. */
#define DECLNODE_ATD(nodetype, Tpl)                                           \
    static void avp_modinit_##Tpl(AvpCore* core) {                           \
        avp_register_node_factory(core, #nodetype "<av::Packet>",      &avpshim::factory<Tpl<av::Packet>>);      \
        avp_register_node_factory(core, #nodetype "<av::VideoFrame>",  &avpshim::factory<Tpl<av::VideoFrame>>);  \
        avp_register_node_factory(core, #nodetype "<av::AudioSamples>",&avpshim::factory<Tpl<av::AudioSamples>>);\
        avp_register_node_factory(core, #nodetype,                     &avpshim::factory_atd<Tpl>); /* auto-pick */ \
    }                                                                         \
    static avpshim::Registrar avp_reg_##Tpl(#nodetype, avp_modinit_##Tpl)

/* DECLNODE_ATD_RAW / _TYPES / _ALIAS: same pattern over the restricted type set. */
```

### Design notes on the shim

- **`node_common.hpp` = `#include "avplumber_node_compat.hpp"`.** Existing nodes keep
  `#include "node_common.hpp"`; the include chain is the only thing that moved.
- **The refcount `TODO`s are the real risk surface.** `wrap` (adopt Rust's ref),
  `unwrap`+`release_ownership` (move ref into the edge), and `refOf` (ref-to-survive-
  pop in `peek`) must exactly match avcpp's `AVFrame`/`AVPacket` ownership. Pin the
  avcpp version and unit-test ref counts before porting real nodes. avcpp's exact
  adopt/wrap ctor spelling (`wrap_ref{}` here is a placeholder) is the first thing to
  nail down.
- **`query_interface` is mechanical but bulky.** Only `IDecoder` is written out; the
  other ~17 follow identically. Generate them with an X-macro over
  `(iface_id, cxx_interface, {method trampolines})` so adding an interface is one
  line — mirroring how the old `DECLNODE` avoided boilerplate.
- **Blocking vs non-blocking detection.** `vtable_for<Child>(nonblocking)` currently
  hardcodes blocking; detect via `std::is_base_of<NonBlockingNodeBase, Child>` (the
  shim keeps that marker base) and wire the `poll` trampoline + `notify_*`.
- **Events → node hooks.** `Source::handleEvent` currently only surfaces EOF as a
  marker (transition compatibility). Once nodes are event-aware it dispatches
  `FlushStart`/`Segment`/`Caps` to optional `on_event` overrides; stateful Tier-S
  nodes (decoders, filters) implement it to reset internal state (§3.6).

---

## 3. Immediate TODO checklist (before porting a real node)

1. Pin avcpp + FFmpeg versions; determine the exact **adopt/wrap ctor** and ref
   semantics for `av::Packet`/`av::VideoFrame`/`av::AudioSamples`. Replace the
   `wrap_ref{}` placeholders. **Unit-test refcounts** (wrap→drop, wrap→put) with
   `av_buffer_get_ref_count`.
2. Implement `Source::refOf` (`av_frame_ref`/`av_packet_ref`/media-vtable retain) and
   confirm `peek`→`pop` keeps exactly one net ref.
3. Generate the full `query_interface` X-macro table (17 entries) + the
   `findNodeUp<Iface>` trampoline that wraps `avp_find_interface_up`.
4. Wire non-blocking detection + the `poll`/`notify_*` path; port `firewall` (Tier R)
   and `rescale_video` (Tier S) as the two proofs.
5. Decide `MetadataFrame` field access (design open Q §12.3) — opaque-move is enough
   for C++↔C++; only add C accessors when a Rust node needs to read it.
