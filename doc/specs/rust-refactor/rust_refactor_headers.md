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
 * receiver owns it until it either frees it or forwards it via avp_edge_push.
 * PTS lives on the AVFrame/AVPacket, in SOURCE TIME, and is never rewritten in
 * transit (design §3.3). No epoch: a seek clears queues rather than tagging
 * buffers (design §3.4). */
typedef struct {
    AvpMediaType type;
    void*        ptr;     /* AVFrame*/AVPacket* for 1..3; opaque for 4..5 */
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
/* In-band CAUSAL control on edges (design §3.2). Takes effect where/when it
 * arrives; nothing is applied retroactively to buffers already downstream.
 * There is NO segment event and NO epoch — rate/offset/pause live on the master
 * clock (AvpSyncGroup below), applied at the output. FLUSH preempts the pipe:
 * it clears queued buffers on the way down (design §3.4). SPEC (stream format)
 * is causal AND latched on the edge: the edge re-presents its current SPEC as the
 * head item to any (re)connecting consumer, before any buffer — so there is no
 * upstream pull walk to recover format (design §4.4.1, §3.6). This replaces the old
 * IVideoFormatSource/IAudioMetadataSource findNodeUp() queries entirely. */
typedef enum {
    AVP_EV_EOF         = 1,
    AVP_EV_FLUSH_START = 2,   /* preempts: clears queues downstream */
    AVP_EV_FLUSH_STOP  = 3,
    AVP_EV_SPEC        = 4    /* uses .spec; latched on the edge */
} AvpEventType;

/* Stream format description (design §4.4.1): the resolved values that flow through
 * an edge. Latched on the edge; NOT a GStreamer capability set and NOT negotiated
 * (a single concrete "this is what flows here now", forward-only). Replaces the
 * Fact interfaces (IVideoFormatSource / IAudioMetadataSource / IFrameRateSource /
 * ITimeBaseSource). */
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
} AvpSpec;

typedef struct {
    AvpEventType type;
    AvpSpec      spec;       /* SPEC */
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

/* Current latched SPEC of this edge (design §4.4.1). Normally a node just receives
 * a SPEC item as the head of its stream and needn't call this; provided so the
 * shim base / a late binder can read the format synchronously without consuming.
 * Returns 1 and fills *out if a SPEC has ever flowed, 0 if none yet. */
int  avp_edge_current_spec(AvpEdge*, AvpSpec* out);

/* Non-blocking wakeup (replaces processWhenSignalled / consumedEvent). The core
 * re-invokes the node's vtable.poll when the edge becomes readable/writable. */
void avp_edge_notify_readable(AvpEdge*, AvpNode*);
void avp_edge_notify_writable(AvpEdge*, AvpNode*);

/* Bind a named edge to this node as source/sink of a given media type. Returns the
 * endpoint handle. capacity==0 lets the core decide (DirectEdge if co-located). */
AvpEdge* avp_node_bind_source(AvpNode*, const char* edge_name, AvpMediaType, size_t capacity);
AvpEdge* avp_node_bind_sink  (AvpNode*, const char* edge_name, AvpMediaType, size_t capacity);

/* ---------------------------------------------------------- master clock */
/* The SyncGroup master clock (design §3.3): the ONE shared primitive that
 * replaces SpeedControlTeam + RealTimeTeam. It owns the playback->wall mapping —
 * rate, offset, pause — for a group of streams that must stay A/V-synchronized.
 * Rate/offset/pause changes are O(1) writes here; buffers keep their source-time
 * PTS and are mapped through the clock only when an OUTPUT stage releases them.
 * A seek resets the clock (one shared reset for the whole group). No node other
 * than an output/clock-sync stage reads this. */
typedef struct AvpSyncGroup AvpSyncGroup;

AvpSyncGroup* avp_sync_group(AvpCore*, const char* name);   /* get/create by name */

/* Control-plane writes (from the control protocol / scheduled actions). */
void avp_clock_set_rate  (AvpSyncGroup*, double rate);      /* <0 = reverse       */
void avp_clock_set_paused(AvpSyncGroup*, int paused);       /* freeze/thaw output */
void avp_clock_reset     (AvpSyncGroup*, int64_t new_pos, AvpRational tb); /* seek */

/* Output-stage read: map a source-time PTS to wall/presentation time. Returns
 * AVP_NOPTS while paused-and-not-yet-due; the stage sleeps until then. */
int64_t avp_clock_map_to_wall(AvpSyncGroup*, int64_t src_pts, AvpRational tb);

/* ------------------------------------------------------- shared timeline */
/* SharedTimeline (design §3.3, was InstanceShared<SharedTimeline>): a named,
 * PTS-keyed key/value store — the fifth core service alongside the SyncGroup.
 * It is NOT on the data plane (no AVFrames): the control protocol/mixer WRITE
 * scheduled values at a source-time PTS, and C++ Tier-S nodes (one_to_many,
 * source_switcher, preheat_video_router, cuda_rect_overlay via TimelineReader)
 * READ "the value in effect at this frame's PTS". Values are opaque JSON so the
 * ABI needn't model the Parameters variant. Crosses the boundary in BOTH
 * directions (Rust control writes, C++ nodes read), hence a C surface. */
typedef struct AvpTimeline AvpTimeline;

AvpTimeline* avp_timeline(AvpCore*, const char* name);   /* get/create by name */

/* Control-plane writes (scheduled values keyed by source-time PTS, ms). */
void avp_timeline_set      (AvpTimeline*, const char* channel, const char* key,
                            int64_t at_pts_ms, const char* value_json);
void avp_timeline_clear_key(AvpTimeline*, const char* channel, const char* key);
void avp_timeline_gc       (AvpTimeline*, int64_t before_pts_ms);

/* Node read: latest value with at_pts_ms <= frame_pts (compared via the frame
 * timebase). Writes the JSON value into `out` (cap bytes) and returns its length,
 * 0 if no entry applies, or the needed length (>cap) if truncated. */
int  avp_timeline_get      (AvpTimeline*, const char* channel, const char* key,
                            int64_t frame_pts, AvpRational tb, char* out, size_t cap);

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
/* Capability discovery is DIRECT only — you query a node you already hold (design
 * §4.4). There is NO avp_find_interface_up / graph walk: the three registers the
 * old findNodeUp<T>() served are now split —
 *   - Facts  (format: IVideoFormatSource/IAudioMetadataSource/IFrameRateSource/
 *             ITimeBaseSource) -> AvpSpec, latched in-band on the edge (above).
 *   - Controls (IPlaybackControl direction, IInputReset, seek/speed/pause) ->
 *             addressed to core services (AvpSyncGroup + the flush discontinuity).
 *   - Live queries (below) -> this table, queried on a named/adjacent node.
 * So only the Register-3 capabilities remain here. Stable, append-only ids. */
typedef enum {
    AVP_IFACE_DECODER              = 1,   /* codec name, discard_until          */
    AVP_IFACE_ENCODER              = 2,
    AVP_IFACE_MUXER                = 3,
    AVP_IFACE_SENTINEL             = 4,   /* card/signal-present stats          */
    AVP_IFACE_RETURNS_OBJECTS      = 5,   /* node.param.get bridge              */
    AVP_IFACE_INPUTS_OBJECTS       = 6,   /* node.param.set bridge              */
    AVP_IFACE_STREAMS_INPUT        = 7,   /* demux stream enumeration           */
    AVP_IFACE_JACK_SINK            = 8    /* adjacent JACK pull callback         */
    /* append only. NOTE: no *_FORMAT_SOURCE / *_PLAYBACK_CONTROL / *_INPUT_RESET
     * / *_FRAME_RATE / *_TIME_BASE / *_FRAME_NUMBER / *_FRAME_TIMESTAMP ids —
     * those were Facts (now AvpSpec) or Controls (now services), not queries.
     * PREFERRED_FORMAT_RX / NEEDS_OUT_FRAME_SIZE are the reverse (downstream->
     * upstream) hints; resolved as a direct query on the immediate downstream
     * node — see design §12.1, added when that open question lands. */
} AvpInterfaceId;

/* Query a node the caller already holds (self, an adjacent node, or a node named
 * by the control protocol). The ONLY interface-discovery primitive. */
const void* avp_node_query_interface(AvpNode*, AvpInterfaceId);

/* Representative per-interface C vtables. First arg is always the AvpNode whose
 * query_interface returned this vtable. The rest follow the same pattern. */
typedef struct {                                   /* AVP_IFACE_DECODER */
    const char* (*codec_name)(AvpNode*);
    const char* (*media_type_string)(AvpNode*);
    void        (*discard_until)(AvpNode*, int64_t pts, AvpRational tb);
} AvpIDecoder;

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
- **One item stream.** Buffers and control tokens share one ordered queue
  (`AvpItem`), so ordering between a flush and the buffers around it is intrinsic;
  a flush additionally preempts (clears) the queue it crosses (§3.4).
- **Ownership is stated per call.** `push` takes the ref on `PUSHED`; `peek` borrows
  until `pop`. This is the whole refcount contract the shim must honor (§3.2 of the
  breakdown).
- **Interface vtables are `const` singletons per node type**; `query_interface`
  returns a pointer to a static struct whose fns re-enter the node. Only the
  Register-3 live-query interfaces (§4.4) exist here; a couple are shown, the rest
  are mechanical. Facts (format) are NOT interfaces — they are `AvpSpec` on the edge.

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
    bool handleEvent(const AvpEdgeEvent&); /* eof/flush/spec -> node hooks; TODO */
    static void* refOf(const AvpBuffer&);  /* av_frame_ref / media retain; TODO */
};

/* ------------------------------------------------------------------ Sink<T> */
template<typename T> class Sink {
    AvpEdge* edge_;
public:
    using DataType = T;
    explicit Sink(AvpEdge* e): edge_(e) {}
    bool put(T data, bool drop_if_full = false) {
        AvpBuffer b { avpshim::Media<T>::tag, avpshim::Media<T>::unwrap(data) };
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
    /* Format arrives as a latched SPEC on the input edge (design §4.4.1), routed to
     * this hook before the first process(). Default: forward unchanged (identity)
     * -> pass-through nodes need no override. A Transform returns its output SPEC;
     * a query-only node reads it, (re)configures, and returns it. Replaces the old
     * findNodeUp<IVideoFormatSource>() init pull. There is NO findNodeUp: no walk. */
    virtual AvpSpec on_spec(const AvpSpec& in) { return in; }
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
        /* QI_ENTRY(AVP_IFACE_ENCODER, qi_encoder<Child>) ... (Register-3 only;
         * no VIDEO_FORMAT_SOURCE — that Fact is served by on_spec, not a query) */
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
  other ~7 Register-3 interfaces (§4.4) follow identically. Generate them with an
  X-macro over `(iface_id, cxx_interface, {method trampolines})` so adding one is a
  single line — mirroring how the old `DECLNODE` avoided boilerplate. (Fact
  interfaces are NOT here; they route through `on_spec`, below.)
- **Blocking vs non-blocking detection.** `vtable_for<Child>(nonblocking)` currently
  hardcodes blocking; detect via `std::is_base_of<NonBlockingNodeBase, Child>` (the
  shim keeps that marker base) and wire the `poll` trampoline + `notify_*`.
- **Control tokens → node hooks.** `Source::handleEvent` currently only surfaces EOF
  as a marker (transition compatibility). Once nodes are control-aware it dispatches
  `FlushStart`/`FlushStop` to optional `on_event` overrides and `Spec` to the
  `on_spec` hook (default: forward unchanged, §4.4.1); stateful Tier-S nodes
  (decoders, filters) implement `FlushStart` to reset internal state. Timeline is
  NOT delivered here — output/clock-sync stages read the master clock directly
  (§3.3).

---

## 3. Immediate TODO checklist (before porting a real node)

1. Pin avcpp + FFmpeg versions; determine the exact **adopt/wrap ctor** and ref
   semantics for `av::Packet`/`av::VideoFrame`/`av::AudioSamples`. Replace the
   `wrap_ref{}` placeholders. **Unit-test refcounts** (wrap→drop, wrap→put,
   **fan-out: push one frame to N edges → drop all N → refcount back to pre-push
   value, no double-free**, and wrap→mutate-in-place→verify-CoW) with
   `av_buffer_get_ref_count`, run under ASan/valgrind. See plan_v2 §8.2.1 for the
   fan-out / shared-buffer mutation invariants these tests enforce.
2. Implement `Source::refOf` (`av_frame_ref`/`av_packet_ref`/media-vtable retain) and
   confirm `peek`→`pop` keeps exactly one net ref.
3. Generate the `query_interface` X-macro table (the ~8 Register-3 entries, §4.4).
   Wire the `on_spec` bridge: for a node that reimplements a Fact interface
   (`rescale_video`/`filters`), read its output getters after processing the input
   `Spec` and emit the resulting `Spec`; for a node that read an upstream Fact at
   init (encoder), feed the latched input `Spec` into the fields the old
   `findNodeUp<IVideoFormatSource>()` pulled. No `avp_find_interface_up` (deleted).
4. Wire non-blocking detection + the `poll`/`notify_*` path; port `firewall` (Tier R)
   and `rescale_video` (Tier S) as the two proofs.
5. Decide `MetadataFrame` field access (design open Q §12.3) — opaque-move is enough
   for C++↔C++; only add C accessors when a Rust node needs to read it.
