# Rust Refactor — Draft Headers (`avplumber_core.h` + `node_common` shim)

> Companion to `rust_refactor_plan_v2.md` (design) and
> `rust_refactor_impl_breakdown.md` (language split). This file contains the
> concrete M0/M2 deliverables as drafts:
>
> 1. **`avplumber_core.h`** — the C ABI the Rust core exports and both C++ and Rust
>    nodes call. Pure C, no `std::`, no C++ types cross it. **Framework vocabulary
>    only** (plan-v2 §4.8): it names no clock, timeline, decoder, or seek.
> 1.1 **`avplumber_interfaces.h`** — Register-3 node capability ids + vtables.
> 1.2 **`avplumber_services_*.h`** — one header per core service (clock, timeline),
>    each owning its `AvpServiceId` and vtable, reached via `avp_core_query_service`.
> 2. **`avplumber_node_compat.hpp`** — the C++ shim that `node_common.hpp` pulls in,
>    so existing Tier-S nodes recompile unchanged in the common case — the
>    exceptions are enumerated in breakdown §3.5 (breakdown §3).
>
> Status: **draft skeleton.** avcpp method names (`.raw()`, wrap/adopt semantics)
> and a few FFmpeg accessors may need adjustment against the pinned avcpp/FFmpeg
> versions. Marked `TODO` where a decision is deferred. These are meant to be
> dropped into `src/` and compiled, then iterated.
>
> Note that the core's *internal* model is not `AvpBuffer` — it is owned `Media`
> (native-core §2) — but that is invisible here by design: this file describes the
> compat layer.
>
> **Section-reference convention** (four documents, so bare `§` is ambiguous): a plain
> `§N` means *this file*. `plan-v2 §N` means `rust_refactor_plan_v2.md`.
> `native-core §N` means `rust_refactor_native_core.md`. `breakdown §N` means
> `rust_refactor_impl_breakdown.md`.

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

/* ------------------------------------------------------------------ handles
 * Opaque. C never sees inside. destroy() returns ownership to the core. */
typedef struct AvpCore     AvpCore;      /* one instance (graph + scheduler)   */
typedef struct AvpNode     AvpNode;      /* a node instance (Rust- or C++-made) */
typedef struct AvpEdge     AvpEdge;      /* one edge endpoint bound to a node   */
typedef struct AvpExecutor AvpExecutor;  /* a clock domain / cooperative loop   */
typedef struct AvpGroup    AvpGroup;     /* a supervisor unit (plan-v2 §5.3)    */

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
 * transit (plan-v2 §3.3). No epoch: a seek clears queues rather than tagging
 * buffers (plan-v2 §3.4). */
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
/* In-band CAUSAL control on edges (plan-v2 §3.2). Takes effect where/when it
 * arrives; nothing is applied retroactively to buffers already downstream.
 * There is NO segment event and NO epoch — rate/offset/pause live on the master
 * clock (AvpSyncGroup below), applied at the output. FLUSH preempts the pipe:
 * it clears queued buffers on the way down (plan-v2 §3.4). SPEC (stream format)
 * is causal AND latched on the edge: the edge re-presents its current SPEC as the
 * head item to any (re)connecting consumer, before any buffer — so there is no
 * upstream pull walk to recover format (plan-v2 §§4.4.1, 3.6). This replaces the
 * old IVideoFormatSource/IAudioMetadataSource findNodeUp() queries entirely. */
typedef enum {
    AVP_EV_EOF         = 1,
    AVP_EV_FLUSH_START = 2,   /* preempts: clears queues downstream */
    AVP_EV_FLUSH_STOP  = 3,
    AVP_EV_SPEC        = 4    /* uses .spec; latched on the edge */
} AvpEventType;

/* Stream format description (plan-v2 §4.4.1): the resolved values that flow through
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
    /* audio: AVChannelLayout flattened into three POD fields, because the
     * layout is a struct with a heap pointer in the CUSTOM/AMBISONIC cases and
     * a pointer cannot cross a stable C ABI by value. Those two orders degrade
     * to UNSPEC + nb_channels — the per-channel map is not carried across this
     * boundary (native-core §2.2). */
    int          sample_rate;
    int          sample_format;      /* AVSampleFormat                    */
    int          channel_order;      /* AVChannelOrder (UNSPEC/NATIVE/...) */
    int          nb_channels;        /* the only meaningful field for UNSPEC */
    uint64_t     channel_mask;       /* AV_CH_* bitmask; valid iff order==NATIVE */
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

/* Consumer side. timeout_ms: <0 block, 0 poll, >0 bounded. */
void avp_edge_pop(AvpEdge*);
int  avp_edge_occupied(AvpEdge*);

/* Peek is an acquire/release/consume triple, not a bare-pointer borrow: "borrowed
 * until pop, ref it to keep it longer" is not enforceable from a raw pointer, and one
 * node in the tree already violates it (native-core §4.3.1). The borrow is a handle
 * with a lifetime.
 * Acquire returns NULL if there is nothing to peek; *out is valid only until the
 * handle is released or consumed. Exactly one of _release / _consume must be called. */
typedef struct AvpPeek AvpPeek;
AvpPeek* avp_edge_peek(AvpEdge*, int timeout_ms, AvpItem* out);
void     avp_edge_peek_release(AvpPeek*);
/* Pops. `out` NULLABLE, and that is the ownership choice: NULL -> the core releases
 * its ref (caller already ref'd from the borrow, which is what the avcpp shim does);
 * non-NULL -> the core MOVES its ref to *out and the caller must release it. */
int      avp_edge_peek_consume(AvpPeek*, AvpBuffer* out /*nullable*/);

/* Ownership-transferring take, for Source<T>::get() (native-core §4.3): removes the
 * head buffer and moves the core's reference to the caller. Zero refcount operations
 * on the common path. 1 = got, 0 = none. */
int  avp_edge_take(AvpEdge*, int timeout_ms, AvpItem* out);

/* Current latched SPEC of this edge (plan-v2 §4.4.1). Normally a node just receives
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
/* Capability discovery is DIRECT only — you query a node you already hold
 * (plan-v2 §4.4). There is NO avp_find_interface_up / graph walk: the three registers
 * the old findNodeUp<T>() served are now split —
 *   - Facts  (format: IVideoFormatSource/IAudioMetadataSource/IFrameRateSource/
 *             ITimeBaseSource) -> AvpSpec, latched in-band on the edge (above).
 *   - Controls (IPlaybackControl direction, IInputReset, seek/speed/pause) ->
 *             addressed to core services (the clock + the flush discontinuity),
 *             reached via avp_core_query_service below.
 *   - Live queries -> avp_node_query_interface, on a named/adjacent node.
 * So only the Register-3 capabilities remain queryable. The mechanism is here;
 * the ids and their per-interface vtables live in avplumber_interfaces.h (§1.1),
 * which this header does NOT include — a node capability is domain vocabulary. */
const void* avp_node_query_interface(AvpNode*, uint32_t iface_id);

/* -------------------------------------------------- service discovery (§4.8) */
/* Core services (the SyncGroup clock, SharedTimeline, seek) are core-owned but
 * domain-specific, so they are NOT named functions here: an embedder asks for a
 * service by id and gets a const vtable. This keeps avplumber_core.h purely
 * framework vocabulary — nothing in it names a clock, a timeline, or a seek.
 * Each service's id + vtable lives in its own avplumber_services_*.h (§1.2). */
typedef uint32_t AvpServiceId;   /* stable, append-only; see avplumber_services_*.h */

/* Returns a const service vtable, or NULL if that service isn't registered in
 * this build (so an embedder can degrade rather than fail to link). */
const void* avp_core_query_service(AvpCore*, AvpServiceId);

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

/* ------------------------------------------- graph-management ABI (plan-v2 §4.7) */
/* The surface an EMBEDDER calls to drive the core — the TCP control server,
 * pyplumber, an OBS plugin, or any host linking the core as a library: build a
 * graph, connect edges, start/stop groups. The TCP server is implemented ON TOP
 * of these, so it is one entry point rather than the only one. */

/* Edge coupling hint. NULL = let the core decide per plan-v2 §5.3: DirectEdge if
 * both endpoints share one executor and neither is blocking, else BufferedEdge. */
typedef struct {
    int    is_direct;   /* 0 = buffered (queue), 1 = direct (push-to-consumer) */
    size_t capacity;    /* 0 = core default; ignored when is_direct           */
} AvpEdgeCoupling;

/* Construction. Params are UTF-8 JSON — the same text the control protocol and
 * .avplumber scripts already carry, so no schema is duplicated in C. Returns
 * NULL on error with *err set to a caller-freed (avp_string_free) message. */
AvpNode* avp_create_node(AvpCore*, const char* type_name, const char* instance_name,
                         const char* json_params, const char** err);
AvpEdge* avp_create_edge(AvpCore*, const char* name,
                         AvpNode* producer, const char* out_pad,
                         AvpNode* consumer, const char* in_pad,
                         const AvpEdgeCoupling* coupling /* NULL = default */);

/* Groups (supervisor units, plan-v2 §5.3): ordered start/stop + restart policy. */
AvpGroup* avp_create_group (AvpCore*, const char* name);
void      avp_group_add    (AvpGroup*, AvpNode*);
void      avp_group_remove (AvpGroup*, AvpNode*);

/* Lifecycle. start is idempotent on a started group; stop drains edges (§3
 * flush semantics) before joining threads; destroy returns ownership to the core. */
int   avp_start_group   (AvpGroup*, const char** err);   /* 0 ok, -1 err */
int   avp_stop_group    (AvpGroup*, const char** err);
void  avp_destroy_node  (AvpCore*, AvpNode*);
void  avp_destroy_edge  (AvpCore*, AvpEdge*);
void  avp_destroy_group (AvpCore*, AvpGroup*);

/* Introspection, for embedders that didn't build the graph themselves. */
AvpNode*  avp_lookup_node (AvpCore*, const char* name);
AvpGroup* avp_lookup_group(AvpCore*, const char* name);

/* --------------------------------------------------------- core lifecycle */
/* Usually the Rust binary owns main(); these exist for the static-lib/OBS embed. */
AvpCore* avp_core_create(void);
void     avp_core_destroy(AvpCore*);
/* Control protocol, implemented on top of the §4.7 calls above. */
int      avp_core_exec_command(AvpCore*, const char* line, char** out_reply);
int      avp_core_serve_tcp(AvpCore*, uint16_t port);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* AVPLUMBER_CORE_H */
```

### Design notes on the C ABI

- **Pull, not push.** `process()`/`poll()` take no item; nodes pull via
  `avp_edge_take`/`avp_edge_peek`. This is faithful to today's `source_->get()`, makes
  the shim a thin wrapper, and means legacy node bodies don't invert — a push
  `poll(item)` would force exactly that.
- **One item stream.** Buffers and control tokens share one ordered queue
  (`AvpItem`), so ordering between a flush and the buffers around it is intrinsic;
  a flush additionally preempts (clears) the queue it crosses (plan-v2 §3.4).
- **Ownership is stated per call.** `push` takes the ref on `PUSHED`; `take` moves the
  ref out; `peek` lends it for the lifetime of an `AvpPeek*`. This is the whole
  refcount contract the shim must honor (breakdown §3.2). "`peek` borrows until
  `pop`" would be a rule stated in prose that nothing checks — and the tree already
  breaks it; the `AvpPeek` handle makes the borrow's end a call rather than a
  convention (native-core §4.3.1).
- **Interface vtables are `const` singletons per node type**; `query_interface`
  returns a pointer to a static struct whose fns re-enter the node. Only the
  Register-3 live-query interfaces (plan-v2 §4.4) exist, in `avplumber_interfaces.h`
  (§1.1); a couple are shown there, the rest are mechanical. Facts (format) are NOT
  interfaces — they are `AvpSpec` on the edge.
- **The core header names no domain concept** (plan-v2 §4.8). It has handles, media,
  edges, nodes, groups, factories, and two discovery *mechanisms* — and nothing
  called a clock, a timeline, a decoder, or a seek. Those live in the companion
  headers below, so the framework can be read, versioned, and reimplemented without
  inheriting avplumber's domain vocabulary. This is why there are no
  `avp_clock_*` / `avp_timeline_*` / `avp_sync_group` functions above: an embedder
  reaches a service through `avp_core_query_service` and a vtable.

### 1.1 `avplumber_interfaces.h` — Register-3 capability ids + vtables

Node capabilities, keyed to `avp_node_query_interface`. Stable and **append-only**,
which is why removed ids leave holes rather than being renumbered:

```c
typedef enum {
    AVP_IFACE_DECODER         = 1,   /* codec name, discard_until          */
    /* 2 and 3 were AVP_IFACE_ENCODER / AVP_IFACE_MUXER; removed — stream
     * config is AvpSpec on the edge (Register-1 Fact), not a query. Do not
     * reuse 2/3. See plan-v2 §4.4. */
    AVP_IFACE_SENTINEL        = 4,   /* card/signal-present stats          */
    AVP_IFACE_RETURNS_OBJECTS = 5,   /* node.param.get bridge              */
    AVP_IFACE_INPUTS_OBJECTS  = 6,   /* node.param.set bridge              */
    AVP_IFACE_STREAMS_INPUT   = 7,   /* demux stream enumeration           */
    /* append only */
} AvpInterfaceId;

typedef struct {                     /* AVP_IFACE_DECODER — reference pattern */
    const char* (*codec_name)(AvpNode*);
    const char* (*media_type_string)(AvpNode*);
    void        (*discard_until)(AvpNode*, int64_t pts, AvpRational tb);
} AvpIDecoder;

typedef struct {                     /* AVP_IFACE_RETURNS_OBJECTS */
    char* (*get_object)(AvpNode*, const char* name);   /* caller: avp_string_free */
} AvpIReturnsObjects;

typedef struct {                     /* AVP_IFACE_INPUTS_OBJECTS */
    void (*set_object)(AvpNode*, const char* name, const char* json);
} AvpIInputsObjects;
/* AvpISentinel / AvpIStreamsInput: same shape, filled in at M2. */
```

Two absences are load-bearing:

- **No `AVP_IFACE_ENCODER` / `AVP_IFACE_MUXER`.** The old C++ handshake
  (`initFromFormatContext*` / `setOutput` / `openEncoder` / `setOutputPostOpen`) was a
  workaround for `findNodeUp`'s inability to fork across a muxer's N inputs, plus a
  query-relay shim on `bsf`/`packet_relay`. With `Spec` on the edge the muxer
  aggregates per-stream `Spec`s exactly as it aggregates packets, relays transform
  `Spec` as they transform packets, and the encoder self-opens. No bidirectional
  negotiation survives (plan-v2 §4.4.2).
- **No `AVP_IFACE_JACK_SINK`.** `JackClient` holds `weak_ptr<IJackSink>` and calls
  *into* the sink (`jack_process`) from JACK's real-time thread
  (`src/JackClient.hpp:16,52`, sole implementer `src/nodes/jack/jack_sink.cpp:17`).
  That is callback registration on a service, not a capability query — so it is a
  dedicated `avp_jack_client_add_sink` on the JACK service at M2, and both nodes stay
  C++ (Tier S).

### 1.2 `avplumber_services_*.h` — one header per core service

Each service owns its id and its vtable. The pattern, for the two services the design
needs at M3:

```c
/* avplumber_services_clock.h — the SyncGroup master clock (plan-v2 §3.3, §6).
 * SyncGroup is a TRAIT, not a struct: the core owns the contract, the impl is
 * pluggable (WallClock / SourceTimeClock / SyntheticClock). Rate/offset/pause are
 * O(1) writes here; buffers keep source-time PTS and are mapped only when an
 * OUTPUT stage releases them. A seek is one shared reset for the whole group. */
#define AVP_SERVICE_CLOCK  ((AvpServiceId)1)
typedef struct AvpSyncGroup AvpSyncGroup;
typedef struct {
    AvpSyncGroup* (*create)     (AvpCore*, const char* name);  /* get/create by name */
    void          (*set_rate)   (AvpSyncGroup*, double rate);  /* <0 = reverse       */
    void          (*set_paused) (AvpSyncGroup*, int paused);    /* freeze/thaw output */
    void          (*reset)      (AvpSyncGroup*, int64_t new_pos, AvpRational tb);
    /* Output-stage read. AVP_NOPTS while paused-and-not-yet-due: the stage sleeps. */
    int64_t       (*map_to_wall)(AvpSyncGroup*, int64_t src_pts, AvpRational tb);
} AvpSyncGroupVtable;

/* avplumber_services_timeline.h — SharedTimeline (plan-v2 §3.3, §6), was
 * InstanceShared<SharedTimeline>: a named, PTS-keyed key/value store. NOT on the
 * data plane — the control protocol WRITES scheduled values at a source-time PTS
 * and C++ Tier-S nodes (one_to_many, source_switcher, preheat_video_router,
 * cuda_rect_overlay via TimelineReader) READ "the value in effect at this PTS".
 * Values are opaque JSON so the ABI needn't model the Parameters variant. It
 * crosses in BOTH directions, which is why it needs a C surface at all. */
#define AVP_SERVICE_TIMELINE  ((AvpServiceId)2)
typedef struct AvpTimeline AvpTimeline;
typedef struct {
    AvpTimeline* (*create)   (AvpCore*, const char* name);
    void         (*set)      (AvpTimeline*, const char* channel, const char* key,
                              int64_t at_pts_ms, const char* value_json);
    void         (*clear_key)(AvpTimeline*, const char* channel, const char* key);
    void         (*gc)       (AvpTimeline*, int64_t before_pts_ms);
    /* Latest value with at_pts_ms <= frame_pts (compared via tb). Returns the
     * JSON length, 0 if no entry applies, or the needed length (>cap) if truncated. */
    int          (*get)      (AvpTimeline*, const char* channel, const char* key,
                              int64_t frame_pts, AvpRational tb, char* out, size_t cap);
} AvpTimelineVtable;
```

A caller does `query_service(core, AVP_SERVICE_CLOCK)`, checks for NULL, then calls
through the vtable. NULL means "not in this build", so a host can degrade instead of
failing to link — which a named `avp_clock_set_rate` could not offer.

---

## 2. `avplumber_node_compat.hpp` — the C++ shim

`node_common.hpp` becomes a one-line include of this. Existing Tier-S nodes keep
their source (plan-v2 §3).

```cpp
/* avplumber_node_compat.hpp — C++ compatibility shim over avplumber_core.h.
 * Re-implements the old node_common.hpp surface (Node, NodeSISO, Source/Sink,
 * DECLNODE*, interfaces) backed by the C ABI. Draft skeleton.
 */
#pragma once
#include "avplumber_core.h"
#include "avplumber_interfaces.h" /* Register-3 ids + vtables (§1.1)          */
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
/* Two things to note (native-core §§4.3, 4.3.1):
 *   - get() uses avp_edge_take, so it costs no refcount operation.
 *   - peek() returns a move-only borrow guard, never a T*. The T* form is already
 *     unsound in the current C++ core, so the shim must not offer it.
 *   Nodes' `if (p)` / `p->pts()` idioms keep compiling via operator bool/->;
 *   `T* p = src->peek();` deliberately does NOT (use `auto`). */

template<typename T> class Source;   /* fwd: Peeked's only constructor is private */

/* Move-only RAII borrow of an edge's head item. Owns the avcpp object it exposes.
 * Drop = leave the item queued. consume() = pop and take the object. */
template<typename T> class Peeked {
    AvpPeek* h_ = nullptr;
    T        obj_;                  /* built once, with the native-core §3 bridges */
    friend class Source<T>;
    Peeked(AvpPeek* h, T&& o): h_(h), obj_(std::move(o)) {}
public:
    Peeked() = default;
    Peeked(const Peeked&) = delete;
    Peeked& operator=(const Peeked&) = delete;
    Peeked(Peeked&& o) noexcept: h_(o.h_), obj_(std::move(o.obj_)) { o.h_ = nullptr; }
    ~Peeked() { if (h_) avp_edge_peek_release(h_); }

    explicit operator bool() const { return h_ != nullptr; }
    const T* operator->() const { return &obj_; }
    const T& operator*()  const { return obj_; }
    T*       mut()              { return &obj_; }   /* named: mutation is deliberate */

    /* Pop the item and hand over the object. Pass NULL: this guard already holds a
     * ref (Media<T>::wrap ref'd it), so the core must RELEASE its own, not move it. */
    T consume() {
        avp_edge_peek_consume(h_, nullptr);
        h_ = nullptr;
        return std::move(obj_);
    }
};

template<typename T> class Source {
    AvpEdge* edge_;
public:
    using DataType = T;
    explicit Source(AvpEdge* e): edge_(e) {}

    /* blocking get: returns next buffer, skipping/handling events (EOF surfaces as
     * an EOF-marker T so isEofMarker() still works during the transition). */
    T get(int timeout_ms = -1) {
        AvpItem it;
        while (avp_edge_take(edge_, timeout_ms, &it)) {
            if (it.is_event) { if (handleEvent(it.event)) return makeEofMarker<T>();
                continue; }                       /* take already advanced the edge */
            return avpshim::Media<T>::wrap(it.buffer.ptr);  /* adopts the moved ref */
        }
        return T();   /* finished */
    }
    /* non-blocking peek (timeout 0). The returned guard bounds the borrow. */
    Peeked<T> peek(int timeout_ms = 0) {
        AvpItem it;
        AvpPeek* h = avp_edge_peek(edge_, timeout_ms, &it);
        if (!h) return Peeked<T>();
        if (it.is_event) { handleEvent(it.event); avp_edge_peek_release(h);
            return Peeked<T>(); }
        /* ref the frame so the guard's object is independently valid */
        return Peeked<T>(h, avpshim::Media<T>::wrap(refOf(it.buffer)));
    }
    bool pop() { avp_edge_pop(edge_); return true; }
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
    /* Format arrives as a latched SPEC on the input edge (plan-v2 §4.4.1), routed to
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

/* EdgeManager is now just the binding context — an AvpNode handle. It keeps the
 * `createCommon(edges, params, ...)` and `init(edges, params)` signatures byte-identical
 * at all 49 call sites, so nodes need no edit; the edge lookup underneath became
 * avp_node_bind_*. Nodes only ever pass it through, never call it. */
class EdgeManager {
    AvpNode* node_;
public:
    explicit EdgeManager(AvpNode* n): node_(n) {}
    AvpNode* node() const { return node_; }
};

template<typename InT, typename OutT>
class NodeSISO : public NodeSingleInput<InT>, public NodeSingleOutput<OutT> {
public:
    /* Byte-identical to the signature the tree already calls (graph_base.hpp:388):
     * same parameter list, same shared_ptr return, same ctor-forwarding. Only the
     * body changed — edges.find<T>(name) became avp_node_bind_*. */
    template<typename Child, typename... Args>
    static std::shared_ptr<Child> createCommon(EdgeManager& edges, const Parameters& p,
                                               Args&&... args) {
        AvpNode* node = edges.node();
        auto src = std::make_unique<Source<InT>>(
            avp_node_bind_source(node, p.at("src").get<std::string>().c_str(),
                                 avpshim::Media<InT>::tag, 0));
        auto dst = std::make_unique<Sink<OutT>>(
            avp_node_bind_sink(node, p.at("dst").get<std::string>().c_str(),
                               avpshim::Media<OutT>::tag, 0));
        /* Child(source, sink, args...) — the ctor the tree's nodes inherit via
         * `using NodeSISO<T,T>::NodeSISO;`. Unchanged. */
        auto r = std::make_shared<Child>(std::move(src), std::move(dst),
                                         std::forward<Args>(args)...);
        r->bind(node);
        return r;
    }
};

/* -------------------------------------------------- vtable + query_interface glue */
namespace avpshim {

/* avp_node_impl holds a heap `shared_ptr<Child>*`, not a raw Child*: nodes call
 * shared_from_this() (graph_core.hpp:90,130,152), which needs a live control block. */
template<typename Child> Child* impl(AvpNode* n) {
    return static_cast<std::shared_ptr<Child>*>(avp_node_impl(n))->get();
}

/* Blocking process trampoline */
template<typename Child> AvpFlow proc_tramp(AvpNode* n) {
    try { impl<Child>(n)->process(); }
    catch (std::exception& e) { logstream << e.what(); return AVP_FLOW_ERROR; }
    return AVP_FLOW_PUSHED;
}
template<typename Child> void start_tramp(AvpNode* n){ impl<Child>(n)->start(); }
template<typename Child> void stop_tramp (AvpNode* n){ impl<Child>(n)->stop();  }
template<typename Child> void dtor_tramp (AvpNode* n){
    delete static_cast<std::shared_ptr<Child>*>(avp_node_impl(n));   /* drops the ref */
}

/* query_interface: dynamic_cast<IFoo*>(child) -> a static C vtable per interface.
 * Shown for IDecoder; the rest via the QI_ENTRY x-macro below. */
template<typename Child> const AvpIDecoder* qi_decoder(AvpNode* n) {
    if (!dynamic_cast<IDecoder*>(impl<Child>(n))) return nullptr;
    static const AvpIDecoder vt = {
        /* codec_name */ [](AvpNode* n)->const char*{
            static thread_local std::string s;
            s = dynamic_cast<IDecoder*>(impl<Child>(n))->codecName();
            return s.c_str(); },
        /* media_type_string */ [](AvpNode* n)->const char*{
            static thread_local std::string s;
            s = dynamic_cast<IDecoder*>(impl<Child>(n))->codecMediaTypeString();
            return s.c_str(); },
        /* discard_until */ [](AvpNode* n, int64_t pts, AvpRational tb){
            dynamic_cast<IDecoder*>(impl<Child>(n))
                ->discardUntil(av::Timestamp(pts, {tb.num, tb.den})); }
    };
    return &vt;
}

template<typename Child> const void* query_interface(AvpNode* n, uint32_t id) {
    switch (id) {
        case AVP_IFACE_DECODER: return qi_decoder<Child>(n);
        /* QI_ENTRY(AVP_IFACE_SENTINEL, qi_sentinel<Child>) ... — the other
         * Register-3 ids (§1.1). No *_VIDEO_FORMAT_SOURCE (that Fact is served by
         * on_spec), no *_ENCODER / *_MUXER (deleted), no *_JACK_SINK (direct
         * registration on the JACK service). */
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

/* Factory: parse JSON, call Child::create(nci), attach impl+vtable.
 * Child::create returns shared_ptr (unchanged from today), and nodes use
 * shared_from_this (graph_core.hpp:90,130,152), so the shim must keep the
 * control block alive: the AvpNode owns a heap shared_ptr, released by dtor_tramp. */
template<typename Child>
AvpNode* factory(AvpCore* core, AvpNode* node, const char* json) {
    Parameters params = Parameters::parse(json);
    EdgeManager edges { node };
    InstanceData& inst = instance_for(core);            /* shim-side, see below */
    NodeManager&  nmgr = nodes_for(core);
    NodeCreationInfo nci { edges, params, inst, nmgr };
    std::shared_ptr<Child> c = Child::create(nci);       /* node's own create()  */
    avp_node_set_impl(node, new std::shared_ptr<Child>(c),
                      vtable_for<Child>(/*nonblocking=*/false)); /* TODO detect */
    return node;
}

struct Registrar {
    Registrar(const char* type, AvpModuleInit init) { avp_register_module_init(init); (void)type; }
};

} /* namespace avpshim */

/* NodeCreationInfo keeps the old struct's four members and their names, because
 * nodes destructure it by name: nci.edges (95 sites), nci.params (121),
 * nci.instance (29), nci.nodes (2). Same shape, re-backed:
 *   - edges    -> the AvpNode binding context (above)
 *   - params   -> unchanged (nlohmann JSON)
 *   - instance -> InstanceSharedObjects<T>::get() over avp_shared_get; the Team
 *                 lookups among the 29 sites instead resolve to core services
 *                 (plan-v2 §3.3), which is the one behavioural change here
 *   - nodes    -> avp_lookup_node / avp_lookup_group (§4.7); only 2 sites
 *                 (speed.cpp:265 sync_node, _unfinished/source_switcher.cpp:95) */
struct NodeCreationInfo {
    EdgeManager&      edges;
    const Parameters& params;
    InstanceData&     instance;
    NodeManager&      nodes;
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
  `unwrap`+`release_ownership` (move ref into the edge), and `refOf` (the guard's own
  ref in `peek`) must exactly match avcpp's `AVFrame`/`AVPacket` ownership. Pin the
  avcpp version and unit-test ref counts before porting real nodes. avcpp's exact
  adopt/wrap ctor spelling (`wrap_ref{}` here is a placeholder) is the first thing to
  nail down.
- **The `Peeked<T>` guard is the shim's one non-mechanical abstraction.** It is small,
  but it is what keeps the borrow's lifetime tied to a value rather than to a comment.
  Write it and its refcount tests before any node is recompiled against the shim, and
  convert the C++ tree to it *first*, against the old core, where the change is
  testable in isolation (native-core §4.3.1, breakdown §4 milestone M-pre).
- **`query_interface` is mechanical but bulky.** Only `IDecoder` is written out; the
  other 4 Register-3 interfaces (§1.1) follow identically. Generate them with an
  X-macro over `(iface_id, cxx_interface, {method trampolines})` so adding one is a
  single line — mirroring how the old `DECLNODE` avoided boilerplate. (Fact
  interfaces are NOT here; they route through `on_spec`, below.)
- **The `NodeCreationInfo` / `createCommon` signatures do not change.** This is what
  keeps 49 `createCommon` call sites and 247 `nci.*` member reads compiling untouched:
  `EdgeManager` becomes a one-field binding context wrapping the `AvpNode*`, and the
  edge lookup inside `createCommon` becomes `avp_node_bind_*`. A node cannot tell the
  difference — which is the whole point of the shim being a *compat* layer.
- **`avp_node_impl` stores a heap `shared_ptr<Child>*`.** Nodes call
  `shared_from_this()` (`src/graph_core.hpp:90,130,152`) and `Child::create` returns a
  `shared_ptr`, so a raw `Child*` in the impl slot would leave the control block
  unowned and `shared_from_this()` throwing. `dtor_tramp` deletes the `shared_ptr`,
  dropping one ref.
- **Blocking vs non-blocking detection.** `vtable_for<Child>(nonblocking)` currently
  hardcodes blocking; detect via `std::is_base_of<NonBlockingNodeBase, Child>` (the
  shim keeps that marker base) and wire the `poll` trampoline + `notify_*`.
- **Control tokens → node hooks.** `Source::handleEvent` currently only surfaces EOF
  as a marker (transition compatibility). Once nodes are control-aware it dispatches
  `FlushStart`/`FlushStop` to optional `on_event` overrides and `Spec` to the
  `on_spec` hook (default: forward unchanged, plan-v2 §4.4.1); stateful Tier-S nodes
  (decoders, filters) implement `FlushStart` to reset internal state. Timeline is
  NOT delivered here — output/clock-sync stages read the master clock directly
  (plan-v2 §3.3).

---

## 3. Immediate TODO checklist (before porting a real node)

1. Pin avcpp + FFmpeg versions; determine the exact **adopt/wrap ctor** and ref
   semantics for `av::Packet`/`av::VideoFrame`/`av::AudioSamples`. Replace the
   `wrap_ref{}` placeholders. **Unit-test refcounts** (wrap→drop, wrap→put,
   **fan-out: push one frame to N edges → drop all N → refcount back to pre-push
   value, no double-free**, and wrap→mutate-in-place→verify-CoW) with
   `av_buffer_get_ref_count`, run under ASan/valgrind. See plan-v2 §8.2.1 for the
   fan-out / shared-buffer mutation invariants these tests enforce.
2. Implement `Source::refOf` (`av_frame_ref`/`av_packet_ref`/media-vtable retain) and
   confirm the guard's paths keep exactly one net ref: peek→drop (item still queued,
   count back to entry), peek→`consume()` (count unchanged — `_consume(h, NULL)`
   releases the core's ref while the guard keeps its own), and `take` (no refcount op
   at all).
3. Generate the `query_interface` X-macro table (the 5 Register-3 entries, §1.1).
   Wire the `on_spec` bridge: for a node that reimplements a Fact interface
   (`rescale_video`/`filters`), read its output getters after processing the input
   `Spec` and emit the resulting `Spec`; for a node that read an upstream Fact at
   init (encoder), feed the latched input `Spec` into the fields the old
   `findNodeUp<IVideoFormatSource>()` pulled. No `avp_find_interface_up` (deleted).
4. Wire non-blocking detection + the `poll`/`notify_*` path; port `firewall` (Tier R)
   and `rescale_video` (Tier S) as the two proofs.
5. Decide `MetadataFrame` field access (plan-v2 §12 item 3, open question) —
   opaque-move is enough for C++↔C++; only add C accessors when a Rust node needs to
   read it.
