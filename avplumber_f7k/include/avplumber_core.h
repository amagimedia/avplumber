/* avplumber_core.h - C ABI exported by the Rust core.
 *
 * Framework surface only. Domain vocabulary (SyncGroup clock, SharedTimeline,
 * seek) lives in avplumber_services_*.h. Node capabilities (decoder, sentinel,
 * ...) live in avplumber_interfaces.h.
 *
 * Rules: opaque handles; explicit ownership; UTF-8 char*; JSON params as strings;
 * no C++ types; stable, append-only enums. C11.
 */
#ifndef AVPLUMBER_CORE_H
#define AVPLUMBER_CORE_H

#include <stdint.h>
#include <stddef.h>
#include "avplumber_ids.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ handles
 * Opaque. C never sees inside. destroy() returns ownership to the core. */
typedef struct AvpCore     AvpCore;      /* one instance (graph + scheduler)    */
typedef struct AvpNode     AvpNode;      /* a node instance (Rust- or C++-made)  */
typedef struct AvpEdge      AvpEdge;      /* one edge endpoint bound to a node    */
typedef struct AvpExecutor  AvpExecutor;  /* a clock domain / cooperative loop    */
typedef struct AvpGroup     AvpGroup;    /* supervisor unit: ordered start/stop */

/* --------------------------------------------------------------- primitives */
typedef struct { int num; int den; } AvpRational;

#define AVP_NOPTS INT64_MIN

/* -------------------------------------------------------------- media types */
typedef enum {
    AVP_MEDIA_PACKET   = 1,  /* ptr = AVPacket*                    */
    AVP_MEDIA_VIDEO     = 2,  /* ptr = AVFrame*                      */
    AVP_MEDIA_AUDIO     = 3,  /* ptr = AVFrame*                      */
    AVP_MEDIA_EGL       = 4,  /* ptr = opaque EglImageFrame* (C++)  */
    AVP_MEDIA_METADATA  = 5   /* ptr = opaque MetadataFrame*  (C++)  */
} AvpMediaType;

/* One media buffer crossing the boundary. Rust passes ONE reference in; the
 * receiver owns it until it either frees it or forwards it via avp_edge_push.
 * PTS lives on the AVFrame/AVPacket, in source time, and is never rewritten in
 * transit. A seek clears queues rather than tagging buffers with an epoch. */
typedef struct {
    AvpMediaType type;
    void*        ptr;     /* AVFrame* or AVPacket* for 1..3; opaque for 4..5 */
} AvpBuffer;

/* For opaque C++-owned media (EGL/Metadata), the C++ side registers how Rust may
 * move/own it without understanding it. AVFrame/AVPacket need no vtable (Rust
 * uses FFmpeg C functions directly). */
typedef struct {
    void    (*retain)(void* obj);
    void    (*release)(void* obj);
    int64_t (*get_pts)(void* obj);                 /* AVP_NOPTS if none      */
    void    (*get_time_base)(void* obj, AvpRational* out);
} AvpMediaVtable;

void avp_register_media_type(AvpCore*, AvpMediaType, const AvpMediaVtable*);

/* -------------------------------------------------------------- edge events
 * In-band causal control on edges. Takes effect where/when it arrives;
 * nothing is applied retroactively to buffers already downstream. There is
 * no segment event and no epoch: rate/offset/pause live on the master clock
 * (avplumber_services_clock.h) and are applied at the output. FLUSH preempts
 * the pipe: it clears queued buffers on the way down. SPEC (stream format)
 * is causal and latched on the edge: a (re)connecting consumer sees the
 * current SPEC as the head item, before any buffer, so it does not walk
 * upstream to recover format. */
typedef enum {
    AVP_EV_EOF         = 1,
    AVP_EV_FLUSH_START  = 2,   /* preempts: clears queues downstream */
    AVP_EV_FLUSH_STOP   = 3,
    AVP_EV_SPEC        = 4    /* uses .spec; latched on the edge */
} AvpEventType;

/* Stream format: the resolved values that flow through an edge. Latched on
 * the edge; not a GStreamer capability set and not negotiated — a single
 * concrete "this is what flows here now", forward-only. */
typedef struct {
    AvpMediaType media;              /* VIDEO or AUDIO                     */
    /* video */
    int          width, height;
    int          pixel_format;        /* AVPixelFormat                     */
    AvpRational  frame_rate;
    AvpRational  sample_aspect_ratio;
    /* audio */
    int          sample_rate;
    int          sample_format;       /* AVSampleFormat                     */
    int          channel_order;       /* AVChannelOrder (UNSPEC/NATIVE/...); see edge.rs */
    int          nb_channels;         /* channel count; the only meaningful field for UNSPEC */
    uint64_t     channel_mask;        /* bitmask of AV_CH_*; valid when channel_order == NATIVE */
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
int  avp_edge_take(AvpEdge*, int timeout_ms, AvpItem* out);  /* 1 got, 0 none — ownership moves */
typedef struct AvpPeek AvpPeek;
AvpPeek* avp_edge_peek(AvpEdge*, int timeout_ms, AvpItem* out); /* NULL = none; *out borrowed */
void avp_edge_peek_release(AvpPeek*);
int  avp_edge_peek_consume(AvpPeek*, AvpBuffer* out /* nullable */);
void avp_edge_pop(AvpEdge*);
int  avp_edge_occupied(AvpEdge*);

/* Current latched SPEC of this edge. Normally a node just receives a SPEC
 * item as the head of its stream and need not call this; provided so a late
 * binder can read the format synchronously without consuming. Returns 1 and
 * fills *out if a SPEC has ever flowed, 0 if none. */
int  avp_edge_current_spec(AvpEdge*, AvpSpec* out);

/* Non-blocking wakeup (replaces processWhenSignalled / consumedEvent). The
 * core re-invokes the node's vtable.poll when the edge becomes
 * readable/writable. */
void avp_edge_notify_readable(AvpEdge*, AvpNode*);
void avp_edge_notify_writable(AvpEdge*, AvpNode*);

/* Bind a named edge to this node as source/sink of a given media type. Returns
 * the shared endpoint handle. capacity==0 uses the core's buffered default;
 * DirectEdge is selected only by explicit graph construction. */
AvpEdge* avp_node_bind_source(AvpNode*, const char* edge_name, AvpMediaType, size_t capacity);
AvpEdge* avp_node_bind_sink  (AvpNode*, const char* edge_name, AvpMediaType, size_t capacity);

/* --------------------------------------------------------- node vtable/api
 * Nodes PULL from their edges (avp_edge_peek). process()/poll() take no item.
 * The executor picks which to call based on which fn pointer is non-NULL:
 * process() for blocking nodes (own OS thread), poll() for cooperative nodes
 * (current-thread runtime per event loop / tick source). */
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

    /* Capability discovery. Returns a const per-interface vtable, or NULL.
     * Query this node only; there is no upstream walk. */
    const void* (*query_interface)(AvpNode*, uint32_t iface_id);
} AvpNodeVtable;

/* The core calls this to attach the C++/Rust object + vtable to the AvpNode. */
void  avp_node_set_impl(AvpNode*, void* self, const AvpNodeVtable*);
void* avp_node_impl(AvpNode*);
const char* avp_node_name(AvpNode*);

/* Core services (clock, timeline, seek) are core-owned but domain-specific.
 * An embedder reaches them by service ID. Numeric IDs are in avplumber_ids.h
 * (generated from Rust); each service's vtable lives in avplumber_services_*.h. */
/* Returns a const service vtable pointer, or NULL if the service isn't
 * registered in this build. The mechanism is in avplumber_core.h; each
 * service's ID + vtable lives in its own header. */
const void* avp_core_query_service(AvpCore*, AvpServiceId);

/* Capability query is on a node you already hold. There is no graph walk:
 *   - stream format travels as AvpSpec, latched in-band on the edge (above);
 *   - playback control (direction, reset, seek/speed/pause) is a core service
 *     (avplumber_services_clock.h);
 *   - live methods on a named/adjacent node use this table.
 * IDs are in avplumber_ids.h; vtables live in avplumber_interfaces.h. */
const void* avp_node_query_interface(AvpNode*, uint32_t iface_id);

/* ---------------------------------------------- factory registration
 * Replaces DECLNODE + generate_node_list. The factory creates the impl,
 * calls avp_node_set_impl, and returns the AvpNode (core-allocated, passed in). */
typedef AvpNode* (*AvpNodeFactoryFn)(AvpCore*, AvpNode* node, const char* json_params);

void avp_register_node_factory(AvpCore*, const char* type_name, AvpNodeFactoryFn);

/* Called once at load to let a translation unit self-register its factories,
 * before any graph is built. C++ shim drives this from static registrars. */
typedef void (*AvpModuleInit)(AvpCore*);
void avp_register_module_init(AvpModuleInit);

void avp_string_free(char*);

/* ---------------------------------------------- instance-shared registry
 * Generic string-keyed shared objects. Typed Rust services (clock, timeline,
 * correction) do not use this. */
void* avp_shared_get(AvpCore*, const char* type_key, const char* name);
void  avp_shared_put(AvpCore*, const char* type_key, const char* name,
                     void* obj, const AvpMediaVtable* ownership /* retain/release */);

/* Graph construction for embedders (TCP control, pyplumber, an OBS plugin,
 * or any host linking the core as a library). */

/* Edge coupling hint (NULL = core default: buffered). DirectEdge is selected
 * only by explicit construction, not inferred from co-location. Direct is
 * zero-queue: offer runs the consumer; backpressure is the end of a
 * Direct-only chain. Both endpoints must be cooperative poll nodes. */
typedef struct {
    int is_direct;   /* 0 = BufferedEdge; 1 = DirectEdge (capacity 0) */
    size_t capacity;  /* 0 = core default (64); ignored when is_direct */
} AvpEdgeCoupling;

/* Graph construction. Params are UTF-8 JSON. Returns NULL on error; *err
 * holds a caller-freed UTF-8 message. */
AvpNode* avp_create_node(AvpCore*, const char* type_name, const char* instance_name,
                         const char* json_params, const char** err);
AvpEdge* avp_create_edge(AvpCore*, const char* name,
                         AvpNode* producer, const char* out_pad,
                         AvpNode* consumer, const char* in_pad,
                         const AvpEdgeCoupling* coupling /* NULL = default */);

/* Groups: ordered lifecycle and generation-fenced restart policy. A node with
 * a non-default restart/error action must belong to exactly one group before
 * start/restart. JSON actions: off, group/restart_group, panic, exit.
 * on, boolean true, and restart_node retain their isolated-node meaning and
 * are rejected until isolated restart is implemented. */
AvpGroup* avp_create_group (AvpCore*, const char* name);
/* Compatibility wrapper: logs rejected membership. New code should use the
 * checked form so lifecycle and exactly-one-group policy errors are visible. */
void      avp_group_add    (AvpGroup*, AvpNode*);
int       avp_group_add_checked(AvpGroup*, AvpNode*, const char** err);
void      avp_group_remove (AvpGroup*, AvpNode*);

/* Lifecycle. start() is idempotent on a started group; stop() drains edges
 * before joining threads. destroy() returns ownership to the core. */
int   avp_start_group   (AvpGroup*, const char** err);   /* 0 ok, -1 err */
int   avp_stop_group    (AvpGroup*, const char** err);
/* restart() uses the same quiesce/fence/rebuild transaction as automatic
 * RestartGroup. status() returns caller-owned JSON; free it with
 * avp_string_free(). Both return 0 on success, -1 with caller-freed *err. */
int   avp_restart_group (AvpGroup*, const char** err);
int   avp_group_status  (AvpGroup*, char** out_status, const char** err);
/* Destroy edges before their endpoint nodes. Destroying a node also removes
 * any remaining incident edges and invalidates their handles. */
void  avp_destroy_node  (AvpCore*, AvpNode*);
void  avp_destroy_edge  (AvpCore*, AvpEdge*);
void  avp_destroy_group (AvpCore*, AvpGroup*);

/* Introspection (for embedders that didn't build the graph themselves). */
AvpNode*  avp_lookup_node (AvpCore*, const char* name);
AvpGroup* avp_lookup_group(AvpCore*, const char* name);

/* --------------------------------------------------------- core lifecycle
 * Usually the Rust binary owns main(); these exist for the static-lib/OBS embed. */
AvpCore* avp_core_create(void);
void     avp_core_destroy(AvpCore*);

/* Control protocol. Line parser lives in control/; TCP is not implemented. */
int  avp_core_exec_command(AvpCore*, const char* line, char** out_reply);
int  avp_core_serve_tcp(AvpCore*, uint16_t port);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* AVPLUMBER_CORE_H */
