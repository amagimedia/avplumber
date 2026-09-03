/* avplumber_services_timeline.h - named PTS-keyed JSON store.
 *
 * Control writes scheduled values at a source-time PTS; nodes read "the value
 * in effect at this frame's PTS". Not on the data plane. Values are opaque
 * JSON. Crosses the boundary in both directions (Rust control writes, C++
 * nodes read), hence a C surface.
 *
 * Reached via avp_core_query_service(core, AVP_SERVICE_ID_TIMELINE). No
 * file-scope avp_timeline_* functions.
 */
#ifndef AVPLUMBER_SERVICES_TIMELINE_H
#define AVPLUMBER_SERVICES_TIMELINE_H

#include "avplumber_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AvpTimeline AvpTimeline;

/*
 * set()        - control-plane write: scheduled value keyed by source-time PTS (ms).
 * clear_key()  - drop a single (channel, key).
 * gc()         - drop all entries with at_pts_ms < before_pts_ms.
 * get()        - node read: latest value with at_pts_ms <= frame_pts (compared
 *                via the frame timebase). Writes the JSON value into `out` (cap
 *                bytes) and returns its length, 0 if no entry applies, or the
 *                needed length (>cap) if truncated. */
typedef struct {
    AvpTimeline* (*create)    (AvpCore*, const char* name);
    void         (*set)       (AvpTimeline*, const char* channel, const char* key,
                               int64_t at_pts_ms, const char* value_json);
    void         (*clear_key)  (AvpTimeline*, const char* channel, const char* key);
    void         (*gc)         (AvpTimeline*, int64_t before_pts_ms);
    int          (*get)        (AvpTimeline*, const char* channel, const char* key,
                               int64_t frame_pts, AvpRational tb, char* out, size_t cap);
} AvpTimelineVtable;

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* AVPLUMBER_SERVICES_TIMELINE_H */
