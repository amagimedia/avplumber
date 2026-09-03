/* avplumber_services_clock.h - SyncGroup master clock.
 *
 * Shared playback-to-wall mapping (rate, offset, pause) for streams that must
 * stay A/V-synchronized. Rate/offset/pause are O(1) writes here; buffers keep
 * source-time PTS and are mapped through the clock only when an output stage
 * releases them. A seek resets the clock once for the whole group.
 *
 * The core owns the contract; implementations are pluggable (WallClock /
 * SourceTimeClock / SyntheticClock) in services/clock.rs. An embedder calls
 * avp_core_query_service(core, AVP_SERVICE_ID_CLOCK) to get the vtable below.
 * No file-scope avp_clock_* functions — keeps avplumber_core.h free of this
 * domain.
 */
#ifndef AVPLUMBER_SERVICES_CLOCK_H
#define AVPLUMBER_SERVICES_CLOCK_H

#include "avplumber_core.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AvpSyncGroup AvpSyncGroup;

/* Returned by avp_core_query_service(core, AVP_SERVICE_ID_CLOCK). The core
 * registers one impl per build (WallClock by default). An embedder may
 * supply its own.
 *
 * create()     - get/create a SyncGroup by name (per clock domain).
 * set_rate()   - <0 = reverse playback (sign of the clock rate).
 * set_paused()  - freeze/thaw the output; back-pressure propagates upstream.
 * reset()       - seek: one shared reset for the whole group.
 * map_to_wall() - output-stage read: source PTS -> wall/presentation time.
 *                Returns AVP_NOPTS while paused-and-not-yet-due. */
typedef struct {
    AvpSyncGroup* (*create)    (AvpCore*, const char* name);
    void          (*set_rate)   (AvpSyncGroup*, double rate);
    void          (*set_paused) (AvpSyncGroup*, int paused);
    void          (*reset)      (AvpSyncGroup*, int64_t new_pos, AvpRational tb);
    int64_t       (*map_to_wall)(AvpSyncGroup*, int64_t src_pts, AvpRational tb);
} AvpSyncGroupVtable;

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* AVPLUMBER_SERVICES_CLOCK_H */
