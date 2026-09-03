/* avplumber_interfaces.h - C vtables for node capabilities.
 *
 * Numeric IDs live in avplumber_ids.h (generated from Rust). This header is
 * the C-only method tables those IDs return.
 *
 * Not capabilities:
 *   - stream format (AvpSpec on the edge);
 *   - playback control (avp_core_query_service);
 *   - JACK sink (registers on the JACK service);
 *   - encoder/muxer config (codec/timebase travel as AvpSpec on the edge).
 */
#ifndef AVPLUMBER_INTERFACES_H
#define AVPLUMBER_INTERFACES_H

#include "avplumber_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* First arg is always the AvpNode whose query_interface returned this vtable.
 * Strings are UTF-8, NUL-terminated, owned by the node unless stated. */

typedef struct {
    const char* (*codec_name)(AvpNode*);
    const char* (*media_type_string)(AvpNode*);
    void        (*discard_until)(AvpNode*, int64_t pts, AvpRational tb);
} AvpIDecoder;

/* Returns an owned JSON string; caller frees with avp_string_free. */
typedef struct {
    char* (*get_object)(AvpNode*, const char* name);
} AvpIReturnsObjects;

typedef struct {
    void (*set_object)(AvpNode*, const char* name, const char* json);
} AvpIInputsObjects;

typedef struct {
    /* TODO: signal_present, card_present, last_pts, etc. */
} AvpISentinel;

typedef struct {
    /* TODO: stream_count, stream_info(index, ...) */
} AvpIStreamsInput;

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* AVPLUMBER_INTERFACES_H */
