/* Generated from src/graph/capability.rs by cbindgen. Do not edit. */

#ifndef AVPLUMBER_IDS_H
#define AVPLUMBER_IDS_H

#include <stdint.h>

// Live methods on a specific node (decoder discard, sentinel stats,
// `node.param` get/set, demux stream list). Query the node you already hold;
// there is no upstream walk.
//
// Not in this enum: stream format (`Spec` on the edge) or playback control
// (`avp_core_query_service`).
enum AvpInterfaceId
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  AVP_INTERFACE_ID_DECODER = 1,
  AVP_INTERFACE_ID_SENTINEL = 2,
  AVP_INTERFACE_ID_RETURNS_OBJECTS = 3,
  AVP_INTERFACE_ID_INPUTS_OBJECTS = 4,
  AVP_INTERFACE_ID_STREAMS_INPUT = 5,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum AvpInterfaceId AvpInterfaceId;
#else
typedef uint32_t AvpInterfaceId;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

// Core-owned services. An embedder looks them up with
// `avp_core_query_service`; each service's ID and vtable live in
// `avplumber_services_*.h`.
enum AvpServiceId
#if defined(__cplusplus) || __STDC_VERSION__ >= 202311L
  : uint32_t
#endif // defined(__cplusplus) || __STDC_VERSION__ >= 202311L
 {
  AVP_SERVICE_ID_CLOCK = 1,
  AVP_SERVICE_ID_TIMELINE = 2,
};
#ifndef __cplusplus
#if __STDC_VERSION__ >= 202311L
typedef enum AvpServiceId AvpServiceId;
#else
typedef uint32_t AvpServiceId;
#endif // __STDC_VERSION__ >= 202311L
#endif // __cplusplus

#endif  /* AVPLUMBER_IDS_H */
