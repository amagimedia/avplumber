#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AvpMpAutoflip AvpMpAutoflip;

typedef struct AvpMpAutoflipConfig {
	int frame_width;
	int frame_height;
	int crop_width;          /* viewport_dst_width */
	int crop_height;         /* viewport_dst_height (must equal frame_height for horizontal-only) */
	int lookahead_frames;    /* default 6; pass 0 to use default */
	/* KinematicOptions fields; use 0.0 for defaults */
	float min_motion_to_reframe; /* 0 = use MediaPipe default */
	float max_velocity;          /* 0 = use MediaPipe default */
} AvpMpAutoflipConfig;

typedef struct AvpMpAutoflipDetection {
	float x1, y1, x2, y2; /* absolute pixel coords in frame */
	float weight;          /* saliency weight 0..100 */
	int role;              /* 0=context, 1=preferred */
} AvpMpAutoflipDetection;

typedef struct AvpMpAutoflipResult {
	int crop_x1;           /* horizontal crop start (y1=0, y2=frame_height) */
	int status;            /* 0=ok, 1=no_subjects, 2=fallback_center, 3=error */
	char status_detail[64];
} AvpMpAutoflipResult;

int avp_mp_autoflip_create(const AvpMpAutoflipConfig* config,
                           AvpMpAutoflip** handle,
                           char* error,
                           size_t error_size);

int avp_mp_autoflip_process(AvpMpAutoflip* handle,
                            int64_t timestamp_us,
                            const AvpMpAutoflipDetection* detections,
                            int detection_count,
                            int scene_cut,
                            AvpMpAutoflipResult* result,
                            char* error,
                            size_t error_size);

void avp_mp_autoflip_destroy(AvpMpAutoflip* handle);

#ifdef __cplusplus
}
#endif
