#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AvpMpFaceDetection AvpMpFaceDetection;

typedef struct AvpMpFaceDetectionConfig {
	int min_detection_confidence_x1000; /* e.g. 500 = 0.5 */
	const char* resource_root;
} AvpMpFaceDetectionConfig;

typedef struct AvpMpFaceDetectionFace {
	float x1; /* normalized [0..1] */
	float y1;
	float x2;
	float y2;
	float score;
} AvpMpFaceDetectionFace;

typedef struct AvpMpFaceDetectionResult {
	int face_count;
	AvpMpFaceDetectionFace* faces;
} AvpMpFaceDetectionResult;

int avp_mp_face_detection_create(const AvpMpFaceDetectionConfig* config,
                                 AvpMpFaceDetection** handle,
                                 char* error,
                                 size_t error_size);

int avp_mp_face_detection_process_egl_image(AvpMpFaceDetection* handle,
                                            void* egl_image,
                                            int width,
                                            int height,
                                            int64_t timestamp_us,
                                            AvpMpFaceDetectionResult* result,
                                            char* error,
                                            size_t error_size);

void avp_mp_face_detection_release_result(AvpMpFaceDetectionResult* result);
void avp_mp_face_detection_destroy(AvpMpFaceDetection* handle);

#ifdef __cplusplus
}
#endif
