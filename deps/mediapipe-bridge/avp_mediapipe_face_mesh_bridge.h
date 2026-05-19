#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AvpMpFaceMesh AvpMpFaceMesh;

typedef struct AvpMpFaceMeshConfig {
	int max_faces;
	int with_attention;
	int use_prev_landmarks;
	const char* resource_root;
} AvpMpFaceMeshConfig;

typedef struct AvpMpFaceMeshLandmark {
	float x;
	float y;
	float z;
} AvpMpFaceMeshLandmark;

typedef struct AvpMpFaceMeshFace {
	int landmark_count;
	AvpMpFaceMeshLandmark* landmarks;
} AvpMpFaceMeshFace;

typedef struct AvpMpFaceMeshResult {
	int face_count;
	AvpMpFaceMeshFace* faces;
} AvpMpFaceMeshResult;

int avp_mp_face_mesh_create(const AvpMpFaceMeshConfig* config,
                            AvpMpFaceMesh** handle,
                            char* error,
                            size_t error_size);

int avp_mp_face_mesh_process_egl_image(AvpMpFaceMesh* handle,
                                       void* egl_image,
                                       int width,
                                       int height,
                                       int64_t timestamp_us,
                                       AvpMpFaceMeshResult* result,
                                       char* error,
                                       size_t error_size);

void avp_mp_face_mesh_release_result(AvpMpFaceMeshResult* result);
void avp_mp_face_mesh_destroy(AvpMpFaceMesh* handle);

#ifdef __cplusplus
}
#endif
