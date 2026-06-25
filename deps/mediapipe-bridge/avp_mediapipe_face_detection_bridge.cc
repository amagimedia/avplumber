#define EGL_EGLEXT_PROTOTYPES 1
#define GL_GLEXT_PROTOTYPES 1

#include "avp_mediapipe_face_detection_bridge.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <vector>
#include <sstream>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "google/protobuf/text_format.h"
#include "mediapipe/framework/calculator.pb.h"
#include "mediapipe/framework/calculator_graph.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/timestamp.h"
#include "mediapipe/framework/deps/file_helpers.h"
#include "mediapipe/gpu/gl_context.h"
#include "mediapipe/gpu/gl_texture_buffer.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_buffer_format.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"
#include "mediapipe/util/resource_util_custom.h"

namespace {

using GlEGLImageTargetTexture2DOESFn = void (*)(GLenum, void*);

constexpr const char* kGraph = R"pbtxt(
input_stream: "input_video"
output_stream: "face_detections"

node {
  calculator: "FaceDetectionShortRangeGpu"
  input_stream: "IMAGE:input_video"
  output_stream: "DETECTIONS:face_detections"
}
)pbtxt";

static void set_error(char* error, size_t error_size, const std::string& message) {
	if (!error || error_size == 0) return;
	std::snprintf(error, error_size, "%s", message.c_str());
}

static std::string status_text(const absl::Status& status) {
	return std::string(status.message());
}

static std::string join_path(const std::string& root, const std::string& path) {
	if (root.empty() || path.empty() || path[0] == '/') return path;
	if (root.back() == '/') return root + path;
	return root + "/" + path;
}

// Headless EGL initialization: pick the first EGL device (NVIDIA GPU) so that
// eglGetDisplay(EGL_DEFAULT_DISPLAY) inside MediaPipe returns a usable display.
static void ensure_headless_egl() {
	auto eglQueryDevicesEXT_fn = reinterpret_cast<
		EGLBoolean (*)(EGLint, EGLDeviceEXT*, EGLint*)>(
			eglGetProcAddress("eglQueryDevicesEXT"));
	auto eglGetPlatformDisplayEXT_fn = reinterpret_cast<
		EGLDisplay (*)(EGLenum, void*, const EGLint*)>(
			eglGetProcAddress("eglGetPlatformDisplayEXT"));
	if (!eglQueryDevicesEXT_fn || !eglGetPlatformDisplayEXT_fn) return;

	EGLint num_devices = 0;
	eglQueryDevicesEXT_fn(0, nullptr, &num_devices);
	if (num_devices <= 0) return;

	std::vector<EGLDeviceEXT> devices(num_devices);
	eglQueryDevicesEXT_fn(num_devices, devices.data(), &num_devices);

	for (int i = 0; i < num_devices; ++i) {
		EGLDisplay dpy = eglGetPlatformDisplayEXT_fn(
			EGL_PLATFORM_DEVICE_EXT, devices[i], nullptr);
		if (dpy == EGL_NO_DISPLAY) continue;
		EGLint major = 0, minor = 0;
		if (eglInitialize(dpy, &major, &minor) == EGL_TRUE) {
			return;
		}
	}
}

static void configure_resource_root(const char* root_ptr) {
	if (!root_ptr || root_ptr[0] == '\0') return;
	const std::string root(root_ptr);
	mediapipe::SetCustomGlobalResourceProvider(
		[root](const std::string& path, std::string* output) -> absl::Status {
			absl::Status status = mediapipe::file::GetContents(join_path(root, path), output, true);
			if (status.ok()) return status;
			return mediapipe::file::GetContents(path, output, true);
		});
}

struct GraphResult {
	bool observed = false;
	std::vector<mediapipe::Detection> detections;
};

class FaceDetectionBridge {
public:
	explicit FaceDetectionBridge(const AvpMpFaceDetectionConfig& config)
		: min_confidence_(config.min_detection_confidence_x1000 / 1000.0f) {}

	absl::Status start(const AvpMpFaceDetectionConfig& config) {
		ensure_headless_egl();
		configure_resource_root(config.resource_root);

		gl_egl_image_target_texture_ =
			reinterpret_cast<GlEGLImageTargetTexture2DOESFn>(eglGetProcAddress("glEGLImageTargetTexture2DOES"));
		if (!gl_egl_image_target_texture_) {
			return absl::InternalError("glEGLImageTargetTexture2DOES unavailable");
		}

		auto gpu_resources_or = mediapipe::GpuResources::Create();
		if (!gpu_resources_or.ok()) return gpu_resources_or.status();
		gpu_resources_ = std::move(gpu_resources_or).value();
		gl_context_ = gpu_resources_->gl_context();
		if (!gl_context_) {
			return absl::InternalError("MediaPipe GPU resources have no GL context");
		}

		mediapipe::CalculatorGraphConfig graph_config;
		if (!google::protobuf::TextFormat::ParseFromString(kGraph, &graph_config)) {
			return absl::InternalError("failed to parse embedded face detection graph");
		}

		graph_ = std::make_unique<mediapipe::CalculatorGraph>();
		absl::Status status = graph_->SetGpuResources(gpu_resources_);
		if (!status.ok()) return status;
		status = graph_->Initialize(graph_config);
		if (!status.ok()) return status;
		status = graph_->ObserveOutputStream(
			"face_detections",
			[this](const mediapipe::Packet& packet) -> absl::Status {
				GraphResult result;
				result.observed = true;
				result.detections = packet.Get<std::vector<mediapipe::Detection>>();
				std::lock_guard<std::mutex> lock(results_mtx_);
				results_[packet.Timestamp().Value()] = std::move(result);
				return absl::OkStatus();
			});
		if (!status.ok()) return status;

		status = graph_->StartRun({});
		if (!status.ok()) return status;
		graph_->SetInputStreamMaxQueueSize("input_video", 1).IgnoreError();
		return absl::OkStatus();
	}

	absl::Status process(void* egl_image, int width, int height, int64_t timestamp_us,
	                     AvpMpFaceDetectionResult* result) {
		if (!egl_image) return absl::InvalidArgumentError("null EGLImage");
		if (width <= 0 || height <= 0) return absl::InvalidArgumentError("invalid image geometry");
		if (!result) return absl::InvalidArgumentError("null result");
		result->face_count = 0;
		result->faces = nullptr;

		mediapipe::GpuBuffer buffer;
		absl::Status status = wrap_frame(egl_image, width, height, &buffer);
		if (!status.ok()) return status;

		auto packet = mediapipe::MakePacket<mediapipe::GpuBuffer>(buffer)
		                  .At(mediapipe::Timestamp::FromMicroseconds(timestamp_us));
		status = graph_->AddPacketToInputStream("input_video", std::move(packet));
		if (!status.ok()) return status;

		status = graph_->WaitUntilIdle();
		if (!status.ok()) return status;

		GraphResult graph_result = take_result(timestamp_us);
		if (!graph_result.observed) return absl::OkStatus();
		if (graph_result.detections.empty()) return absl::OkStatus();

		// Filter by confidence and convert to result struct
		std::vector<AvpMpFaceDetectionFace> faces;
		faces.reserve(graph_result.detections.size());
		for (const auto& det : graph_result.detections) {
			const float score = det.score_size() > 0 ? det.score(0) : 0.0f;
			if (score < min_confidence_) continue;

			if (!det.has_location_data()) continue;
			const auto& loc = det.location_data();
			if (loc.format() != mediapipe::LocationData::RELATIVE_BOUNDING_BOX) continue;
			const auto& bbox = loc.relative_bounding_box();

			AvpMpFaceDetectionFace face;
			face.x1 = bbox.xmin();
			face.y1 = bbox.ymin();
			face.x2 = bbox.xmin() + bbox.width();
			face.y2 = bbox.ymin() + bbox.height();
			face.score = score;
			faces.push_back(face);
		}

		if (faces.empty()) return absl::OkStatus();

		result->face_count = int(faces.size());
		result->faces = new AvpMpFaceDetectionFace[faces.size()]();
		for (size_t i = 0; i < faces.size(); ++i) {
			result->faces[i] = faces[i];
		}
		return absl::OkStatus();
	}

	void stop() {
		if (graph_) {
			graph_->CloseAllInputStreams().IgnoreError();
			graph_->WaitUntilDone().IgnoreError();
			graph_.reset();
		}
		gl_context_.reset();
		gpu_resources_.reset();
	}

	~FaceDetectionBridge() {
		stop();
	}

private:
	float min_confidence_ = 0.5f;
	std::unique_ptr<mediapipe::CalculatorGraph> graph_;
	std::shared_ptr<mediapipe::GpuResources> gpu_resources_;
	std::shared_ptr<mediapipe::GlContext> gl_context_;
	GlEGLImageTargetTexture2DOESFn gl_egl_image_target_texture_ = nullptr;
	std::mutex results_mtx_;
	std::map<int64_t, GraphResult> results_;

	absl::Status wrap_frame(void* egl_image, int width, int height, mediapipe::GpuBuffer* out) {
		std::shared_ptr<mediapipe::GlTextureBuffer> texture_buffer;
		absl::Status status = gl_context_->Run([&]() -> absl::Status {
			GLuint tex = 0;
			glGenTextures(1, &tex);
			if (!tex) return absl::InternalError("glGenTextures failed");
			glBindTexture(GL_TEXTURE_2D, tex);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

			gl_egl_image_target_texture_(GL_TEXTURE_2D, egl_image);
			GLenum err = glGetError();
			if (err != GL_NO_ERROR) {
				glBindTexture(GL_TEXTURE_2D, 0);
				glDeleteTextures(1, &tex);
				std::ostringstream oss;
				oss << "glEGLImageTargetTexture2DOES failed: 0x" << std::hex << err;
				return absl::InternalError(oss.str());
			}

			auto cleanup = [ctx = gl_context_, tex](std::shared_ptr<mediapipe::GlSyncPoint> sync_token) {
				if (sync_token) sync_token->Wait();
				ctx->Run([tex]() -> absl::Status {
					GLuint t = tex;
					glDeleteTextures(1, &t);
					return absl::OkStatus();
				}).IgnoreError();
			};
			auto wrapped = mediapipe::GlTextureBuffer::Wrap(
				GL_TEXTURE_2D, tex, width, height, mediapipe::GpuBufferFormat::kBGRA32, gl_context_, std::move(cleanup));
			if (!wrapped) {
				glBindTexture(GL_TEXTURE_2D, 0);
				glDeleteTextures(1, &tex);
				return absl::InternalError("GlTextureBuffer::Wrap returned null");
			}
			texture_buffer = std::shared_ptr<mediapipe::GlTextureBuffer>(wrapped.release());
			glBindTexture(GL_TEXTURE_2D, 0);
			return absl::OkStatus();
		});
		if (!status.ok()) return status;
		*out = mediapipe::GpuBuffer(texture_buffer);
		return absl::OkStatus();
	}

	GraphResult take_result(int64_t timestamp_us) {
		std::lock_guard<std::mutex> lock(results_mtx_);
		auto it = results_.find(timestamp_us);
		if (it == results_.end()) return {};
		GraphResult result = std::move(it->second);
		results_.erase(it);
		return result;
	}
};

}  // namespace

struct AvpMpFaceDetection {
	std::unique_ptr<FaceDetectionBridge> impl;
};

extern "C" int avp_mp_face_detection_create(const AvpMpFaceDetectionConfig* config,
                                            AvpMpFaceDetection** handle,
                                            char* error,
                                            size_t error_size) {
	if (!config || !handle) {
		set_error(error, error_size, "invalid create arguments");
		return -1;
	}
	try {
		std::unique_ptr<AvpMpFaceDetection> out(new AvpMpFaceDetection());
		out->impl = std::make_unique<FaceDetectionBridge>(*config);
		absl::Status status = out->impl->start(*config);
		if (!status.ok()) {
			set_error(error, error_size, status_text(status));
			return -1;
		}
		*handle = out.release();
		return 0;
	} catch (const std::exception& e) {
		set_error(error, error_size, e.what());
		return -1;
	}
}

extern "C" int avp_mp_face_detection_process_egl_image(AvpMpFaceDetection* handle,
                                                       void* egl_image,
                                                       int width,
                                                       int height,
                                                       int64_t timestamp_us,
                                                       AvpMpFaceDetectionResult* result,
                                                       char* error,
                                                       size_t error_size) {
	if (!handle || !handle->impl) {
		set_error(error, error_size, "invalid face detection handle");
		return -1;
	}
	try {
		absl::Status status = handle->impl->process(egl_image, width, height, timestamp_us, result);
		if (!status.ok()) {
			set_error(error, error_size, status_text(status));
			return -1;
		}
		return 0;
	} catch (const std::exception& e) {
		set_error(error, error_size, e.what());
		return -1;
	}
}

extern "C" void avp_mp_face_detection_release_result(AvpMpFaceDetectionResult* result) {
	if (!result) return;
	delete[] result->faces;
	result->face_count = 0;
	result->faces = nullptr;
}

extern "C" void avp_mp_face_detection_destroy(AvpMpFaceDetection* handle) {
	delete handle;
}
