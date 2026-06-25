#include "../node_common.hpp"
#include "avp_mediapipe_face_detection_bridge.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#ifndef AVP_MEDIAPIPE_DEFAULT_RESOURCE_ROOT
#define AVP_MEDIAPIPE_DEFAULT_RESOURCE_ROOT ""
#endif

class MediaPipeFaceDetectionGpu:
	public NodeSISO<EglImageFrame, MetadataFrame>,
	public ReportsFinishByFlag {
private:
	std::string metadata_key_ = "face_detections_v1";
	std::string resource_root_ = AVP_MEDIAPIPE_DEFAULT_RESOURCE_ROOT;
	double min_detection_confidence_ = 0.5;
	int infer_every_n_ = 1;
	bool emit_dropped_metadata_ = true;
	int debug_log_every_n_ = 0;

	AvpMpFaceDetection* face_detection_ = nullptr;
	int64_t last_mp_ts_ = -1;
	uint64_t frame_counter_ = 0;

	int64_t nextMediaPipeTimestamp(const av::Timestamp& pts) {
		int64_t candidate = rescaleTS(pts, av::Rational(1, 1000000)).timestamp();
		if (candidate <= last_mp_ts_) {
			candidate = last_mp_ts_ + 1;
		}
		last_mp_ts_ = candidate;
		return candidate;
	}

	void ensureBridge() {
		if (face_detection_) return;
		AvpMpFaceDetectionConfig config = {};
		config.min_detection_confidence_x1000 = int(min_detection_confidence_ * 1000.0 + 0.5);
		config.resource_root = resource_root_.empty() ? nullptr : resource_root_.c_str();
		char error[4096] = {};
		if (avp_mp_face_detection_create(&config, &face_detection_, error, sizeof(error)) != 0) {
			throw Error(std::string("mediapipe_face_detection_gpu: create bridge failed: ") + error);
		}
	}

	Parameters buildMetadata(const EglImageFrame& frame,
	                         const AvpMpFaceDetectionResult& bridge_result,
	                         const std::string& status) {
		Parameters faces = Parameters::array();
		for (int i = 0; i < bridge_result.face_count; ++i) {
			const AvpMpFaceDetectionFace& f = bridge_result.faces[i];
			const float fw = float(frame.width());
			const float fh = float(frame.height());
			Parameters face_md = Parameters::object();
			face_md["id"] = i;
			face_md["bbox"] = {
				int(std::lround(double(f.x1) * fw)),
				int(std::lround(double(f.y1) * fh)),
				int(std::lround(double(f.x2) * fw)),
				int(std::lround(double(f.y2) * fh))
			};
			face_md["bbox_norm"] = {f.x1, f.y1, f.x2, f.y2};
			face_md["score"] = f.score;
			faces.push_back(std::move(face_md));
		}

		Parameters payload = Parameters::object();
		payload["version"] = 1;
		payload["source"] = "mediapipe_face_detection_gpu";
		payload["status"] = status;
		payload["width"] = frame.width();
		payload["height"] = frame.height();
		payload["faces"] = std::move(faces);

		Parameters out = Parameters::object();
		out[metadata_key_] = std::move(payload);
		return out;
	}

	void emitMetadata(const EglImageFrame& frame,
	                  const AvpMpFaceDetectionResult& bridge_result,
	                  const std::string& status) {
		MetadataFrame out(frame.pts(), buildMetadata(frame, bridge_result, status));
		this->sink_->put(out);
		if (debug_log_every_n_ > 0 && (frame_counter_ % uint64_t(debug_log_every_n_)) == 0) {
			logstream << "mediapipe_face_detection_gpu: frame=" << frame_counter_
			          << " status=" << status
			          << " faces=" << bridge_result.face_count;
		}
	}

public:
	using NodeSISO<EglImageFrame, MetadataFrame>::NodeSISO;

	void start() override {
		NodeSingleOutput<MetadataFrame>::start();
		ensureBridge();
	}

	void stop() override {
		NodeSingleInput<EglImageFrame>::stop();
		if (face_detection_) {
			avp_mp_face_detection_destroy(face_detection_);
			face_detection_ = nullptr;
		}
	}

	void process() override {
		EglImageFrame frame = this->source_->get();
		if (!frame.isComplete() || !frame.pts().isValid()) {
			return;
		}

		frame_counter_ += 1;
		if (infer_every_n_ > 1 && ((frame_counter_ - 1) % uint64_t(infer_every_n_)) != 0) {
			if (emit_dropped_metadata_) {
				AvpMpFaceDetectionResult empty = {};
				emitMetadata(frame, empty, "skipped");
			}
			return;
		}

		try {
			ensureBridge();
			const int64_t mp_ts = nextMediaPipeTimestamp(frame.pts());
			AvpMpFaceDetectionResult bridge_result = {};
			char error[4096] = {};
			const int rc = avp_mp_face_detection_process_egl_image(
				face_detection_,
				reinterpret_cast<void*>(frame.image()),
				frame.width(),
				frame.height(),
				mp_ts,
				&bridge_result,
				error,
				sizeof(error));
			if (rc != 0) {
				AvpMpFaceDetectionResult empty = {};
				emitMetadata(frame, empty, std::string("error:") + error);
				return;
			}
			const std::string status = bridge_result.face_count > 0 ? "ok" : "no_face";
			emitMetadata(frame, bridge_result, status);
			avp_mp_face_detection_release_result(&bridge_result);
		} catch (const std::exception& e) {
			AvpMpFaceDetectionResult empty = {};
			emitMetadata(frame, empty, std::string("error:") + e.what());
		}
	}

	static std::shared_ptr<MediaPipeFaceDetectionGpu> create(NodeCreationInfo& nci) {
		EdgeManager& edges = nci.edges;
		const Parameters& params = nci.params;
		auto node = NodeSISO<EglImageFrame, MetadataFrame>::template createCommon<MediaPipeFaceDetectionGpu>(edges, params);
		node->metadata_key_ = params.value("metadata_key", std::string("face_detections_v1"));
		node->resource_root_ = params.value("resource_root", std::string(AVP_MEDIAPIPE_DEFAULT_RESOURCE_ROOT));
		node->min_detection_confidence_ = params.value("min_detection_confidence", 0.5);
		node->infer_every_n_ = std::max(1, params.value("infer_every_n", 1));
		node->emit_dropped_metadata_ = params.value("emit_dropped_metadata", true);
		node->debug_log_every_n_ = params.value("debug_log_every_n", 0);
		return node;
	}
};

DECLNODE(mediapipe_face_detection_gpu, MediaPipeFaceDetectionGpu)
