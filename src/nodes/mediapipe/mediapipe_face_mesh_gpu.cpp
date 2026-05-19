#include "../node_common.hpp"
#include "avp_mediapipe_face_mesh_bridge.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#ifndef AVP_MEDIAPIPE_DEFAULT_RESOURCE_ROOT
#define AVP_MEDIAPIPE_DEFAULT_RESOURCE_ROOT ""
#endif

struct LandmarkPoint {
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
};

struct GraphResult {
	bool observed = false;
	std::vector<std::vector<LandmarkPoint>> faces;
};

static double dist2d(const LandmarkPoint& a, const LandmarkPoint& b) {
	const double dx = double(a.x) - double(b.x);
	const double dy = double(a.y) - double(b.y);
	return std::sqrt(dx * dx + dy * dy);
}

static double deg(double radians) {
	return radians * 180.0 / 3.14159265358979323846;
}

static bool landmarkAt(const std::vector<LandmarkPoint>& list, int index, LandmarkPoint& out) {
	if (index < 0 || index >= int(list.size())) return false;
	out = list[size_t(index)];
	return true;
}

static Parameters landmarkJson(const std::vector<LandmarkPoint>& list) {
	Parameters out = Parameters::array();
	for (const LandmarkPoint& lm : list) {
		out.push_back({lm.x, lm.y, lm.z});
	}
	return out;
}

static GraphResult copyBridgeResult(const AvpMpFaceMeshResult& bridge_result) {
	GraphResult result;
	result.observed = bridge_result.face_count > 0;
	result.faces.reserve(std::max(0, bridge_result.face_count));
	for (int i = 0; i < bridge_result.face_count; ++i) {
		const AvpMpFaceMeshFace& src_face = bridge_result.faces[i];
		std::vector<LandmarkPoint> face;
		face.reserve(std::max(0, src_face.landmark_count));
		for (int j = 0; j < src_face.landmark_count; ++j) {
			const AvpMpFaceMeshLandmark& lm = src_face.landmarks[j];
			face.push_back({lm.x, lm.y, lm.z});
		}
		result.faces.push_back(std::move(face));
	}
	return result;
}

class MediaPipeFaceMeshGpu:
	public NodeSISO<EglImageFrame, MetadataFrame>,
	public ReportsFinishByFlag {
private:
	struct FaceState {
		bool speaking = false;
		int start_count = 0;
		int stop_count = 0;
		int missing_count = 0;
		std::optional<double> prev_open_ratio;
	};

	std::string metadata_key_ = "face_landmarks_v1";
	std::string resource_root_ = AVP_MEDIAPIPE_DEFAULT_RESOURCE_ROOT;
	int max_faces_ = 1;
	bool with_attention_ = true;
	bool use_prev_landmarks_ = true;
	int infer_every_n_ = 1;
	bool emit_dropped_metadata_ = true;
	int debug_log_every_n_ = 0;
	double speaking_start_open_ratio_ = 0.045;
	double speaking_stop_open_ratio_ = 0.030;
	int speaking_start_confirm_frames_ = 2;
	int speaking_stop_confirm_frames_ = 5;

	AvpMpFaceMesh* face_mesh_ = nullptr;
	int64_t last_mp_ts_ = -1;
	uint64_t frame_counter_ = 0;
	std::vector<FaceState> face_states_;

	int64_t nextMediaPipeTimestamp(const av::Timestamp& pts) {
		int64_t candidate = rescaleTS(pts, av::Rational(1, 1000000)).timestamp();
		if (candidate <= last_mp_ts_) {
			candidate = last_mp_ts_ + 1;
		}
		last_mp_ts_ = candidate;
		return candidate;
	}

	void ensureBridge() {
		if (face_mesh_) return;
		AvpMpFaceMeshConfig config = {};
		config.max_faces = max_faces_;
		config.with_attention = with_attention_ ? 1 : 0;
		config.use_prev_landmarks = use_prev_landmarks_ ? 1 : 0;
		config.resource_root = resource_root_.empty() ? nullptr : resource_root_.c_str();
		char error[4096] = {};
		if (avp_mp_face_mesh_create(&config, &face_mesh_, error, sizeof(error)) != 0) {
			throw Error(std::string("mediapipe_face_mesh_gpu: create bridge failed: ") + error);
		}
		face_states_.resize(std::max(1, max_faces_));
	}

	Parameters headPoseJson(const std::vector<LandmarkPoint>& face) {
		LandmarkPoint left_eye, right_eye, nose, chin;
		Parameters out = Parameters::object();
		if (!landmarkAt(face, 33, left_eye) ||
		    !landmarkAt(face, 263, right_eye) ||
		    !landmarkAt(face, 1, nose) ||
		    !landmarkAt(face, 152, chin)) {
			out["available"] = false;
			return out;
		}

		const double eye_dx = std::max(1e-6, double(right_eye.x) - double(left_eye.x));
		const double eye_dy = double(right_eye.y) - double(left_eye.y);
		const double eye_dist = std::max(1e-6, dist2d(left_eye, right_eye));
		const double mid_x = (double(left_eye.x) + double(right_eye.x)) * 0.5;
		const double mid_y = (double(left_eye.y) + double(right_eye.y)) * 0.5;

		const double yaw = deg(std::atan2(double(nose.x) - mid_x, eye_dist));
		const double pitch = deg(std::atan2(double(nose.y) - mid_y, std::max(1e-6, double(chin.y) - mid_y))) - 35.0;
		const double roll = deg(std::atan2(eye_dy, eye_dx));

		out["available"] = true;
		out["yaw_deg"] = yaw;
		out["pitch_deg"] = pitch;
		out["roll_deg"] = roll;
		return out;
	}

	Parameters mouthJson(const std::vector<LandmarkPoint>& face, size_t face_index, Parameters& events) {
		LandmarkPoint upper, lower, left, right;
		Parameters out = Parameters::object();
		if (!landmarkAt(face, 13, upper) ||
		    !landmarkAt(face, 14, lower) ||
		    !landmarkAt(face, 61, left) ||
		    !landmarkAt(face, 291, right)) {
			out["available"] = false;
			return out;
		}

		if (face_index >= face_states_.size()) {
			face_states_.resize(face_index + 1);
		}
		FaceState& state = face_states_[face_index];
		const double width = std::max(1e-6, dist2d(left, right));
		const double open_ratio = dist2d(upper, lower) / width;
		const double motion_score = state.prev_open_ratio ? std::abs(open_ratio - *state.prev_open_ratio) : 0.0;
		state.prev_open_ratio = open_ratio;
		state.missing_count = 0;

		if (!state.speaking) {
			state.stop_count = 0;
			if (open_ratio >= speaking_start_open_ratio_) {
				state.start_count += 1;
			} else {
				state.start_count = 0;
			}
			if (state.start_count >= speaking_start_confirm_frames_) {
				state.speaking = true;
				state.start_count = 0;
				events.push_back({
					{"type", "started_speaking"},
					{"face_id", int(face_index)},
					{"open_ratio", open_ratio},
					{"motion_score", motion_score},
				});
			}
		} else {
			state.start_count = 0;
			if (open_ratio <= speaking_stop_open_ratio_) {
				state.stop_count += 1;
			} else {
				state.stop_count = 0;
			}
			if (state.stop_count >= speaking_stop_confirm_frames_) {
				state.speaking = false;
				state.stop_count = 0;
				events.push_back({
					{"type", "stopped_speaking"},
					{"face_id", int(face_index)},
					{"open_ratio", open_ratio},
					{"motion_score", motion_score},
				});
			}
		}

		out["available"] = true;
		out["open_ratio"] = open_ratio;
		out["motion_score"] = motion_score;
		out["speaking"] = state.speaking;
		return out;
	}

	void markMissingFaces(size_t visible_faces, Parameters& events) {
		for (size_t i = visible_faces; i < face_states_.size(); ++i) {
			FaceState& state = face_states_[i];
			if (!state.speaking) continue;
			state.missing_count += 1;
			if (state.missing_count >= speaking_stop_confirm_frames_) {
				state.speaking = false;
				state.stop_count = 0;
				state.start_count = 0;
				state.prev_open_ratio.reset();
				events.push_back({
					{"type", "stopped_speaking"},
					{"face_id", int(i)},
					{"reason", "face_missing"},
				});
			}
		}
	}

	Parameters buildMetadata(const EglImageFrame& frame, const GraphResult& result, const std::string& status) {
		Parameters events = Parameters::array();
		Parameters faces = Parameters::array();
		if (result.observed) {
			for (size_t i = 0; i < result.faces.size(); ++i) {
				const auto& face = result.faces[i];
				Parameters face_md = Parameters::object();
				face_md["id"] = int(i);
				face_md["landmark_count"] = face.size();
				face_md["landmarks"] = landmarkJson(face);
				face_md["head_pose"] = headPoseJson(face);
				face_md["mouth"] = mouthJson(face, i, events);
				faces.push_back(std::move(face_md));
			}
			markMissingFaces(result.faces.size(), events);
		} else {
			markMissingFaces(0, events);
		}

		Parameters payload = Parameters::object();
		payload["version"] = 1;
		payload["source"] = "mediapipe_face_mesh_gpu";
		payload["status"] = status;
		payload["pts"] = frame.pts().timestamp();
		payload["timebase"] = {
			{"num", frame.timeBase().getNumerator()},
			{"den", frame.timeBase().getDenominator()},
		};
		payload["width"] = frame.width();
		payload["height"] = frame.height();
		payload["faces"] = std::move(faces);
		payload["events"] = std::move(events);

		Parameters out = Parameters::object();
		out[metadata_key_] = std::move(payload);
		return out;
	}

	void emitMetadata(const EglImageFrame& frame, const GraphResult& result, const std::string& status) {
		MetadataFrame out(frame.pts(), buildMetadata(frame, result, status));
		this->sink_->put(out);
		if (debug_log_every_n_ > 0 && (frame_counter_ % uint64_t(debug_log_every_n_)) == 0) {
			logstream << "mediapipe_face_mesh_gpu: frame=" << frame_counter_
			          << " status=" << status
			          << " faces=" << result.faces.size();
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
		if (face_mesh_) {
			avp_mp_face_mesh_destroy(face_mesh_);
			face_mesh_ = nullptr;
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
				emitMetadata(frame, {}, "skipped");
			}
			return;
		}

		try {
			ensureBridge();
			const int64_t mp_ts = nextMediaPipeTimestamp(frame.pts());
			AvpMpFaceMeshResult bridge_result = {};
			char error[4096] = {};
			const int rc = avp_mp_face_mesh_process_egl_image(
				face_mesh_,
				reinterpret_cast<void*>(frame.image()),
				frame.width(),
				frame.height(),
				mp_ts,
				&bridge_result,
				error,
				sizeof(error));
			if (rc != 0) {
				emitMetadata(frame, {}, std::string("error:") + error);
				return;
			}
			GraphResult result = copyBridgeResult(bridge_result);
			avp_mp_face_mesh_release_result(&bridge_result);
			emitMetadata(frame, result, result.observed ? "ok" : "no_face");
		} catch (const std::exception& e) {
			emitMetadata(frame, {}, std::string("error:") + e.what());
		}
	}

	static std::shared_ptr<MediaPipeFaceMeshGpu> create(NodeCreationInfo& nci) {
		EdgeManager& edges = nci.edges;
		const Parameters& params = nci.params;
		auto node = NodeSISO<EglImageFrame, MetadataFrame>::template createCommon<MediaPipeFaceMeshGpu>(edges, params);
		node->metadata_key_ = params.value("metadata_key", std::string("face_landmarks_v1"));
		node->resource_root_ = params.value("resource_root", std::string(AVP_MEDIAPIPE_DEFAULT_RESOURCE_ROOT));
		node->max_faces_ = std::max(1, params.value("max_faces", 1));
		node->with_attention_ = params.value("with_attention", true);
		node->use_prev_landmarks_ = params.value("use_prev_landmarks", true);
		node->infer_every_n_ = std::max(1, params.value("infer_every_n", 1));
		node->emit_dropped_metadata_ = params.value("emit_dropped_metadata", true);
		node->debug_log_every_n_ = params.value("debug_log_every_n", 0);
		node->speaking_start_open_ratio_ = params.value("speaking_start_open_ratio", 0.045);
		node->speaking_stop_open_ratio_ = params.value("speaking_stop_open_ratio", 0.030);
		node->speaking_start_confirm_frames_ = std::max(1, params.value("speaking_start_confirm_frames", 2));
		node->speaking_stop_confirm_frames_ = std::max(1, params.value("speaking_stop_confirm_frames", 5));
		return node;
	}
};

DECLNODE(mediapipe_face_mesh_gpu, MediaPipeFaceMeshGpu);
