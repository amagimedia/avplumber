#ifdef HAVE_MEDIAPIPE_AUTOFLIP

#include "../node_common.hpp"
#include "avp_mediapipe_autoflip_bridge.h"

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/frame.h>
}

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

class MediaPipeAutoflipCropMetadata:
	public NodeSISO<av::VideoFrame, av::VideoFrame>,
	public ReportsFinishByFlag {
private:
	struct SaliencySource {
		std::string metadata_key;
		int role = 0;        // 0=context, 1=preferred
		float weight = 1.0f;
		float min_conf = 0.0f;
		bool optional_input = false;
		// When non-empty, this source is only used when the named frame-metadata
		// key contains a JSON object whose `shot_type_field` value equals
		// `shot_type_value` (string comparison).
		std::string shot_type_key;
		std::string shot_type_field = "camera_shot";
		std::string shot_type_value;
		// When true, this source is only consulted if all non-fallback sources
		// yielded zero detections (ball/handler absent).
		bool fallback_when_empty = false;
	};

	std::vector<std::string> metadata_key_ins_ = {"face_detections_v1"};
	std::string metadata_key_out_ = "smoothed_crop_viewport_v1";
	std::string scene_cut_metadata_key_;
	std::string scene_cut_field_ = "scene_cut";
	bool reset_on_scene_cut_ = true;
	std::string mode_ = "kinematic";
	int lookahead_frames_ = 6;
	float min_motion_to_reframe_ = 0.0f;
	float max_velocity_ = 0.0f;
	int crop_w_ = 0;
	int crop_h_ = 0;
	int debug_log_every_n_ = 0;
	std::vector<SaliencySource> saliency_sources_;
	// model-content remap params (fallback when JSON payload lacks model dims)
	int model_content_width_    = 0;
	int model_content_height_   = 0;
	int model_content_offset_x_ = 0;
	int model_content_offset_y_ = 0;

	AvpMpAutoflip* autoflip_ = nullptr;
	uint64_t frame_counter_ = 0;
	bool bridge_initialized_ = false;

	// Computed at first frame
	int frame_width_ = 0;
	int frame_height_ = 0;
	int effective_crop_w_ = 0;
	int effective_crop_h_ = 0;

	static int roundEven(int v) {
		return v & ~1;
	}

	void initBridge(int frame_w, int frame_h) {
		if (bridge_initialized_) return;
		frame_width_ = frame_w;
		frame_height_ = frame_h;

		effective_crop_h_ = (crop_h_ > 0) ? crop_h_ : frame_h;
		effective_crop_w_ = (crop_w_ > 0) ? crop_w_
		                                   : roundEven(int(std::lround(double(frame_h) * 9.0 / 16.0)));
		effective_crop_w_ = std::min(effective_crop_w_, frame_w);
		effective_crop_h_ = std::min(effective_crop_h_, frame_h);

		AvpMpAutoflipConfig config = {};
		config.frame_width = frame_w;
		config.frame_height = frame_h;
		config.crop_width = effective_crop_w_;
		config.crop_height = effective_crop_h_;
		config.lookahead_frames = lookahead_frames_;
		config.min_motion_to_reframe = min_motion_to_reframe_;
		config.max_velocity = max_velocity_;

		char error[4096] = {};
		if (avp_mp_autoflip_create(&config, &autoflip_, error, sizeof(error)) != 0) {
			throw Error(std::string("mediapipe_autoflip_crop_metadata: create bridge failed: ") + error);
		}
		bridge_initialized_ = true;
	}

	// Read a string metadata entry from frame's AVDictionary.
	// Returns empty string if not found.
	static std::string readFrameMetadata(const av::VideoFrame& frm, const std::string& key) {
		const AVFrame* raw = frm.raw();
		if (!raw || !raw->metadata) return {};
		AVDictionaryEntry* entry = av_dict_get(raw->metadata, key.c_str(), nullptr, 0);
		if (!entry || !entry->value) return {};
		return std::string(entry->value);
	}

	// Write a string metadata entry to frame's AVDictionary.
	static void writeFrameMetadata(av::VideoFrame& frm, const std::string& key, const std::string& value) {
		av_dict_set(&frm.raw()->metadata, key.c_str(), value.c_str(), 0);
	}

	bool checkSceneCut(const av::VideoFrame& frm) const {
		if (scene_cut_metadata_key_.empty()) return false;
		const std::string raw = readFrameMetadata(frm, scene_cut_metadata_key_);
		if (raw.empty()) return false;
		try {
			Parameters md = Parameters::parse(raw);
			if (!md.contains(scene_cut_field_)) return false;
			const auto& val = md[scene_cut_field_];
			if (val.is_boolean()) return val.get<bool>();
			if (val.is_number_integer()) return val.get<int>() != 0;
			if (val.is_number()) return val.get<double>() != 0.0;
		} catch (...) {}
		return false;
	}

	// Parse detections from frame metadata for a given saliency source.
	// Returns false if the key is missing and optional_input is false.
	bool parseDetections(const av::VideoFrame& frm,
	                     const SaliencySource& src,
	                     std::vector<AvpMpAutoflipDetection>& out) const {
		// Gate on shot type if configured.
		if (!src.shot_type_key.empty()) {
			const std::string shot_raw = readFrameMetadata(frm, src.shot_type_key);
			bool gate_pass = false;
			if (!shot_raw.empty()) {
				try {
					Parameters shot_md = Parameters::parse(shot_raw);
					if (shot_md.contains(src.shot_type_field)) {
						gate_pass = (shot_md[src.shot_type_field].get<std::string>() == src.shot_type_value);
					}
				} catch (...) {}
			}
			if (!gate_pass) return true; // gated out — not an error
		}

		const std::string raw = readFrameMetadata(frm, src.metadata_key);
		if (raw.empty()) {
			return src.optional_input;
		}
		try {
			Parameters md = Parameters::parse(raw);

			// YOLO/tracker format: {"model_width":960,"model_height":544,"detections":[{"xyxy":[x1,y1,x2,y2],"conf":0.85,"label":"..."}]}
			if (md.contains("detections") && md["detections"].is_array()) {
				int mw = md.value("model_width",  model_content_width_  > 0 ? model_content_width_  : frame_width_);
				int mh = md.value("model_height", model_content_height_ > 0 ? model_content_height_ : frame_height_);
				int ox = md.value("model_content_offset_x", model_content_offset_x_);
				int oy = md.value("model_content_offset_y", model_content_offset_y_);
				for (const auto& det : md["detections"]) {
					float conf = det.contains("conf") ? det["conf"].get<float>()
					           : det.contains("score") ? det["score"].get<float>() : 1.0f;
					if (conf < src.min_conf) continue;
					if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) continue;
					const auto& xy = det["xyxy"];
					AvpMpAutoflipDetection d = {};
					d.weight = src.weight;
					d.role   = src.role;
					d.x1 = (xy[0].get<float>() - ox) * float(frame_width_)  / float(mw);
					d.y1 = (xy[1].get<float>() - oy) * float(frame_height_) / float(mh);
					d.x2 = (xy[2].get<float>() - ox) * float(frame_width_)  / float(mw);
					d.y2 = (xy[3].get<float>() - oy) * float(frame_height_) / float(mh);
					out.push_back(d);
				}
				return true;
			}

			// face_detections_v1 / bbox formats
			const Parameters* faces_array = nullptr;
			if (md.is_array()) {
				faces_array = &md;
			} else if (md.contains("faces") && md["faces"].is_array()) {
				faces_array = &md["faces"];
			} else {
				return src.optional_input;
			}

			for (const auto& det : *faces_array) {
				float score = 1.0f;
				if (det.contains("conf"))  score = det["conf"].get<float>();
				else if (det.contains("score")) score = det["score"].get<float>();
				if (score < src.min_conf) continue;

				AvpMpAutoflipDetection d = {};
				d.weight = src.weight;
				d.role   = src.role;

				if (det.contains("bbox") && det["bbox"].is_array() && det["bbox"].size() >= 4) {
					const auto& bbox = det["bbox"];
					d.x1 = bbox[0].get<float>();
					d.y1 = bbox[1].get<float>();
					d.x2 = bbox[2].get<float>();
					d.y2 = bbox[3].get<float>();
				} else if (det.contains("bbox_norm") && det["bbox_norm"].is_array() && det["bbox_norm"].size() >= 4) {
					const auto& bbox = det["bbox_norm"];
					d.x1 = bbox[0].get<float>() * float(frame_width_);
					d.y1 = bbox[1].get<float>() * float(frame_height_);
					d.x2 = bbox[2].get<float>() * float(frame_width_);
					d.y2 = bbox[3].get<float>() * float(frame_height_);
				} else {
					continue;
				}
				out.push_back(d);
			}
		} catch (...) {
			return src.optional_input;
		}
		return true;
	}

	// Parse detections from frame metadata using the legacy metadata_key_ins_ list.
	// Each key is expected to contain a JSON blob with "faces" array at top level
	// (matching face_detections_v1 format).
	bool parseDetectionsFromKeyIns(const av::VideoFrame& frm,
	                               std::vector<AvpMpAutoflipDetection>& out) const {
		for (const auto& key : metadata_key_ins_) {
			const std::string raw = readFrameMetadata(frm, key);
			if (raw.empty()) continue;
			try {
				Parameters md = Parameters::parse(raw);
				// face_detections_v1 top-level payload may contain "faces"
				const Parameters* faces_array = nullptr;
				if (md.is_array()) {
					faces_array = &md;
				} else if (md.contains("faces") && md["faces"].is_array()) {
					faces_array = &md["faces"];
				} else {
					continue;
				}
				for (const auto& det : *faces_array) {
					AvpMpAutoflipDetection d = {};
					d.weight = 1.0f;
					d.role = 1; // preferred by default for face detections
					float score = 1.0f;
					if (det.contains("score")) score = det["score"].get<float>();
					(void)score; // no min_conf filter for legacy path

					if (det.contains("bbox") && det["bbox"].is_array() && det["bbox"].size() >= 4) {
						const auto& bbox = det["bbox"];
						d.x1 = bbox[0].get<float>();
						d.y1 = bbox[1].get<float>();
						d.x2 = bbox[2].get<float>();
						d.y2 = bbox[3].get<float>();
					} else if (det.contains("bbox_norm") && det["bbox_norm"].is_array() && det["bbox_norm"].size() >= 4) {
						const auto& bbox = det["bbox_norm"];
						d.x1 = bbox[0].get<float>() * float(frame_width_);
						d.y1 = bbox[1].get<float>() * float(frame_height_);
						d.x2 = bbox[2].get<float>() * float(frame_width_);
						d.y2 = bbox[3].get<float>() * float(frame_height_);
					} else {
						continue;
					}
					out.push_back(d);
				}
			} catch (...) {
				continue;
			}
		}
		return true;
	}

	int64_t framePtsUs(const av::VideoFrame& frm) const {
		if (!frm.pts().isValid()) return frame_counter_ * 33333LL;
		return rescaleTS(frm.pts(), av::Rational(1, 1000000)).timestamp();
	}

public:
	using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

	void stop() override {
		NodeSingleInput<av::VideoFrame>::stop();
		if (autoflip_) {
			avp_mp_autoflip_destroy(autoflip_);
			autoflip_ = nullptr;
			bridge_initialized_ = false;
		}
	}

	void process() override {
		av::VideoFrame frm = this->source_->get();

		if (isEofMarker(frm)) {
			this->sink_->put(frm);
			this->finished_ = true;
			return;
		}
		if (!frm) return;

		++frame_counter_;

		const int fw = frm.width();
		const int fh = frm.height();

		// Lazy init bridge on first frame
		if (!bridge_initialized_) {
			initBridge(fw, fh);
		}

		// Check scene cut
		const int scene_cut = (reset_on_scene_cut_ && checkSceneCut(frm)) ? 1 : 0;

		// Gather detections
		std::vector<AvpMpAutoflipDetection> detections;
		if (!saliency_sources_.empty()) {
			// Pass 1: primary sources
			for (const auto& src : saliency_sources_) {
				if (!src.fallback_when_empty)
					parseDetections(frm, src, detections);
			}
			// Pass 2: fallback sources — only when primary pass yielded nothing
			if (detections.empty()) {
				for (const auto& src : saliency_sources_) {
					if (src.fallback_when_empty)
						parseDetections(frm, src, detections);
				}
			}
		} else {
			parseDetectionsFromKeyIns(frm, detections);
		}

		// Compute timestamp
		const int64_t ts_us = framePtsUs(frm);

		// Call bridge
		AvpMpAutoflipResult result = {};
		char error[4096] = {};
		const int rc = avp_mp_autoflip_process(
			autoflip_,
			ts_us,
			detections.empty() ? nullptr : detections.data(),
			int(detections.size()),
			scene_cut,
			&result,
			error,
			sizeof(error));

		std::string status_str = "ok";
		int crop_x1 = (fw - effective_crop_w_) / 2;
		if (rc != 0) {
			status_str = std::string("error:") + error;
		} else {
			crop_x1 = result.crop_x1;
			if (result.status == 1) status_str = "no_subjects";
			else if (result.status == 2) status_str = "fallback_center";
			else if (result.status == 3) status_str = std::string("error:") + result.status_detail;
		}

		// Build output metadata JSON
		Parameters payload = Parameters::object();
		payload["viewport_bbox"] = {crop_x1, 0, crop_x1 + effective_crop_w_, fh};
		payload["viewport_dst_width"] = effective_crop_w_;
		payload["viewport_dst_height"] = effective_crop_h_;
		payload["full_frame_width"] = fw;
		payload["full_frame_height"] = fh;
		payload["status"] = status_str;

		const std::string json_str = payload.dump();
		writeFrameMetadata(frm, metadata_key_out_, json_str);

		if (debug_log_every_n_ > 0 && (frame_counter_ % uint64_t(debug_log_every_n_)) == 0) {
			logstream << "mediapipe_autoflip_crop_metadata: frame=" << frame_counter_
			          << " crop_x1=" << crop_x1
			          << " crop_w=" << effective_crop_w_
			          << " status=" << status_str
			          << " detections=" << detections.size();
			for (size_t i = 0; i < detections.size(); ++i) {
				const auto& d = detections[i];
				float cx = (d.x1 + d.x2) * 0.5f;
				float cy = (d.y1 + d.y2) * 0.5f;
				logstream << " det[" << i << "]=(cx=" << cx << ",cy=" << cy
				          << ",role=" << d.role << ",w=" << d.weight << ")";
			}
		}

		this->sink_->put(frm);
	}

	static std::shared_ptr<MediaPipeAutoflipCropMetadata> create(NodeCreationInfo& nci) {
		EdgeManager& edges = nci.edges;
		const Parameters& params = nci.params;
		auto node = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<MediaPipeAutoflipCropMetadata>(edges, params);

		// metadata_key_ins: string or array of strings
		if (params.contains("metadata_key_ins")) {
			const auto& ki = params["metadata_key_ins"];
			node->metadata_key_ins_.clear();
			if (ki.is_string()) {
				node->metadata_key_ins_.push_back(ki.get<std::string>());
			} else if (ki.is_array()) {
				for (const auto& item : ki) {
					node->metadata_key_ins_.push_back(item.get<std::string>());
				}
			}
		}

		node->metadata_key_out_ = params.value("metadata_key_out", std::string("smoothed_crop_viewport_v1"));
		node->scene_cut_metadata_key_ = params.value("scene_cut_metadata_key", std::string(""));
		node->scene_cut_field_ = params.value("scene_cut_field", std::string("scene_cut"));
		node->reset_on_scene_cut_ = params.value("reset_on_scene_cut", true);
		node->mode_ = params.value("mode", std::string("kinematic"));
		node->lookahead_frames_ = params.value("lookahead_frames", 6);
		node->min_motion_to_reframe_ = params.value("min_motion_to_reframe", 0.0f);
		node->max_velocity_ = params.value("max_velocity", 0.0f);
		node->crop_w_ = params.value("crop_w", 0);
		node->crop_h_ = params.value("crop_h", 0);
		node->debug_log_every_n_ = params.value("debug_log_every_n", 0);
		node->model_content_width_    = params.value("model_content_width",    0);
		node->model_content_height_   = params.value("model_content_height",   0);
		node->model_content_offset_x_ = params.value("model_content_offset_x", 0);
		node->model_content_offset_y_ = params.value("model_content_offset_y", 0);

		// saliency: array of { metadata_key, role, weight, min_conf, optional_input }
		if (params.contains("saliency") && params["saliency"].is_array()) {
			for (const auto& s : params["saliency"]) {
				SaliencySource src;
				src.metadata_key = s.value("metadata_key", std::string(""));
				if (src.metadata_key.empty()) continue;
				const std::string role_str = s.value("role", std::string("context"));
				src.role = (role_str == "preferred") ? 1 : 0;
				src.weight = s.value("weight", 1.0f);
				src.min_conf = s.value("min_conf", 0.0f);
				src.optional_input = s.value("optional_input", false);
				src.shot_type_key      = s.value("shot_type_key",      std::string(""));
				src.shot_type_field    = s.value("shot_type_field",    std::string("camera_shot"));
				src.shot_type_value    = s.value("shot_type_value",    std::string(""));
				src.fallback_when_empty = s.value("fallback_when_empty", false);
				node->saliency_sources_.push_back(std::move(src));
			}
		}

		return node;
	}
};

DECLNODE(mediapipe_autoflip_crop_metadata, MediaPipeAutoflipCropMetadata)

#endif // HAVE_MEDIAPIPE_AUTOFLIP
