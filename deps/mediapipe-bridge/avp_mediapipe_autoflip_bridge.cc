#include "avp_mediapipe_autoflip_bridge.h"

#include "mediapipe/examples/desktop/autoflip/quality/frame_crop_region_computer.h"
#include "mediapipe/examples/desktop/autoflip/quality/kinematic_path_solver.h"
#include "mediapipe/examples/desktop/autoflip/quality/kinematic_path_solver.pb.h"
#include "mediapipe/examples/desktop/autoflip/autoflip_messages.pb.h"

#include "absl/status/status.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>

namespace {

static void set_error(char* error, size_t error_size, const std::string& message) {
	if (!error || error_size == 0) return;
	std::snprintf(error, error_size, "%s", message.c_str());
}

static int clamp_int(int value, int lo, int hi) {
	if (value < lo) return lo;
	if (value > hi) return hi;
	return value;
}

class AutoflipBridge {
public:
	explicit AutoflipBridge(const AvpMpAutoflipConfig& cfg)
		: frame_width_(cfg.frame_width),
		  frame_height_(cfg.frame_height),
		  crop_width_(cfg.crop_width),
		  crop_height_(cfg.crop_height) {
		mediapipe::autoflip::KinematicOptions kinematic_options;
		if (cfg.min_motion_to_reframe > 0.0f) {
			kinematic_options.set_min_motion_to_reframe(cfg.min_motion_to_reframe);
		}
		if (cfg.max_velocity > 0.0f) {
			kinematic_options.set_max_velocity(cfg.max_velocity);
		}
		solver_ = std::make_unique<mediapipe::autoflip::KinematicPathSolver>(
			kinematic_options, 0, frame_width_, 1.0f);

		// FrameCropRegionComputer: Google's original algorithm that finds the
		// minimum crop window covering all salient regions.
		mediapipe::autoflip::KeyFrameCropOptions crop_options;
		crop_options.set_target_width(crop_width_);
		crop_options.set_target_height(crop_height_);
		crop_options.set_score_aggregation_type(
			mediapipe::autoflip::KeyFrameCropOptions::SUM_ALL);
		frame_crop_computer_ = std::make_unique<mediapipe::autoflip::FrameCropRegionComputer>(
			crop_options);
	}

	absl::Status process(int64_t timestamp_us,
	                     const AvpMpAutoflipDetection* detections,
	                     int detection_count,
	                     int scene_cut,
	                     AvpMpAutoflipResult* result) {
		if (!result) return absl::InvalidArgumentError("null result");
		result->crop_x1 = (frame_width_ - crop_width_) / 2;
		result->status = 0;
		result->status_detail[0] = '\0';

		if (scene_cut) {
			solver_->ClearHistory();
		}

		bool have_subjects = (detection_count > 0);
		float observed_center_x = float(frame_width_) * 0.5f;

		if (have_subjects) {
			// Build KeyFrameInfo with SalientRegion detections.
			// preferred (role==1) get full weight; context (role==0) get 1/10 weight
			// so they only influence the crop when no preferred subject is present.
			mediapipe::autoflip::KeyFrameInfo frame_info;
			frame_info.set_timestamp_ms(timestamp_us / 1000);
			auto* det_set = frame_info.mutable_detections();

			for (int i = 0; i < detection_count; ++i) {
				const AvpMpAutoflipDetection& det = detections[i];
				auto* region = det_set->add_detections();
				auto* loc = region->mutable_location();
				loc->set_x(int(det.x1));
				loc->set_y(int(det.y1));
				loc->set_width(int(det.x2 - det.x1));
				loc->set_height(int(det.y2 - det.y1));
				float w = (det.role == 1) ? det.weight : det.weight * 0.1f;
				region->set_score(w);
				region->set_is_required(false);
			}

			// Ask FrameCropRegionComputer for the tightest covering crop.
			mediapipe::autoflip::KeyFrameCropResult crop_result;
			auto status = frame_crop_computer_->ComputeFrameCropRegion(
				frame_info, &crop_result);
			if (!status.ok()) {
				std::snprintf(result->status_detail, sizeof(result->status_detail),
				              "%s", std::string(status.message()).c_str());
				result->status = 3;
				return absl::OkStatus();
			}

			// Use the center of the computed crop region as the observation.
			if (crop_result.has_region()) {
				const auto& region = crop_result.region();
				observed_center_x = float(region.x()) + float(region.width()) * 0.5f;
			}
		}

		// Feed the computed center through KinematicPathSolver for temporal smoothing.
		absl::Status solver_status;
		if (have_subjects) {
			solver_status = solver_->AddObservation(observed_center_x, timestamp_us);
		} else {
			solver_status = solver_->UpdatePrediction(timestamp_us);
			result->status = 1; // no_subjects
			std::snprintf(result->status_detail, sizeof(result->status_detail), "no_subjects");
		}
		if (!solver_status.ok()) {
			std::snprintf(result->status_detail, sizeof(result->status_detail),
			              "%s", std::string(solver_status.message()).c_str());
			result->status = 3;
			return absl::OkStatus();
		}

		float state_x = float(frame_width_) * 0.5f;
		auto get_status = solver_->GetState(&state_x);
		if (!get_status.ok()) {
			std::snprintf(result->status_detail, sizeof(result->status_detail),
			              "%s", std::string(get_status.message()).c_str());
			result->status = 3;
			return absl::OkStatus();
		}

		const int max_x1 = std::max(0, frame_width_ - crop_width_);
		result->crop_x1 = clamp_int(
			int(std::lround(double(state_x) - double(crop_width_) * 0.5)),
			0, max_x1);

		if (result->status == 0) {
			std::snprintf(result->status_detail, sizeof(result->status_detail), "ok");
		}
		return absl::OkStatus();
	}

private:
	int frame_width_ = 0;
	int frame_height_ = 0;
	int crop_width_ = 0;
	int crop_height_ = 0;
	std::unique_ptr<mediapipe::autoflip::KinematicPathSolver> solver_;
	std::unique_ptr<mediapipe::autoflip::FrameCropRegionComputer> frame_crop_computer_;
};

}  // namespace

struct AvpMpAutoflip {
	std::unique_ptr<AutoflipBridge> impl;
};

extern "C" int avp_mp_autoflip_create(const AvpMpAutoflipConfig* config,
                                      AvpMpAutoflip** handle,
                                      char* error,
                                      size_t error_size) {
	if (!config || !handle) {
		set_error(error, error_size, "invalid create arguments");
		return -1;
	}
	if (config->frame_width <= 0 || config->frame_height <= 0) {
		set_error(error, error_size, "frame_width and frame_height must be positive");
		return -1;
	}
	if (config->crop_width <= 0 || config->crop_height <= 0) {
		set_error(error, error_size, "crop_width and crop_height must be positive");
		return -1;
	}
	try {
		std::unique_ptr<AvpMpAutoflip> out(new AvpMpAutoflip());
		out->impl = std::make_unique<AutoflipBridge>(*config);
		*handle = out.release();
		return 0;
	} catch (const std::exception& e) {
		set_error(error, error_size, e.what());
		return -1;
	}
}

extern "C" int avp_mp_autoflip_process(AvpMpAutoflip* handle,
                                       int64_t timestamp_us,
                                       const AvpMpAutoflipDetection* detections,
                                       int detection_count,
                                       int scene_cut,
                                       AvpMpAutoflipResult* result,
                                       char* error,
                                       size_t error_size) {
	if (!handle || !handle->impl) {
		set_error(error, error_size, "invalid autoflip handle");
		return -1;
	}
	try {
		absl::Status status = handle->impl->process(
			timestamp_us, detections, detection_count, scene_cut, result);
		if (!status.ok()) {
			set_error(error, error_size, std::string(status.message()).c_str());
			return -1;
		}
		return 0;
	} catch (const std::exception& e) {
		set_error(error, error_size, e.what());
		return -1;
	}
}

extern "C" void avp_mp_autoflip_destroy(AvpMpAutoflip* handle) {
	delete handle;
}
