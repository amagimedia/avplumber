#include "../../node_common.hpp"
#include "../../../kalman1d.hpp"
#include "../common/yolo_side_data.hpp"
#include "ball_tracker_utils.hpp"

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/frame.h>
}

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

using ball_tracker_detail::DetectionBox;
using ball_tracker_detail::TrailPoint;
using ball_tracker_detail::GateDecision;
using ball_tracker_detail::ParsedFrameMetadata;
using ball_tracker_detail::CandidateSelection;
using ball_tracker_detail::TrackingDecision;
using ball_tracker_detail::CourtBoundsCheckConfig;
using ball_tracker_detail::CourtBoundsCheckResult;
using ball_tracker_detail::boxWidth;
using ball_tracker_detail::boxHeight;
using ball_tracker_detail::centerX;
using ball_tracker_detail::centerY;
using ball_tracker_detail::finiteBox;
using ball_tracker_detail::iou;
using ball_tracker_detail::hypot2;
using ball_tracker_detail::sizePenalty;
using ball_tracker_detail::roundnessPenalty;
using ball_tracker_detail::addTrailPoint;
using ball_tracker_detail::checkCourtBoundsHorizontalOverlap;

class BallTracker : public NodeSISO<av::VideoFrame, av::VideoFrame>, public IInputReset, public ReportsFinishByFlag {
private:
    // Target filtering
    std::string metadata_key_ = "yolo_detections";
    std::string target_label_ = "basketball";
    std::vector<std::string> target_labels_;
    int target_class_ = -1;
    double min_conf_ = 0.04;

    // Shot-aware mode handling
    std::string camera_shot_metadata_key_;  // empty = disabled

    // Best ball selection
    double target_ball_size_rel_ = 0.036;
    double w_conf_ = 1.0;
    double w_dist_ = 0.015;
    double w_size_ = 0.5;
    double w_round_ = 0.4;

    // Gating
    double max_jump_rel_ = 0.12;
    double gate_rel_ = 0.08;
    double gate_min_px_ = 16.0;
    bool gate_use_pred_ = true;
    double min_iou_ = 0.20;
    double speed_mult_ = 3.0;

    // CPU court-bounds veto for reacquires
    bool court_bounds_veto_enabled_ = true;
    std::string court_bounds_seg_metadata_key_ = "yolo_seg";
    int court_bounds_side_data_slot_ = 0;
    std::string court_bounds_camera_shot_metadata_key_ = "camera_shot_info";
    std::unordered_set<int> court_bounds_class_indices_ = {0, 1};
    bool court_bounds_require_wide_shot_ = true;
    double court_bounds_min_court_coverage_ = 0.1;
    double court_bounds_mask_threshold_ = 0.5;
    int court_bounds_sample_rows_ = 3;
    double court_bounds_min_horizontal_overlap_ratio_ = 0.35;
    double court_bounds_max_horizontal_outside_px_ = 2.0;

    // Detection override
    double override_conf_ = 0.28;
    int override_after_ = 2;
    int reacquire_frames_ = 6;
    double override_max_jump_rel_ = 0.12;
    int override_far_confirm_frames_ = 9;
    double override_far_gate_mult_ = 3.0;
    double override_far_match_mult_ = 1.5;
    int override_far_moving_confirm_frames_ = 4;
    double override_far_motion_rel_ = 0.035;
    bool coast_edge_jump_veto_enabled_ = false;
    double coast_edge_zone_rel_ = 0.14;
    double coast_edge_jump_rel_ = 0.18;
    int coast_edge_confirm_frames_ = 9;

    // Coasting
    bool coast_ = true;
    int coast_max_ = 6;

    // Kalman
    double kalman_q_pos_ = 1.0;
    double kalman_q_vel_ = 0.1;
    double kalman_r_meas_ = 4.0;

    // Trail
    size_t trail_max_ = 200;

    // Debug
    int debug_log_every_n_ = 0;

    // Dump file
    std::string dump_file_path_;
    std::ofstream dump_file_;

    // --- Internal state ---
    Kalman1D kx_, ky_;
    bool kalman_initialized_ = false;
    std::deque<TrailPoint> trail_;
    DetectionBox last_emitted_box_;
    bool have_last_emitted_ = false;
    DetectionBox last_detected_template_; // for carrying forward model_index etc on coast
    bool have_detected_template_ = false;
    DetectionBox last_real_detected_box_;
    bool have_last_real_detected_ = false;
    uint64_t last_real_detected_frame_ = 0;
    DetectionBox pending_override_box_;
    DetectionBox pending_override_start_box_;
    bool have_pending_override_ = false;
    int pending_override_hits_ = 0;
    uint64_t pending_override_last_frame_ = 0;
    double recent_speed_ = -1.0;
    int coast_streak_ = 0;
    int det_reject_streak_ = 0;
    uint64_t frame_counter_ = 0;
    int track_id_ = 1;
    double last_accepted_conf_ = 0.0;
    std::string last_shot_type_;
    bool have_last_shot_type_ = false;
    bool closeup_tracking_armed_ = true;

    // --- Statistics ---
    uint64_t stat_detected_ = 0;
    uint64_t stat_override_ = 0;
    uint64_t stat_coasted_ = 0;
    uint64_t stat_no_ball_ = 0;
    uint64_t stat_dropped_by_gate_ = 0;
    double stat_ball_size_sum_ = 0.0;
    uint64_t stat_ball_size_count_ = 0;
    std::array<uint64_t, 10> conf_histogram_{};
    double stat_vel_min_ = 1e9;
    double stat_vel_max_ = 0.0;
    double stat_vel_sum_ = 0.0;
    uint64_t stat_vel_count_ = 0;

    double dump_last_x_ = 0.0, dump_last_y_ = 0.0;
    bool dump_have_last_ = false;

    void dumpFrame(const std::string& source, double x, double y, double conf) {
        if (!dump_file_.is_open()) return;
        dump_file_ << frame_counter_ << ",";
        if (!source.empty()) {
            double dx = 0.0, dy = 0.0, dist = 0.0;
            if (dump_have_last_) {
                dx = x - dump_last_x_;
                dy = y - dump_last_y_;
                dist = std::sqrt(dx * dx + dy * dy);
            }
            dump_file_ << x << "," << y << "," << conf << "," << source
                       << "," << dx << "," << dy << "," << dist;
            dump_last_x_ = x;
            dump_last_y_ = y;
            dump_have_last_ = true;
        } else {
            dump_file_ << ",,,,,,";
        }
        dump_file_ << "\n";
    }

    void initKalman(double cx, double cy) {
        kx_ = Kalman1D(kalman_q_pos_, kalman_q_vel_, kalman_r_meas_);
        ky_ = Kalman1D(kalman_q_pos_, kalman_q_vel_, kalman_r_meas_);
        kx_.init(cx);
        ky_.init(cy);
        kalman_initialized_ = true;
    }

    void resetTrackingState() {
        kalman_initialized_ = false;
        trail_.clear();
        have_last_emitted_ = false;
        have_detected_template_ = false;
        have_last_real_detected_ = false;
        last_real_detected_frame_ = 0;
        clearPendingOverride();
        recent_speed_ = -1.0;
        coast_streak_ = 0;
        det_reject_streak_ = 0;
    }

    void resetState() {
        resetTrackingState();
        last_shot_type_.clear();
        have_last_shot_type_ = false;
        closeup_tracking_armed_ = true;
    }

    void clearPendingOverride() {
        pending_override_box_ = DetectionBox{};
        pending_override_start_box_ = DetectionBox{};
        have_pending_override_ = false;
        pending_override_hits_ = 0;
        pending_override_last_frame_ = 0;
    }

    bool shouldDebugLog() const {
        return debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0;
    }

    void predictKalman() {
        if (!kalman_initialized_) return;
        kx_.predict(1.0);
        ky_.predict(1.0);
    }

    void forwardNoBall(av::VideoFrame& frm, bool predict_first) {
        if (predict_first) predictKalman();
        stat_no_ball_++;
        dumpFrame("", 0.0, 0.0, 0.0);
        this->sink_->put(frm);
    }

    bool matchesTarget(const DetectionBox& det) const {
        bool label_match = false;
        bool class_match = false;
        if (!target_label_.empty() && det.has_label && det.label == target_label_) {
            label_match = true;
        }
        for (const auto& lbl : target_labels_) {
            if (det.has_label && det.label == lbl) {
                label_match = true;
                break;
            }
        }
        if (target_class_ >= 0 && det.cls == target_class_) {
            class_match = true;
        }
        const bool have_targets = !target_label_.empty() || !target_labels_.empty() || target_class_ >= 0;
        if (!have_targets) return true;
        return label_match || class_match;
    }

    DetectionBox parseOneDetection(const Parameters& item) const {
        DetectionBox det;
        det.cls = item.value("cls", -1);
        det.conf = item.value("conf", 0.0);
        if (item.contains("label") && item["label"].is_string()) {
            det.label = item["label"].get<std::string>();
            det.has_label = true;
        }
        if (item.contains("model_index")) {
            det.model_index = item["model_index"].get<int>();
        }
        if (item.contains("engine_name") && item["engine_name"].is_string()) {
            det.engine_name = item["engine_name"].get<std::string>();
            det.has_engine_name = true;
        }
        if (item.contains("xyxy") && item["xyxy"].is_array() && item["xyxy"].size() >= 4) {
            det.x1 = item["xyxy"][0].get<double>();
            det.y1 = item["xyxy"][1].get<double>();
            det.x2 = item["xyxy"][2].get<double>();
            det.y2 = item["xyxy"][3].get<double>();
        }
        return det;
    }

    bool stripBallDetectionsFromMetadata(av::VideoFrame& frm) const {
        AVFrame* raw = frm.raw();
        if (!raw || !raw->metadata) return false;

        AVDictionaryEntry* ball_entry = av_dict_get(raw->metadata, metadata_key_.c_str(), nullptr, 0);
        if (!ball_entry || !ball_entry->value) return false;

        try {
            Parameters ball_md = Parameters::parse(ball_entry->value);
            Parameters other_dets = Parameters::array();
            if (ball_md.contains("detections") && ball_md["detections"].is_array()) {
                for (const auto& item : ball_md["detections"]) {
                    DetectionBox det = parseOneDetection(item);
                    if (!matchesTarget(det)) {
                        other_dets.push_back(item);
                    }
                }
            }
            ball_md["detections"] = other_dets;
            ball_md["trail"] = Parameters::array();
            av_dict_set(&raw->metadata, metadata_key_.c_str(), ball_md.dump().c_str(), 0);
            return true;
        } catch (...) {
            return false;
        }
    }

    std::string getShotType(const AVFrame* raw) const {
        if (camera_shot_metadata_key_.empty()) return std::string();
        if (!raw || !raw->metadata) return std::string();

        AVDictionaryEntry* shot_entry = av_dict_get(raw->metadata, camera_shot_metadata_key_.c_str(), nullptr, 0);
        if (!shot_entry || !shot_entry->value) return std::string();

        try {
            Parameters shot_md = Parameters::parse(shot_entry->value);
            return shot_md.value("camera_shot_type", std::string());
        } catch (...) {
            return std::string();
        }
    }

    void updateShotMode(const std::string& shot_type) {
        const bool is_closeup = (shot_type == "closeup");
        const bool was_closeup = have_last_shot_type_ && last_shot_type_ == "closeup";

        if (!have_last_shot_type_) {
            closeup_tracking_armed_ = !is_closeup;
        } else if (was_closeup != is_closeup) {
            resetTrackingState();
            closeup_tracking_armed_ = !is_closeup;
            if (shouldDebugLog()) {
                logstream << "ball_tracker: frame=" << frame_counter_
                          << " shot_transition from=" << (was_closeup ? "closeup" : "wide")
                          << " to=" << (is_closeup ? "closeup" : "wide")
                          << " armed=" << closeup_tracking_armed_;
            }
        }

        last_shot_type_ = shot_type;
        have_last_shot_type_ = true;
    }

    bool handleCloseupSuppression(av::VideoFrame& frm, const std::string& shot_type,
                                  const ParsedFrameMetadata& parsed) {
        if (shot_type != "closeup") return false;

        stripBallDetectionsFromMetadata(frm);
        stat_no_ball_++;
        writeOutputMetadata(frm, parsed, false, DetectionBox{}, "", 0.0);
        dumpFrame("", 0.0, 0.0, 0.0);
        if (shouldDebugLog()) {
            logstream << "ball_tracker: frame=" << frame_counter_
                      << " closeup_suppressed"
                      << " raw_candidates=" << parsed.ball_dets.size();
        }
        this->sink_->put(frm);
        return true;
    }

    bool parseFrameMetadata(av::VideoFrame& frm, ParsedFrameMetadata& parsed) {
        const AVFrame* raw = frm.raw();
        if (!raw || !raw->metadata) {
            forwardNoBall(frm, true);
            return false;
        }

        AVDictionaryEntry* entry = av_dict_get(raw->metadata, metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) {
            forwardNoBall(frm, true);
            return false;
        }

        try {
            parsed.md = Parameters::parse(entry->value);
        } catch (...) {
            forwardNoBall(frm, true);
            return false;
        }

        parsed.model_w = parsed.md.value("model_width", (double)frm.width());
        parsed.model_h = parsed.md.value("model_height", (double)frm.height());
        parsed.other_det_items = Parameters::array();
        parsed.ball_dets.clear();

        if (parsed.md.contains("detections") && parsed.md["detections"].is_array()) {
            for (const auto& item : parsed.md["detections"]) {
                if (!item.is_object()) continue;
                DetectionBox det = parseOneDetection(item);
                if (!finiteBox(det)) {
                    parsed.other_det_items.push_back(item);
                    continue;
                }
                if (matchesTarget(det) && det.conf >= min_conf_) {
                    parsed.ball_dets.push_back(det);
                } else {
                    parsed.other_det_items.push_back(item);
                }
            }
        }
        if (court_bounds_veto_enabled_) {
            populateShotInfo(raw, parsed);
            populateCourtMaskData(raw, parsed);
        }
        return true;
    }

    void populateShotInfo(const AVFrame* raw, ParsedFrameMetadata& parsed) const {
        if (!raw || !raw->metadata || court_bounds_camera_shot_metadata_key_.empty()) return;

        AVDictionaryEntry* shot_entry = av_dict_get(raw->metadata, court_bounds_camera_shot_metadata_key_.c_str(), nullptr, 0);
        if (!shot_entry || !shot_entry->value) return;

        try {
            Parameters shot_md = Parameters::parse(shot_entry->value);
            parsed.have_camera_shot_info = true;
            parsed.shot_type = shot_md.value("camera_shot_type", std::string());
            parsed.court_coverage = shot_md.value("court_coverage", 0.0);
        } catch (...) {
            parsed.have_camera_shot_info = false;
            parsed.shot_type.clear();
            parsed.court_coverage = 0.0;
        }
    }

    void populateCourtMaskData(const AVFrame* raw, ParsedFrameMetadata& parsed) const {
        if (!raw || !raw->metadata) return;

        AVDictionaryEntry* seg_entry = av_dict_get(raw->metadata, court_bounds_seg_metadata_key_.c_str(), nullptr, 0);
        if (!seg_entry || !seg_entry->value) return;

        try {
            Parameters seg_md = Parameters::parse(seg_entry->value);
            if (seg_md.contains("detections") && seg_md["detections"].is_array()) {
                int idx = 0;
                for (const auto& det : seg_md["detections"]) {
                    if (det.contains("cls")) {
                        int cls = det["cls"].get<int>();
                        if (court_bounds_class_indices_.count(cls)) {
                            parsed.court_mask_indices.push_back(idx);
                        }
                    }
                    ++idx;
                }
            }
        } catch (...) {
            parsed.court_mask_indices.clear();
            return;
        }

        if (parsed.court_mask_indices.empty()) return;

        const AVFrameSideData* sd = av_frame_get_side_data(raw, yoloSegCpuSideDataType(court_bounds_side_data_slot_));
        if (!sd || sd->size < 16) return;

        const uint32_t* header = reinterpret_cast<const uint32_t*>(sd->data);
        const uint32_t num_masks = header[0];
        const uint32_t mask_w = header[1];
        const uint32_t mask_h = header[2];
        if (num_masks == 0 || mask_w == 0 || mask_h == 0) return;

        const size_t pixels_per_mask = (size_t)mask_w * (size_t)mask_h;
        const size_t required_size = 16 + num_masks * pixels_per_mask * sizeof(float);
        if ((size_t)sd->size < required_size) return;

        parsed.have_court_mask_data = true;
        parsed.court_num_masks = num_masks;
        parsed.court_mask_w = mask_w;
        parsed.court_mask_h = mask_h;
        parsed.court_mask_data = reinterpret_cast<const float*>(sd->data + 16);
    }

    CourtBoundsCheckConfig buildCourtBoundsCheckConfig() const {
        CourtBoundsCheckConfig cfg;
        cfg.enabled = court_bounds_veto_enabled_;
        cfg.require_wide_shot = court_bounds_require_wide_shot_;
        cfg.min_court_coverage = court_bounds_min_court_coverage_;
        cfg.mask_threshold = court_bounds_mask_threshold_;
        cfg.sample_rows = court_bounds_sample_rows_;
        cfg.min_horizontal_overlap_ratio = court_bounds_min_horizontal_overlap_ratio_;
        cfg.max_horizontal_outside_px = court_bounds_max_horizontal_outside_px_;
        return cfg;
    }

    CourtBoundsCheckResult evaluateCourtBounds(const DetectionBox& cand,
                                               const ParsedFrameMetadata& parsed) const {
        const auto result = checkCourtBoundsHorizontalOverlap(cand, parsed, buildCourtBoundsCheckConfig());
        if (shouldDebugLog()) {
            logstream << "ball_tracker: frame=" << frame_counter_
                      << " court_bounds_check cx=" << centerX(cand)
                      << " cy=" << centerY(cand)
                      << " rows=" << result.sample_rows_used
                      << " camera_shot=" << (parsed.have_camera_shot_info ? parsed.shot_type : std::string("missing"))
                      << " court_cov=" << parsed.court_coverage
                      << " best_overlap=" << result.best_overlap_ratio
                      << " worst_overlap=" << result.worst_overlap_ratio
                      << " left_out=" << result.max_left_out_px
                      << " right_out=" << result.max_right_out_px
                      << " usable=" << result.usable
                      << " veto=" << result.veto
                      << " reason=" << result.reason;
        }
        return result;
    }

    bool coastEdgeJumpVetoApplies(const DetectionBox& cand,
                                  double model_w, double model_h,
                                  bool stale, double dist_prev,
                                  bool& edge_zone_hit) const {
        edge_zone_hit = false;
        if (!coast_edge_jump_veto_enabled_ || model_w <= 0.0 || model_h <= 0.0) return false;
        if (!(stale || coast_streak_ > 0) || trail_.empty()) return false;

        const double min_dim = std::min(model_w, model_h);
        if (dist_prev < coast_edge_jump_rel_ * min_dim) return false;

        const double cx = centerX(cand);
        const double edge_zone = coast_edge_zone_rel_ * model_w;
        edge_zone_hit = (cx <= edge_zone) || (cx >= (model_w - edge_zone));
        return edge_zone_hit;
    }

    double computeBallScore(const DetectionBox& cand, double model_w, double model_h) const {
        const double cx = centerX(cand);
        const double cy = centerY(cand);
        const double bw = boxWidth(cand);
        const double bh = boxHeight(cand);
        const double norm = 0.5 * (model_w + model_h);
        double dist_pen = 0.0;
        if (kalman_initialized_ && norm > 0.0) {
            dist_pen = hypot2(cx - kx_.pos(), cy - ky_.pos()) / norm;
        }
        return w_conf_ * cand.conf
             - w_dist_ * dist_pen
             - w_size_ * sizePenalty(bw, bh, model_w, model_h, target_ball_size_rel_)
             - w_round_ * roundnessPenalty(bw, bh);
    }

    GateDecision evaluateGate(const DetectionBox& cand, double model_w, double model_h) const {
        GateDecision gd;
        const double cx = centerX(cand);
        const double cy = centerY(cand);
        const double min_dim = std::min(model_w, model_h);
        gd.hard_cap = max_jump_rel_ * min_dim;
        gd.base_gate = std::max(gate_min_px_, gate_rel_ * min_dim);

        // Distance to the current tracked path
        if (!trail_.empty()) {
            const auto& last = trail_.back();
            gd.dist_prev = hypot2(cx - last.x, cy - last.y);
        }

        // Distance to the last real gated detection. This is the jump we care about
        // most when trying to suppress one-frame detector glitches.
        if (have_last_real_detected_ && last_real_detected_frame_ > 0) {
            gd.dist_real = hypot2(cx - centerX(last_real_detected_box_),
                                  cy - centerY(last_real_detected_box_));
            gd.real_gap_frames = frame_counter_ - last_real_detected_frame_;
        }

        // Stage 1: Hard cap against actual detector jumps. Scale by the number of
        // frames since the last real detection so reacquisition after coasting is allowed.
        if (have_last_real_detected_ && gd.real_gap_frames > 0
            && gd.dist_real > gd.hard_cap * std::max<uint64_t>(1, gd.real_gap_frames)) {
            gd.hard_cap_ok = false;
        }

        // Stage 2: Distance gate
        gd.distance_ok = trail_.empty() || (gd.dist_prev <= gd.base_gate);

        // Stage 3: Prediction gate (OR)
        if (gate_use_pred_ && kalman_initialized_) {
            gd.d_pred = hypot2(cx - kx_.pos(), cy - ky_.pos());
            gd.prediction_ok = gd.d_pred <= gd.base_gate * 1.25;
        }

        // Stage 4: IoU continuity
        if (have_last_emitted_) {
            gd.iou_ok = iou(cand, last_emitted_box_) >= min_iou_;
        }

        // Stage 5: Speed constraint
        if (recent_speed_ > 0.0 && !trail_.empty()) {
            gd.speed_gate = std::max(gd.base_gate, speed_mult_ * (recent_speed_ + 1e-6));
            gd.speed_ok = gd.dist_prev <= gd.speed_gate;
        }

        gd.continuity_ok = gd.distance_ok || gd.prediction_ok || gd.iou_ok;
        gd.accept = gd.hard_cap_ok
                 && gd.continuity_ok
                 && (gd.speed_ok || gd.prediction_ok || gd.iou_ok);
        return gd;
    }

    // detection_override: force accept after gap/streak
    bool detectionOverride(const DetectionBox& cand, double model_w, double model_h,
                           const ParsedFrameMetadata& parsed) {
        if (cand.conf < override_conf_) return false;

        const double min_dim = std::min(model_w, model_h);
        const double base_gate = std::max(gate_min_px_, gate_rel_ * min_dim);
        const double override_hard_cap = override_max_jump_rel_ * min_dim;
        const double far_gate = override_far_gate_mult_ * base_gate;
        const double pending_match_gate = override_far_match_mult_ * base_gate;
        const double moving_pending_gate = std::max(override_far_motion_rel_ * std::min(model_w, model_h),
                                                    0.5 * base_gate);

        uint64_t gap_frames = 9999;
        double dist_prev = 0.0;
        bool far_from_prev = false;
        if (!trail_.empty()) {
            gap_frames = frame_counter_ - trail_.back().frame;
            dist_prev = hypot2(centerX(cand) - trail_.back().x, centerY(cand) - trail_.back().y);
            far_from_prev = dist_prev > far_gate;
        }

        uint64_t real_gap_frames = 9999;
        bool far_from_real = false;
        if (have_last_real_detected_ && last_real_detected_frame_ > 0) {
            real_gap_frames = frame_counter_ - last_real_detected_frame_;
            const double dist_real = hypot2(centerX(cand) - centerX(last_real_detected_box_),
                                            centerY(cand) - centerY(last_real_detected_box_));
            far_from_real = dist_real > far_gate;
        }

        bool far_from_pred = false;
        if (kalman_initialized_) {
            const double d_pred = hypot2(centerX(cand) - kx_.pos(), centerY(cand) - ky_.pos());
            far_from_pred = d_pred > far_gate;
        }

        const bool stale = det_reject_streak_ >= override_after_
                        || (int)gap_frames >= reacquire_frames_
                        || (int)real_gap_frames >= reacquire_frames_;
        if (stale) {
            const auto court_bounds = evaluateCourtBounds(cand, parsed);
            if (court_bounds.veto) {
                clearPendingOverride();
                return false;
            }
        }

        // When the tracker is stale, treat large jumps from the recent coasted path
        // as provisional reacquires even if they happen to land near an older real hit.
        const bool far_reacquire = stale && (far_from_prev || far_from_real || far_from_pred);
        bool edge_zone_hit = false;
        const bool edge_jump_veto =
            coastEdgeJumpVetoApplies(cand, model_w, model_h, stale, dist_prev, edge_zone_hit);
        const bool provisional_reacquire = far_reacquire || edge_jump_veto;

        if (provisional_reacquire) {
            const bool pending_fresh = have_pending_override_
                                    && pending_override_last_frame_ + 1 >= frame_counter_;
            const bool pending_matches = pending_fresh
                                      && hypot2(centerX(cand) - centerX(pending_override_box_),
                                                centerY(cand) - centerY(pending_override_box_))
                                             <= pending_match_gate;
            if (!pending_matches) {
                pending_override_start_box_ = cand;
                pending_override_box_ = cand;
                have_pending_override_ = true;
                pending_override_hits_ = 1;
                pending_override_last_frame_ = frame_counter_;
                return false;
            }

            pending_override_box_ = cand;
            pending_override_hits_++;
            pending_override_last_frame_ = frame_counter_;
            const double pending_motion = hypot2(centerX(cand) - centerX(pending_override_start_box_),
                                                 centerY(cand) - centerY(pending_override_start_box_));
            if (!edge_jump_veto
                && pending_override_hits_ >= override_far_moving_confirm_frames_
                && pending_motion >= moving_pending_gate) {
                return true;
            }
            const int required_confirm_frames =
                edge_jump_veto ? std::max(override_far_confirm_frames_, coast_edge_confirm_frames_)
                               : override_far_confirm_frames_;
            if (shouldDebugLog() && edge_jump_veto) {
                logstream << "ball_tracker: frame=" << frame_counter_
                          << " edge_jump_veto cx=" << centerX(cand)
                          << " dist_prev=" << dist_prev
                          << " edge_zone_hit=" << edge_zone_hit
                          << " pending_hits=" << pending_override_hits_
                          << " required_hits=" << required_confirm_frames;
            }
            if (pending_override_hits_ < required_confirm_frames) {
                return false;
            }
        } else {
            clearPendingOverride();
        }

        // Keep override from reviving immediate huge jumps that gate already rejected.
        // Once a stale far candidate is under provisional confirmation, allow it to
        // accumulate hits instead of failing here every frame.
        if (!provisional_reacquire && gap_frames <= 1 && dist_prev > override_hard_cap) {
            return false;
        }

        if (det_reject_streak_ >= override_after_
            || (int)gap_frames >= reacquire_frames_
            || dist_prev <= 2.5 * base_gate) {
            return true;
        }
        return false;
    }

    CandidateSelection evaluateCandidates(const std::vector<DetectionBox>& ball_dets,
                                          double model_w, double model_h) const {
        CandidateSelection selection;
        selection.scores.assign(ball_dets.size(), -1e18);
        selection.gates.resize(ball_dets.size());

        for (int i = 0; i < (int)ball_dets.size(); ++i) {
            const auto& cand = ball_dets[(size_t)i];
            const double score = computeBallScore(cand, model_w, model_h);
            selection.scores[(size_t)i] = score;
            selection.gates[(size_t)i] = evaluateGate(cand, model_w, model_h);
            if (score > selection.best_score) {
                selection.best_score = score;
                selection.best_idx = i;
            }
            if (selection.gates[(size_t)i].accept && score > selection.best_gated_score) {
                selection.best_gated_score = score;
                selection.best_gated_idx = i;
            }
        }
        return selection;
    }

    TrackingDecision chooseDetection(const std::vector<DetectionBox>& ball_dets,
                                     const CandidateSelection& selection,
                                     double model_w, double model_h,
                                     const ParsedFrameMetadata& parsed) {
        TrackingDecision decision;

        if (selection.best_gated_idx >= 0) {
            const auto& cand = ball_dets[(size_t)selection.best_gated_idx];
            decision.accepted = true;
            decision.source = "detected";
            decision.tracked_det = cand;
            decision.ball_score = selection.scores[(size_t)selection.best_gated_idx];
            clearPendingOverride();
            return decision;
        }

        if (selection.best_idx < 0) return decision;

        const auto& cand = ball_dets[(size_t)selection.best_idx];
        decision.ball_score = selection.scores[(size_t)selection.best_idx];
        det_reject_streak_++;
        stat_dropped_by_gate_++;
        if (detectionOverride(cand, model_w, model_h, parsed)) {
            decision.accepted = true;
            decision.source = "override";
            decision.tracked_det = cand;
        }
        return decision;
    }

    void updateSpeed(double cx, double cy) {
        if (!trail_.empty()) {
            const auto& last = trail_.back();
            double step = hypot2(cx - last.x, cy - last.y);
            if (recent_speed_ < 0.0) {
                recent_speed_ = step;
            } else {
                recent_speed_ = 0.8 * recent_speed_ + 0.2 * step;
            }
        }
    }

    void recordVelocity() {
        if (!kalman_initialized_) return;
        double spd = hypot2(kx_.vel(), ky_.vel());
        stat_vel_min_ = std::min(stat_vel_min_, spd);
        stat_vel_max_ = std::max(stat_vel_max_, spd);
        stat_vel_sum_ += spd;
        stat_vel_count_++;
    }

    void recordConfidence(double conf) {
        int bucket = std::min((int)(conf * 10.0), 9);
        if (bucket < 0) bucket = 0;
        conf_histogram_[(size_t)bucket]++;
    }

    void recordBallSize(const DetectionBox& det, double model_w, double model_h) {
        double min_dim = std::min(model_w, model_h);
        if (min_dim <= 0.0) return;
        double max_box = std::max(boxWidth(det), boxHeight(det));
        stat_ball_size_sum_ += max_box / min_dim;
        stat_ball_size_count_++;
    }

    bool tryCoastDetection(DetectionBox& tracked_det, std::string& source,
                           double model_w, double model_h) {
        if (!coast_ || !kalman_initialized_ || trail_.empty() || coast_streak_ >= coast_max_) {
            return false;
        }

        const double px = kx_.pos();
        const double py = ky_.pos();
        if (!std::isfinite(px) || !std::isfinite(py)
            || px < 0.0 || px >= model_w
            || py < 0.0 || py >= model_h) {
            return false;
        }

        const double hard_cap = max_jump_rel_ * std::min(model_w, model_h);
        const double dist = hypot2(px - trail_.back().x, py - trail_.back().y);
        if (dist > 1.25 * hard_cap) return false;

        if (have_detected_template_) {
            tracked_det = last_detected_template_;
        } else {
            tracked_det = DetectionBox{};
            tracked_det.cls = 0;
            tracked_det.label = target_label_;
            tracked_det.has_label = true;
        }

        coast_streak_++;
        const double coasted_conf = last_accepted_conf_ * std::pow(0.85, (double)coast_streak_);
        double hw = 0.0;
        double hh = 0.0;
        if (have_last_emitted_) {
            hw = boxWidth(last_emitted_box_) * 0.5;
            hh = boxHeight(last_emitted_box_) * 0.5;
        } else {
            const double tgt = target_ball_size_rel_ * std::min(model_w, model_h);
            hw = hh = tgt * 0.5;
        }

        tracked_det.x1 = px - hw;
        tracked_det.y1 = py - hh;
        tracked_det.x2 = px + hw;
        tracked_det.y2 = py + hh;
        tracked_det.conf = coasted_conf;

        updateSpeed(px, py);
        addTrailPoint(trail_, (int)std::round(px), (int)std::round(py),
                      frame_counter_, false, 5, trail_max_);
        last_emitted_box_ = tracked_det;
        have_last_emitted_ = true;

        recordVelocity();
        source = "coasted";
        stat_coasted_++;
        return true;
    }

    void acceptDetection(const DetectionBox& det, const std::string& source) {
        double cx = centerX(det);
        double cy = centerY(det);

        if (source == "override") {
            // Hard reset Kalman
            initKalman(cx, cy);
        } else {
            if (!kalman_initialized_) {
                initKalman(cx, cy);
            } else {
                kx_.correct(cx);
                ky_.correct(cy);
            }
        }

        updateSpeed(cx, cy);
        if (source == "override") {
            trail_.clear();
            addTrailPoint(trail_, (int)std::round(cx), (int)std::round(cy),
                          frame_counter_, false, 5, trail_max_);
        } else {
            addTrailPoint(trail_, (int)std::round(cx), (int)std::round(cy),
                          frame_counter_, true, 5, trail_max_);
        }

        last_emitted_box_ = det;
        have_last_emitted_ = true;
        last_detected_template_ = det;
        have_detected_template_ = true;
        if (source == "detected") {
            last_real_detected_box_ = det;
            have_last_real_detected_ = true;
            last_real_detected_frame_ = frame_counter_;
        }
        last_accepted_conf_ = det.conf;
        coast_streak_ = 0;
        det_reject_streak_ = 0;
        clearPendingOverride();

        recordConfidence(det.conf);
        recordVelocity();
    }

    Parameters buildTrackedDetection(const DetectionBox& det, const std::string& source,
                                     int candidates_count, double ball_score) const {
        Parameters item;
        item["cls"] = det.cls;
        if (det.has_label) item["label"] = det.label;
        item["conf"] = det.conf;
        item["xyxy"] = {det.x1, det.y1, det.x2, det.y2};
        if (det.model_index >= 0) item["model_index"] = det.model_index;
        if (det.has_engine_name) item["engine_name"] = det.engine_name;
        item["track_id"] = track_id_;
        item["source"] = source;
        item["predicted"] = (source == "coasted");
        item["missed_frames"] = coast_streak_;
        item["coast_streak"] = coast_streak_;
        item["gate_status"] = source;
        item["ball_score"] = ball_score;
        item["candidates_count"] = candidates_count;
        if (kalman_initialized_) {
            double vx = kx_.vel();
            double vy = ky_.vel();
            item["velocity_x"] = vx;
            item["velocity_y"] = vy;
            item["velocity_px_per_frame"] = hypot2(vx, vy);
        }
        return item;
    }

    Parameters buildTrailArray() const {
        Parameters arr = Parameters::array();
        for (const auto& pt : trail_) {
            arr.push_back({pt.x, pt.y, (int64_t)pt.frame});
        }
        return arr;
    }

    void writeOutputMetadata(av::VideoFrame& frm, const ParsedFrameMetadata& parsed,
                             bool accepted, const DetectionBox& tracked_det,
                             const std::string& source, double ball_score) const {
        Parameters out_md;
        out_md["coord_space"] = parsed.md.value("coord_space", std::string("model"));
        out_md["model_width"] = parsed.model_w;
        out_md["model_height"] = parsed.model_h;
        if (parsed.md.contains("models")) out_md["models"] = parsed.md["models"];

        Parameters out_dets = parsed.other_det_items;
        if (accepted && finiteBox(tracked_det)) {
            out_dets.push_back(buildTrackedDetection(tracked_det, source,
                                                     (int)parsed.ball_dets.size(), ball_score));
        }
        out_md["detections"] = out_dets;
        out_md["trail"] = buildTrailArray();

        std::string serialized = out_md.dump();
        av_dict_set(&frm.raw()->metadata, metadata_key_.c_str(), serialized.c_str(), 0);
    }

    void logCandidateDetails(const std::vector<DetectionBox>& ball_dets,
                             const CandidateSelection& selection) const {
        if (!shouldDebugLog()) return;

        for (int i = 0; i < (int)ball_dets.size(); ++i) {
            const auto& cand = ball_dets[(size_t)i];
            const auto& gd = selection.gates[(size_t)i];
            logstream << "ball_tracker: frame=" << frame_counter_
                      << " cand[" << i << "]"
                      << " selected=" << (i == selection.best_idx ? 1 : 0)
                      << " selected_gated=" << (i == selection.best_gated_idx ? 1 : 0)
                      << " cx=" << centerX(cand)
                      << " cy=" << centerY(cand)
                      << " conf=" << cand.conf
                      << " score=" << selection.scores[(size_t)i]
                      << " gate_accept=" << gd.accept
                      << " hard_cap_ok=" << gd.hard_cap_ok
                      << " distance_ok=" << gd.distance_ok
                      << " prediction_ok=" << gd.prediction_ok
                      << " iou_ok=" << gd.iou_ok
                      << " speed_ok=" << gd.speed_ok
                      << " dist_prev=" << gd.dist_prev
                      << " dist_real=" << gd.dist_real
                      << " d_pred=" << gd.d_pred
                      << " base_gate=" << gd.base_gate
                      << " hard_cap=" << gd.hard_cap
                      << " real_gap_frames=" << gd.real_gap_frames;
        }
        if (have_pending_override_) {
            logstream << "ball_tracker: frame=" << frame_counter_
                      << " pending_override hits=" << pending_override_hits_
                      << " cx=" << centerX(pending_override_box_)
                      << " cy=" << centerY(pending_override_box_)
                      << " last_frame=" << pending_override_last_frame_;
        }
    }

    void logFrameSummary(size_t candidates_count, bool accepted, const std::string& source) const {
        if (!shouldDebugLog()) return;
        logstream << "ball_tracker: frame=" << frame_counter_
                  << " candidates=" << candidates_count
                  << " source=" << (accepted ? source : "none")
                  << " coast_streak=" << coast_streak_
                  << " reject_streak=" << det_reject_streak_;
    }

public:
    using NodeSISO::NodeSISO;
    bool consumeEofIfPresent() override {
        return false;
    }

    void resetInput() override {
        resetState();
    }

    ~BallTracker() {
        uint64_t total = stat_detected_ + stat_override_ + stat_coasted_ + stat_no_ball_;
        if (total == 0) return;

        auto pct = [&](uint64_t n) { return 100.0 * n / total; };
        logstream << "ball_tracker: === tracking summary ===";
        logstream << "  total frames:          " << total;
        logstream << "  detected (gated):      " << stat_detected_ << "  (" << pct(stat_detected_) << "%)";
        logstream << "  detected (override):   " << stat_override_ << "  (" << pct(stat_override_) << "%)";
        logstream << "  coasted:               " << stat_coasted_ << "  (" << pct(stat_coasted_) << "%)";
        logstream << "  no ball:               " << stat_no_ball_ << "  (" << pct(stat_no_ball_) << "%)";
        logstream << "  dropped by gate:       " << stat_dropped_by_gate_;
        if (stat_ball_size_count_ > 0) {
            logstream << "  avg accepted ball size: " << (100.0 * stat_ball_size_sum_ / stat_ball_size_count_) << "% of min(H,W)";
        }
        logstream << "ball_tracker: confidence histogram (accepted):";
        for (int i = 0; i < 10; ++i) {
            logstream << "  " << (i * 0.1) << "-" << ((i + 1) * 0.1) << ": " << conf_histogram_[(size_t)i];
        }
        if (stat_vel_count_ > 0) {
            logstream << "ball_tracker: velocity stats (px/frame):";
            logstream << "  min: " << stat_vel_min_ << "  max: " << stat_vel_max_
                      << "  avg: " << (stat_vel_sum_ / stat_vel_count_);
        }
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();

        if (isEofMarker(frm)) {
            resetState();
            this->sink_->put(frm);
            this->finished_ = true;
            return;
        }
        if (!frm) return;

        ++frame_counter_;
        const std::string shot_type = getShotType(frm.raw());
        updateShotMode(shot_type);

        ParsedFrameMetadata parsed;
        if (!parseFrameMetadata(frm, parsed)) return;
        if (handleCloseupSuppression(frm, shot_type, parsed)) return;

        predictKalman();

        const CandidateSelection selection = evaluateCandidates(parsed.ball_dets, parsed.model_w, parsed.model_h);
        TrackingDecision decision = chooseDetection(parsed.ball_dets, selection, parsed.model_w, parsed.model_h, parsed);

        logCandidateDetails(parsed.ball_dets, selection);

        if (decision.accepted) {
            acceptDetection(decision.tracked_det, decision.source);
            recordBallSize(decision.tracked_det, parsed.model_w, parsed.model_h);
            if (decision.source == "detected") stat_detected_++;
            else stat_override_++;
        } else {
            if (selection.best_idx < 0) {
                clearPendingOverride();
                det_reject_streak_ = 0;
            }
            if (!tryCoastDetection(decision.tracked_det, decision.source, parsed.model_w, parsed.model_h)) {
                coast_streak_ = 0;
                stat_no_ball_++;
            } else {
                decision.accepted = true;
            }
        }

        writeOutputMetadata(frm, parsed, decision.accepted, decision.tracked_det,
                            decision.source, decision.ball_score);

        if (decision.accepted && finiteBox(decision.tracked_det)) {
            dumpFrame(decision.source, centerX(decision.tracked_det), centerY(decision.tracked_det),
                      decision.tracked_det.conf);
        } else {
            dumpFrame("", 0.0, 0.0, 0.0);
        }

        logFrameSummary(parsed.ball_dets.size(), decision.accepted, decision.source);

        this->sink_->put(frm);
    }

    static std::shared_ptr<BallTracker> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<BallTracker>(edges, params);
        r->auto_eof_ = false;

        // Shot-aware mode handling
        if (params.count("camera_shot_metadata_key")) r->camera_shot_metadata_key_ = params["camera_shot_metadata_key"].get<std::string>();
        // Target filtering
        if (params.count("metadata_key")) r->metadata_key_ = params["metadata_key"].get<std::string>();
        if (params.count("target_label")) r->target_label_ = params["target_label"].get<std::string>();
        if (params.count("target_labels")) {
            if (!params["target_labels"].is_array()) throw Error("ball_tracker: target_labels must be a string array");
            const std::list<std::string> labels = jsonToStringList(params["target_labels"]);
            r->target_labels_.assign(labels.begin(), labels.end());
        }
        if (params.count("target_class")) r->target_class_ = params["target_class"];
        if (params.count("min_conf")) r->min_conf_ = params["min_conf"];

        // Best ball selection
        if (params.count("target_ball_size_rel")) r->target_ball_size_rel_ = params["target_ball_size_rel"];
        if (params.count("w_conf")) r->w_conf_ = params["w_conf"];
        if (params.count("w_dist")) r->w_dist_ = params["w_dist"];
        if (params.count("w_size")) r->w_size_ = params["w_size"];
        if (params.count("w_round")) r->w_round_ = params["w_round"];

        // Gating
        if (params.count("max_jump_rel")) r->max_jump_rel_ = params["max_jump_rel"];
        if (params.count("gate_rel")) r->gate_rel_ = params["gate_rel"];
        if (params.count("gate_min_px")) r->gate_min_px_ = params["gate_min_px"];
        if (params.count("gate_use_pred")) r->gate_use_pred_ = params["gate_use_pred"];
        if (params.count("min_iou")) r->min_iou_ = params["min_iou"];
        if (params.count("speed_mult")) r->speed_mult_ = params["speed_mult"];

        if (params.count("court_bounds_veto_enabled")) r->court_bounds_veto_enabled_ = params["court_bounds_veto_enabled"];
        if (params.count("override_off_court_veto_enabled")) r->court_bounds_veto_enabled_ = params["override_off_court_veto_enabled"];
        if (params.count("court_bounds_seg_metadata_key")) r->court_bounds_seg_metadata_key_ = params["court_bounds_seg_metadata_key"].get<std::string>();
        if (params.count("override_off_court_seg_metadata_key")) r->court_bounds_seg_metadata_key_ = params["override_off_court_seg_metadata_key"].get<std::string>();
        if (params.count("court_bounds_side_data_slot")) r->court_bounds_side_data_slot_ = params["court_bounds_side_data_slot"];
        if (params.count("override_off_court_side_data_slot")) r->court_bounds_side_data_slot_ = params["override_off_court_side_data_slot"];
        if (!yoloSegIsValidSlot(r->court_bounds_side_data_slot_)) {
            throw Error("ball_tracker: court_bounds_side_data_slot out of range [0," + std::to_string(kMaxYoloSegSlots - 1) + "]");
        }
        if (params.count("court_bounds_camera_shot_metadata_key")) r->court_bounds_camera_shot_metadata_key_ = params["court_bounds_camera_shot_metadata_key"].get<std::string>();
        if (params.count("court_bounds_require_wide_shot")) r->court_bounds_require_wide_shot_ = params["court_bounds_require_wide_shot"];
        if (params.count("court_bounds_min_court_coverage")) r->court_bounds_min_court_coverage_ = params["court_bounds_min_court_coverage"];
        if (params.count("court_bounds_class_indices")) {
            r->court_bounds_class_indices_.clear();
            for (const auto& item : params["court_bounds_class_indices"]) {
                r->court_bounds_class_indices_.insert(item.get<int>());
            }
        }
        if (params.count("override_off_court_class_indices")) {
            r->court_bounds_class_indices_.clear();
            for (const auto& item : params["override_off_court_class_indices"]) {
                r->court_bounds_class_indices_.insert(item.get<int>());
            }
        }
        if (params.count("court_bounds_mask_threshold")) r->court_bounds_mask_threshold_ = params["court_bounds_mask_threshold"];
        if (params.count("override_off_court_mask_threshold")) r->court_bounds_mask_threshold_ = params["override_off_court_mask_threshold"];
        if (params.count("court_bounds_sample_rows")) r->court_bounds_sample_rows_ = params["court_bounds_sample_rows"];
        if (params.count("court_bounds_min_horizontal_overlap_ratio")) r->court_bounds_min_horizontal_overlap_ratio_ = params["court_bounds_min_horizontal_overlap_ratio"];
        if (params.count("court_bounds_max_horizontal_outside_px")) r->court_bounds_max_horizontal_outside_px_ = params["court_bounds_max_horizontal_outside_px"];

        // Detection override
        if (params.count("override_conf")) r->override_conf_ = params["override_conf"];
        if (params.count("override_after")) r->override_after_ = params["override_after"];
        if (params.count("reacquire_frames")) r->reacquire_frames_ = params["reacquire_frames"];
        if (params.count("override_max_jump_rel")) r->override_max_jump_rel_ = params["override_max_jump_rel"];
        if (params.count("override_far_confirm_frames")) r->override_far_confirm_frames_ = params["override_far_confirm_frames"];
        if (params.count("override_far_gate_mult")) r->override_far_gate_mult_ = params["override_far_gate_mult"];
        if (params.count("override_far_match_mult")) r->override_far_match_mult_ = params["override_far_match_mult"];
        if (params.count("override_far_moving_confirm_frames")) r->override_far_moving_confirm_frames_ = params["override_far_moving_confirm_frames"];
        if (params.count("override_far_motion_rel")) r->override_far_motion_rel_ = params["override_far_motion_rel"];
        if (params.count("coast_edge_jump_veto_enabled")) r->coast_edge_jump_veto_enabled_ = params["coast_edge_jump_veto_enabled"];
        if (params.count("coast_edge_zone_rel")) r->coast_edge_zone_rel_ = params["coast_edge_zone_rel"];
        if (params.count("coast_edge_jump_rel")) r->coast_edge_jump_rel_ = params["coast_edge_jump_rel"];
        if (params.count("coast_edge_confirm_frames")) r->coast_edge_confirm_frames_ = params["coast_edge_confirm_frames"];

        // Coasting
        if (params.count("coast")) r->coast_ = params["coast"];
        if (params.count("coast_max")) r->coast_max_ = params["coast_max"];

        // Kalman
        if (params.count("kalman_q_pos")) r->kalman_q_pos_ = params["kalman_q_pos"];
        if (params.count("kalman_q_vel")) r->kalman_q_vel_ = params["kalman_q_vel"];
        if (params.count("kalman_r_meas")) r->kalman_r_meas_ = params["kalman_r_meas"];

        // Trail
        if (params.count("trail_max")) r->trail_max_ = params["trail_max"];

        // Debug
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"];

        // Dump file
        if (params.count("dump_file")) {
            r->dump_file_path_ = params["dump_file"].get<std::string>();
            r->dump_file_.open(r->dump_file_path_, std::ios::out | std::ios::trunc);
            if (r->dump_file_.is_open()) {
                r->dump_file_ << "frame,x,y,conf,source,dx,dy,dist\n";
            } else {
                std::cerr << "ball_tracker: WARNING: could not open dump file: " << r->dump_file_path_ << std::endl;
            }
        }

        // Initialize Kalman with configured params
        r->kx_ = Kalman1D(r->kalman_q_pos_, r->kalman_q_vel_, r->kalman_r_meas_);
        r->ky_ = Kalman1D(r->kalman_q_pos_, r->kalman_q_vel_, r->kalman_r_meas_);

        return r;
    }
};

DECLNODE(ball_tracker, BallTracker)
