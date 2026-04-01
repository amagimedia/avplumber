#include "node_common.hpp"
#include "../kalman1d.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <string>
#include <vector>

namespace {

struct DetectionBox {
    int cls = -1;
    std::string label;
    bool has_label = false;
    double conf = 0.0;
    double x1 = 0.0, y1 = 0.0, x2 = 0.0, y2 = 0.0;
    int model_index = -1;
    std::string engine_name;
    bool has_engine_name = false;
};

struct TrailPoint {
    int x = 0, y = 0;
    uint64_t frame = 0;
};

static double boxWidth(const DetectionBox& b) { return b.x2 - b.x1; }
static double boxHeight(const DetectionBox& b) { return b.y2 - b.y1; }
static double centerX(const DetectionBox& b) { return (b.x1 + b.x2) * 0.5; }
static double centerY(const DetectionBox& b) { return (b.y1 + b.y2) * 0.5; }

static bool finiteBox(const DetectionBox& b) {
    return std::isfinite(b.x1) && std::isfinite(b.y1)
        && std::isfinite(b.x2) && std::isfinite(b.y2)
        && b.x2 > b.x1 && b.y2 > b.y1;
}

static double iou(const DetectionBox& a, const DetectionBox& b) {
    const double ix1 = std::max(a.x1, b.x1);
    const double iy1 = std::max(a.y1, b.y1);
    const double ix2 = std::min(a.x2, b.x2);
    const double iy2 = std::min(a.y2, b.y2);
    const double iw = std::max(0.0, ix2 - ix1);
    const double ih = std::max(0.0, iy2 - iy1);
    const double inter = iw * ih;
    if (inter <= 0.0) return 0.0;
    const double aa = std::max(0.0, boxWidth(a)) * std::max(0.0, boxHeight(a));
    const double ab = std::max(0.0, boxWidth(b)) * std::max(0.0, boxHeight(b));
    const double uni = aa + ab - inter;
    return uni > 0.0 ? inter / uni : 0.0;
}

static double hypot2(double dx, double dy) { return std::sqrt(dx * dx + dy * dy); }

static double sizePenalty(double w, double h, double model_w, double model_h, double target_rel) {
    const double s = std::max(w, h);
    const double tgt = target_rel * std::min(model_w, model_h);
    if (tgt <= 0.0) return 0.0;
    return std::clamp(std::abs(s - tgt) / tgt, 0.0, 2.0) * 0.5;
}

static double roundnessPenalty(double w, double h) {
    const double ar = w / std::max(h, 1e-6);
    return 1.0 - std::exp(-((ar - 1.0) * (ar - 1.0)) / 0.15);
}

static void addTrailPoint(std::deque<TrailPoint>& trail, int x, int y, uint64_t frame,
                          bool densify, int max_gap, size_t max_len) {
    if (densify && !trail.empty()) {
        const auto& last = trail.back();
        int gap = (int)(frame - last.frame);
        if (gap > 1 && gap <= max_gap) {
            for (int t = 1; t < gap; ++t) {
                double alpha = (double)t / (double)gap;
                int xi = (int)std::round((1.0 - alpha) * last.x + alpha * x);
                int yi = (int)std::round((1.0 - alpha) * last.y + alpha * y);
                trail.push_back({xi, yi, last.frame + (uint64_t)t});
                while (trail.size() > max_len) trail.pop_front();
            }
        }
    }
    trail.push_back({x, y, frame});
    while (trail.size() > max_len) trail.pop_front();
}

} // anonymous namespace

class BallTracker : public NodeSISO<av::VideoFrame, av::VideoFrame>, public IInputReset {
private:
    // Target filtering
    std::string metadata_key_ = "yolo_detections";
    std::string target_label_ = "basketball";
    std::vector<std::string> target_labels_;
    int target_class_ = -1;
    double min_conf_ = 0.04;

    // Best ball selection
    double target_ball_size_rel_ = 0.036;
    double w_conf_ = 1.0;
    double w_dist_ = 0.015;
    double w_size_ = 0.5;
    double w_round_ = 0.4;

    // Gating
    double max_jump_rel_ = 0.12;
    double gate_rel_ = 0.06;
    double gate_min_px_ = 12.0;
    bool gate_use_pred_ = true;
    double min_iou_ = 0.20;
    double speed_mult_ = 3.0;

    // Detection override
    double override_conf_ = 0.28;
    int override_after_ = 2;
    int reacquire_frames_ = 6;

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

    // --- Internal state ---
    Kalman1D kx_, ky_;
    bool kalman_initialized_ = false;
    std::deque<TrailPoint> trail_;
    DetectionBox last_emitted_box_;
    bool have_last_emitted_ = false;
    DetectionBox last_detected_template_; // for carrying forward model_index etc on coast
    bool have_detected_template_ = false;
    double recent_speed_ = -1.0;
    int coast_streak_ = 0;
    int det_reject_streak_ = 0;
    uint64_t frame_counter_ = 0;
    int track_id_ = 1;
    double last_accepted_conf_ = 0.0;

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

    void initKalman(double cx, double cy) {
        kx_ = Kalman1D(kalman_q_pos_, kalman_q_vel_, kalman_r_meas_);
        ky_ = Kalman1D(kalman_q_pos_, kalman_q_vel_, kalman_r_meas_);
        kx_.init(cx);
        ky_.init(cy);
        kalman_initialized_ = true;
    }

    void resetState() {
        kalman_initialized_ = false;
        trail_.clear();
        have_last_emitted_ = false;
        have_detected_template_ = false;
        recent_speed_ = -1.0;
        coast_streak_ = 0;
        det_reject_streak_ = 0;
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

    // select_best_ball: returns index into ball_dets, or -1
    int selectBestBall(const std::vector<DetectionBox>& ball_dets, double model_w, double model_h) const {
        if (ball_dets.empty()) return -1;

        double pred_cx = 0.0, pred_cy = 0.0;
        bool have_pred = kalman_initialized_;
        if (have_pred) {
            pred_cx = kx_.pos();
            pred_cy = ky_.pos();
        }

        const double norm = 0.5 * (model_w + model_h);
        int best_idx = -1;
        double best_score = -1e18;

        for (int i = 0; i < (int)ball_dets.size(); ++i) {
            const auto& d = ball_dets[i];
            const double cx = centerX(d);
            const double cy = centerY(d);
            const double w = boxWidth(d);
            const double h = boxHeight(d);

            double s_conf = d.conf;
            double dist_pen = 0.0;
            if (have_pred && norm > 0.0) {
                dist_pen = hypot2(cx - pred_cx, cy - pred_cy) / norm;
            }
            double size_pen = sizePenalty(w, h, model_w, model_h, target_ball_size_rel_);
            double round_pen = roundnessPenalty(w, h);

            double score = w_conf_ * s_conf
                         - w_dist_ * dist_pen
                         - w_size_ * size_pen
                         - w_round_ * round_pen;

            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }
        return best_idx;
    }

    // gate_accept: 5-stage gating
    bool gateAccept(const DetectionBox& cand, double model_w, double model_h) const {
        const double cx = centerX(cand);
        const double cy = centerY(cand);
        const double min_dim = std::min(model_w, model_h);
        const double hard_cap = max_jump_rel_ * min_dim;
        const double base_gate = std::max(gate_min_px_, gate_rel_ * min_dim);

        // Distance to last trail point
        double dist_prev = 0.0;
        if (!trail_.empty()) {
            const auto& last = trail_.back();
            dist_prev = hypot2(cx - last.x, cy - last.y);
        }

        // Stage 1: Hard cap
        if (!trail_.empty() && dist_prev > hard_cap) {
            return false;
        }

        // Stage 2: Distance gate
        bool distance_ok = trail_.empty() || (dist_prev <= base_gate);

        // Stage 3: Prediction gate (OR)
        bool prediction_ok = false;
        if (gate_use_pred_ && kalman_initialized_) {
            double d_pred = hypot2(cx - kx_.pos(), cy - ky_.pos());
            prediction_ok = d_pred <= base_gate * 1.25;
        }

        // Stage 4: IoU continuity
        bool iou_ok = true;
        if (have_last_emitted_) {
            iou_ok = iou(cand, last_emitted_box_) >= min_iou_;
        }

        // Stage 5: Speed constraint
        bool speed_ok = true;
        if (recent_speed_ > 0.0 && !trail_.empty()) {
            speed_ok = dist_prev <= speed_mult_ * (recent_speed_ + 1e-6);
        }

        return (distance_ok && iou_ok && speed_ok) || prediction_ok;
    }

    // detection_override: force accept after gap/streak
    bool detectionOverride(const DetectionBox& cand, double model_w, double model_h) {
        if (cand.conf < override_conf_) return false;

        const double min_dim = std::min(model_w, model_h);
        const double base_gate = std::max(gate_min_px_, gate_rel_ * min_dim);

        uint64_t gap_frames = 9999;
        double dist_prev = 0.0;
        if (!trail_.empty()) {
            gap_frames = frame_counter_ - trail_.back().frame;
            dist_prev = hypot2(centerX(cand) - trail_.back().x, centerY(cand) - trail_.back().y);
        }

        if (det_reject_streak_ >= override_after_
            || (int)gap_frames >= reacquire_frames_
            || dist_prev <= 2.5 * base_gate) {
            return true;
        }
        return false;
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
        addTrailPoint(trail_, (int)std::round(cx), (int)std::round(cy),
                      frame_counter_, true, 5, trail_max_);

        last_emitted_box_ = det;
        have_last_emitted_ = true;
        last_detected_template_ = det;
        have_detected_template_ = true;
        last_accepted_conf_ = det.conf;
        coast_streak_ = 0;
        det_reject_streak_ = 0;

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

public:
    using NodeSISO::NodeSISO;

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
        if (!frm) return;

        if (isEofMarker(frm)) {
            resetState();
            this->sink_->put(frm);
            return;
        }

        ++frame_counter_;

        // Parse metadata
        const AVFrame* raw = frm.raw();
        if (!raw || !raw->metadata) {
            // No metadata — run Kalman predict, maybe coast
            if (kalman_initialized_) kx_.predict(1.0);
            if (kalman_initialized_) ky_.predict(1.0);
            stat_no_ball_++;
            this->sink_->put(frm);
            return;
        }

        AVDictionaryEntry* entry = av_dict_get(raw->metadata, metadata_key_.c_str(), nullptr, 0);
        if (!entry || !entry->value) {
            if (kalman_initialized_) { kx_.predict(1.0); ky_.predict(1.0); }
            stat_no_ball_++;
            this->sink_->put(frm);
            return;
        }

        Parameters md;
        try {
            md = Parameters::parse(entry->value);
        } catch (...) {
            if (kalman_initialized_) { kx_.predict(1.0); ky_.predict(1.0); }
            stat_no_ball_++;
            this->sink_->put(frm);
            return;
        }

        const double model_w = md.value("model_width", (double)frm.width());
        const double model_h = md.value("model_height", (double)frm.height());

        // Split detections into ball and other
        std::vector<DetectionBox> ball_dets;
        Parameters other_det_items = Parameters::array();

        if (md.contains("detections") && md["detections"].is_array()) {
            for (const auto& item : md["detections"]) {
                if (!item.is_object()) continue;
                DetectionBox det = parseOneDetection(item);
                if (!finiteBox(det)) {
                    other_det_items.push_back(item);
                    continue;
                }
                if (matchesTarget(det) && det.conf >= min_conf_) {
                    ball_dets.push_back(det);
                } else {
                    other_det_items.push_back(item);
                }
            }
        }

        // Kalman predict step (every frame)
        if (kalman_initialized_) {
            kx_.predict(1.0);
            ky_.predict(1.0);
        }

        // Select best ball candidate
        int best_idx = selectBestBall(ball_dets, model_w, model_h);

        std::string source;
        bool accepted = false;
        double ball_score = 0.0;
        DetectionBox tracked_det;

        if (best_idx >= 0) {
            const auto& cand = ball_dets[(size_t)best_idx];

            // Compute ball_score for metadata
            {
                double cx = centerX(cand), cy = centerY(cand);
                double bw = boxWidth(cand), bh = boxHeight(cand);
                double norm = 0.5 * (model_w + model_h);
                double dist_pen = 0.0;
                if (kalman_initialized_ && norm > 0.0) {
                    dist_pen = hypot2(cx - kx_.pos(), cy - ky_.pos()) / norm;
                }
                ball_score = w_conf_ * cand.conf
                           - w_dist_ * dist_pen
                           - w_size_ * sizePenalty(bw, bh, model_w, model_h, target_ball_size_rel_)
                           - w_round_ * roundnessPenalty(bw, bh);
            }

            if (gateAccept(cand, model_w, model_h)) {
                accepted = true;
                source = "detected";
                tracked_det = cand;
            } else {
                // Try detection override
                det_reject_streak_++;
                stat_dropped_by_gate_++;
                if (detectionOverride(cand, model_w, model_h)) {
                    accepted = true;
                    source = "override";
                    tracked_det = cand;
                }
            }
        }

        if (accepted) {
            acceptDetection(tracked_det, source);
            recordBallSize(tracked_det, model_w, model_h);
            if (source == "detected") stat_detected_++;
            else stat_override_++;
        } else {
            if (best_idx >= 0) {
                // Had candidate but rejected
                // det_reject_streak already incremented above
            } else {
                det_reject_streak_ = 0;
            }

            // Try coasting
            bool coasted = false;
            if (coast_ && kalman_initialized_ && !trail_.empty()
                && coast_streak_ < coast_max_) {
                double px = kx_.pos();
                double py = ky_.pos();
                if (std::isfinite(px) && std::isfinite(py)
                    && px >= 0.0 && px < model_w
                    && py >= 0.0 && py < model_h) {
                    double hard_cap = max_jump_rel_ * std::min(model_w, model_h);
                    double dist = hypot2(px - trail_.back().x, py - trail_.back().y);
                    if (dist <= 1.25 * hard_cap) {
                        // Build coasted detection from template
                        if (have_detected_template_) {
                            tracked_det = last_detected_template_;
                        } else {
                            tracked_det = DetectionBox{};
                            tracked_det.cls = 0;
                            tracked_det.label = target_label_;
                            tracked_det.has_label = true;
                        }
                        coast_streak_++;
                        double coasted_conf = last_accepted_conf_ * std::pow(0.85, (double)coast_streak_);
                        // Rebuild box around predicted center using last box size
                        double hw = 0.0, hh = 0.0;
                        if (have_last_emitted_) {
                            hw = boxWidth(last_emitted_box_) * 0.5;
                            hh = boxHeight(last_emitted_box_) * 0.5;
                        } else {
                            double tgt = target_ball_size_rel_ * std::min(model_w, model_h);
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
                        accepted = true;
                        coasted = true;
                        stat_coasted_++;
                    }
                }
            }

            if (!coasted) {
                coast_streak_ = 0;
                stat_no_ball_++;
            }
        }

        // Rebuild metadata
        Parameters out_md;
        out_md["coord_space"] = md.value("coord_space", std::string("model"));
        out_md["model_width"] = model_w;
        out_md["model_height"] = model_h;
        if (md.contains("models")) out_md["models"] = md["models"];

        Parameters out_dets = other_det_items;
        if (accepted && finiteBox(tracked_det)) {
            out_dets.push_back(buildTrackedDetection(tracked_det, source,
                                                     (int)ball_dets.size(), ball_score));
        }
        out_md["detections"] = out_dets;
        out_md["trail"] = buildTrailArray();

        std::string serialized = out_md.dump();
        av_dict_set(&frm.raw()->metadata, metadata_key_.c_str(), serialized.c_str(), 0);

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "ball_tracker: frame=" << frame_counter_
                      << " candidates=" << ball_dets.size()
                      << " source=" << (accepted ? source : "none")
                      << " coast_streak=" << coast_streak_
                      << " reject_streak=" << det_reject_streak_;
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<BallTracker> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<BallTracker>(edges, params);

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

        // Detection override
        if (params.count("override_conf")) r->override_conf_ = params["override_conf"];
        if (params.count("override_after")) r->override_after_ = params["override_after"];
        if (params.count("reacquire_frames")) r->reacquire_frames_ = params["reacquire_frames"];

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

        // Initialize Kalman with configured params
        r->kx_ = Kalman1D(r->kalman_q_pos_, r->kalman_q_vel_, r->kalman_r_meas_);
        r->ky_ = Kalman1D(r->kalman_q_pos_, r->kalman_q_vel_, r->kalman_r_meas_);

        return r;
    }
};

DECLNODE(ball_tracker, BallTracker)
