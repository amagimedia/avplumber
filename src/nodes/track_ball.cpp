#include "node_common.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {
struct DetectionBox {
    int cls = -1;
    std::string label;
    bool has_label = false;
    double conf = 0.0;
    double x1 = 0.0;
    double y1 = 0.0;
    double x2 = 0.0;
    double y2 = 0.0;
};

struct MetadataEnvelope {
    int version = 1;
    std::string coord_space = "model";
    double model_width = 0.0;
    double model_height = 0.0;
    Parameters thresholds;
};

struct TrackState {
    bool active = false;
    int track_id = 0;
    DetectionBox box;
    double vx = 0.0;
    double vy = 0.0;
    int missed_frames = 0;
};

struct TrackSample {
    DetectionBox box;
    bool predicted = false;
};

static bool finiteBox(const DetectionBox& box) {
    return std::isfinite(box.x1) && std::isfinite(box.y1)
        && std::isfinite(box.x2) && std::isfinite(box.y2)
        && box.x2 > box.x1 && box.y2 > box.y1;
}

static double boxWidth(const DetectionBox& box) {
    return box.x2 - box.x1;
}

static double boxHeight(const DetectionBox& box) {
    return box.y2 - box.y1;
}

static double centerX(const DetectionBox& box) {
    return (box.x1 + box.x2) * 0.5;
}

static double centerY(const DetectionBox& box) {
    return (box.y1 + box.y2) * 0.5;
}

static DetectionBox shiftedBox(const DetectionBox& box, double dx, double dy) {
    DetectionBox out = box;
    out.x1 += dx;
    out.x2 += dx;
    out.y1 += dy;
    out.y2 += dy;
    return out;
}

static DetectionBox clampBoxToCanvas(const DetectionBox& box, double width, double height) {
    DetectionBox out = box;
    if (width > 0.0) {
        out.x1 = std::max(0.0, std::min(out.x1, width));
        out.x2 = std::max(0.0, std::min(out.x2, width));
    }
    if (height > 0.0) {
        out.y1 = std::max(0.0, std::min(out.y1, height));
        out.y2 = std::max(0.0, std::min(out.y2, height));
    }
    if (out.x2 < out.x1) std::swap(out.x1, out.x2);
    if (out.y2 < out.y1) std::swap(out.y1, out.y2);
    return out;
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
    const double area_a = std::max(0.0, boxWidth(a)) * std::max(0.0, boxHeight(a));
    const double area_b = std::max(0.0, boxWidth(b)) * std::max(0.0, boxHeight(b));
    const double uni = area_a + area_b - inter;
    return uni > 0.0 ? inter / uni : 0.0;
}

static double centerDistance(const DetectionBox& a, const DetectionBox& b) {
    const double dx = centerX(a) - centerX(b);
    const double dy = centerY(a) - centerY(b);
    return std::sqrt(dx * dx + dy * dy);
}
}

class TrackBall : public NodeSISO<av::VideoFrame, av::VideoFrame>, public IInputReset {
private:
    std::string metadata_key_in_ = "yolo_detections_v1";
    std::string metadata_key_out_ = "ball_track_v1";
    std::string target_label_ = "sports ball";
    int target_class_ = -1;
    std::vector<std::string> target_labels_;
    std::vector<int> target_classes_;
    double min_conf_ = 0.10;
    int max_missed_frames_ = 8;
    double max_center_distance_ = 160.0;
    double min_iou_match_ = 0.0;
    double match_min_motion_ = 2.0;
    double match_max_motion_ = 64.0;
    double match_min_cosine_similarity_ = -0.2;
    double match_max_prediction_error_ = 28.0;
    double match_max_velocity_delta_ = 24.0;
    double slow_match_max_prediction_error_ = 12.0;
    double slow_match_min_overlap_ = 0.10;
    double huge_jump_frame_fraction_ = 0.25;
    int huge_jump_history_window_ = 8;
    int history_size_ = 30;
    int history_motion_window_ = 12;
    double history_match_min_cosine_similarity_ = -0.1;
    double history_max_motion_scale_ = 2.5;
    double history_max_motion_slack_ = 16.0;
    double acquisition_min_motion_ = 4.0;
    double acquisition_max_match_distance_ = 120.0;
    double acquisition_min_cosine_similarity_ = 0.2;
    bool emit_predicted_ = true;
    double prediction_decay_ = 0.92;
    double velocity_smoothing_ = 0.60;
    int debug_log_every_n_ = 0;
    uint64_t frame_counter_ = 0;
    int next_track_id_ = 1;
    TrackState track_;
    std::vector<TrackSample> track_history_;
    std::vector<DetectionBox> prev_prev_dets_;
    std::vector<DetectionBox> prev_dets_;

    void clearTrack() {
        track_ = TrackState{};
    }

    void clearHistory() {
        track_history_.clear();
        prev_prev_dets_.clear();
        prev_dets_.clear();
    }

    void resetTrackHistory() {
        track_history_.clear();
    }

    void pushTrackHistory(const DetectionBox& box, bool predicted) {
        track_history_.push_back(TrackSample{box, predicted});
        const int keep = std::max(1, history_size_);
        if ((int)track_history_.size() > keep) {
            track_history_.erase(track_history_.begin(),
                                 track_history_.begin() + ((int)track_history_.size() - keep));
        }
    }

    struct MotionStats {
        bool have_history = false;
        bool have_direction = false;
        double avg_vx = 0.0;
        double avg_vy = 0.0;
        double avg_speed = 0.0;
        double max_speed = 0.0;
    };

    MotionStats computeMotionStats() const {
        MotionStats stats;
        if (track_history_.size() < 2) {
            const double speed = std::sqrt(track_.vx * track_.vx + track_.vy * track_.vy);
            if (speed > 0.0) {
                stats.have_history = true;
                stats.avg_vx = track_.vx;
                stats.avg_vy = track_.vy;
                stats.avg_speed = speed;
                stats.max_speed = speed;
                stats.have_direction = speed >= match_min_motion_;
            }
            return stats;
        }

        const size_t motion_count = track_history_.size() - 1;
        const size_t take = (size_t)std::max(1, history_motion_window_);
        const size_t start = motion_count > take ? track_history_.size() - take : 1;

        int used = 0;
        for (size_t i = start; i < track_history_.size(); ++i) {
            const DetectionBox& prev = track_history_[i - 1].box;
            const DetectionBox& cur = track_history_[i].box;
            const double vx = centerX(cur) - centerX(prev);
            const double vy = centerY(cur) - centerY(prev);
            const double speed = std::sqrt(vx * vx + vy * vy);
            stats.avg_vx += vx;
            stats.avg_vy += vy;
            stats.avg_speed += speed;
            stats.max_speed = std::max(stats.max_speed, speed);
            ++used;
        }

        if (used <= 0) {
            return stats;
        }

        stats.have_history = true;
        stats.avg_vx /= used;
        stats.avg_vy /= used;
        stats.avg_speed /= used;
        stats.have_direction = std::sqrt(stats.avg_vx * stats.avg_vx + stats.avg_vy * stats.avg_vy) >= match_min_motion_;
        return stats;
    }

    bool recentHistoryHasHugeJump(double frame_span) const {
        if (frame_span <= 0.0 || track_history_.size() < 2) {
            return false;
        }

        const double huge_jump_px = frame_span * huge_jump_frame_fraction_;
        const size_t take = (size_t)std::max(1, huge_jump_history_window_);
        const size_t start = track_history_.size() > (take + 1) ? track_history_.size() - (take + 1) : 1;
        for (size_t i = start; i < track_history_.size(); ++i) {
            const DetectionBox& prev = track_history_[i - 1].box;
            const DetectionBox& cur = track_history_[i].box;
            if (centerDistance(prev, cur) >= huge_jump_px) {
                return true;
            }
        }
        return false;
    }

    bool detectionMatchesTarget(const DetectionBox& det) const {
        bool class_match = false;
        bool label_match = false;
        if (target_class_ >= 0 && det.cls == target_class_) {
            class_match = true;
        }
        for (int cls : target_classes_) {
            if (det.cls == cls) {
                class_match = true;
                break;
            }
        }
        if (!target_label_.empty() && det.has_label && det.label == target_label_) {
            label_match = true;
        }
        if (det.has_label) {
            for (const std::string& label : target_labels_) {
                if (det.label == label) {
                    label_match = true;
                    break;
                }
            }
        }
        const bool have_class_targets = target_class_ >= 0 || !target_classes_.empty();
        const bool have_label_targets = !target_label_.empty() || !target_labels_.empty();
        if (!have_class_targets && !have_label_targets) {
            return true;
        }
        return class_match || label_match;
    }

    bool parseDetections(const av::VideoFrame& frm, MetadataEnvelope& env_out, std::vector<DetectionBox>& dets_out) const {
        dets_out.clear();
        env_out = MetadataEnvelope{};
        env_out.model_width = frm.width();
        env_out.model_height = frm.height();

        const AVFrame* raw = frm.raw();
        if (!raw || !raw->metadata) return false;

        AVDictionaryEntry* entry = av_dict_get(raw->metadata, metadata_key_in_.c_str(), nullptr, 0);
        if (!entry || !entry->value) return false;

        try {
            Parameters md = Parameters::parse(entry->value);
            env_out.version = md.value("version", 1);
            env_out.coord_space = md.value("coord_space", std::string("model"));
            env_out.model_width = md.value("model_width", (double)frm.width());
            env_out.model_height = md.value("model_height", (double)frm.height());
            if (md.contains("thresholds")) {
                env_out.thresholds = md["thresholds"];
            }

            if (!md.contains("detections") || !md["detections"].is_array()) {
                return true;
            }

            for (const auto& item : md["detections"]) {
                if (!item.is_object()) continue;
                if (!item.contains("xyxy") || !item["xyxy"].is_array() || item["xyxy"].size() < 4) continue;

                DetectionBox det;
                det.cls = item.value("cls", -1);
                if (item.contains("label") && item["label"].is_string()) {
                    det.label = item["label"].get<std::string>();
                    det.has_label = true;
                }
                det.conf = item.value("conf", 0.0);
                det.x1 = item["xyxy"][0].get<double>();
                det.y1 = item["xyxy"][1].get<double>();
                det.x2 = item["xyxy"][2].get<double>();
                det.y2 = item["xyxy"][3].get<double>();

                if (det.conf < min_conf_) continue;
                if (!finiteBox(det)) continue;
                if (!detectionMatchesTarget(det)) continue;
                dets_out.push_back(det);
            }
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    DetectionBox predictBox(const MetadataEnvelope& env) const {
        DetectionBox predicted = shiftedBox(track_.box, track_.vx, track_.vy);
        predicted.conf = std::max(min_conf_, track_.box.conf * std::pow(prediction_decay_, (double)(track_.missed_frames + 1)));
        predicted.cls = track_.box.cls;
        predicted.label = track_.box.label;
        predicted.has_label = track_.box.has_label;
        return clampBoxToCanvas(predicted, env.model_width, env.model_height);
    }

    int chooseBestMatch(const std::vector<DetectionBox>& dets,
                        const DetectionBox& reference_box,
                        double frame_span,
                        double max_center_distance,
                        int frame_gap) const {
        int best_index = -1;
        double best_score = -std::numeric_limits<double>::infinity();
        const double gap = std::max(1, frame_gap);
        const double track_speed = std::sqrt(track_.vx * track_.vx + track_.vy * track_.vy);
        const MotionStats stats = computeMotionStats();
        const bool recent_huge_jump = recentHistoryHasHugeJump(frame_span);
        const double history_max_motion = stats.have_history
            ? std::max(match_max_motion_, stats.max_speed * history_max_motion_scale_ + history_max_motion_slack_)
            : match_max_motion_;
        for (size_t i = 0; i < dets.size(); ++i) {
            const DetectionBox& det = dets[i];
            const double dist = centerDistance(reference_box, det);
            const double overlap = iou(reference_box, det);
            const double step_dist = centerDistance(track_.box, det) / gap;
            if (dist > max_center_distance && overlap < min_iou_match_) {
                continue;
            }
            if (dist > match_max_prediction_error_) {
                continue;
            }

            const bool slow_candidate = step_dist < match_min_motion_;
            if (slow_candidate) {
                if (recent_huge_jump) {
                    continue;
                }
                if (dist > slow_match_max_prediction_error_ && overlap < slow_match_min_overlap_) {
                    continue;
                }
            }
            if (step_dist > history_max_motion) {
                continue;
            }

            const double meas_vx = (centerX(det) - centerX(track_.box)) / gap;
            const double meas_vy = (centerY(det) - centerY(track_.box)) / gap;
            double cosine_bonus = 0.0;
            if (!slow_candidate && track_speed >= match_min_motion_) {
                const double denom = std::max(1.0, track_speed * step_dist);
                const double cosine = (track_.vx * meas_vx + track_.vy * meas_vy) / denom;
                if (cosine < match_min_cosine_similarity_) {
                    continue;
                }
                cosine_bonus = cosine * 2.0;

                const double velocity_delta = std::sqrt((meas_vx - track_.vx) * (meas_vx - track_.vx)
                                                      + (meas_vy - track_.vy) * (meas_vy - track_.vy));
                if (velocity_delta > match_max_velocity_delta_) {
                    continue;
                }
            }

            double history_bonus = 0.0;
            if (!slow_candidate && stats.have_direction) {
                const double history_speed = std::sqrt(stats.avg_vx * stats.avg_vx + stats.avg_vy * stats.avg_vy);
                const double denom = std::max(1.0, history_speed * step_dist);
                const double cosine = (stats.avg_vx * meas_vx + stats.avg_vy * meas_vy) / denom;
                if (cosine < history_match_min_cosine_similarity_) {
                    continue;
                }
                history_bonus = cosine * 3.0;
            }

            double slow_bonus = 0.0;
            if (slow_candidate) {
                slow_bonus = overlap * 3.0 + (1.0 - dist / std::max(1.0, slow_match_max_prediction_error_)) * 3.0;
            }

            const double score = det.conf * 4.0
                               + overlap * 2.0
                               + cosine_bonus
                               + history_bonus
                               + slow_bonus
                               - 2.0 * dist / std::max(1.0, match_max_prediction_error_)
                               - step_dist / std::max(1.0, history_max_motion);
            if (score > best_score) {
                best_score = score;
                best_index = (int)i;
            }
        }
        return best_index;
    }

    int chooseMovingAcquisition(const std::vector<DetectionBox>& dets) const {
        if (prev_dets_.empty() || prev_prev_dets_.empty()) {
            return -1;
        }

        int best_index = -1;
        double best_score = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < dets.size(); ++i) {
            const DetectionBox& det = dets[i];
            for (const DetectionBox& prev : prev_dets_) {
                const double step_dist = centerDistance(prev, det);
                if (step_dist > acquisition_max_match_distance_) continue;
                if (step_dist < acquisition_min_motion_) continue;

                const double vx1 = centerX(det) - centerX(prev);
                const double vy1 = centerY(det) - centerY(prev);
                const double mag1 = std::sqrt(vx1 * vx1 + vy1 * vy1);
                if (mag1 < acquisition_min_motion_) continue;

                for (const DetectionBox& prev_prev : prev_prev_dets_) {
                    const double prev_step_dist = centerDistance(prev_prev, prev);
                    if (prev_step_dist > acquisition_max_match_distance_) continue;
                    if (prev_step_dist < acquisition_min_motion_) continue;

                    const double vx0 = centerX(prev) - centerX(prev_prev);
                    const double vy0 = centerY(prev) - centerY(prev_prev);
                    const double mag0 = std::sqrt(vx0 * vx0 + vy0 * vy0);
                    if (mag0 < acquisition_min_motion_) continue;

                    const double denom = std::max(1.0, mag0 * mag1);
                    const double cosine = (vx0 * vx1 + vy0 * vy1) / denom;
                    if (cosine < acquisition_min_cosine_similarity_) continue;

                    const double motion_score = (mag0 + mag1) / std::max(1.0, acquisition_min_motion_);
                    const double score = det.conf * 4.0 + prev.conf * 2.0 + motion_score + cosine * 2.0;
                    if (score > best_score) {
                        best_score = score;
                        best_index = (int)i;
                    }
                }
            }
        }
        return best_index;
    }

    void startTrack(const DetectionBox& det) {
        clearTrack();
        resetTrackHistory();
        track_.active = true;
        track_.track_id = next_track_id_++;
        track_.box = det;
    }

    void updateTrack(const DetectionBox& det, int frame_gap) {
        const double prev_cx = centerX(track_.box);
        const double prev_cy = centerY(track_.box);
        const double next_cx = centerX(det);
        const double next_cy = centerY(det);
        const double gap = std::max(1, frame_gap);
        const double meas_vx = (next_cx - prev_cx) / gap;
        const double meas_vy = (next_cy - prev_cy) / gap;
        const double carry = std::clamp(velocity_smoothing_, 0.0, 0.95);
        track_.vx = track_.vx * carry + meas_vx * (1.0 - carry);
        track_.vy = track_.vy * carry + meas_vy * (1.0 - carry);
        track_.box = det;
        track_.missed_frames = 0;
    }

    Parameters buildOutputMetadata(const MetadataEnvelope& env,
                                   const DetectionBox* det,
                                   bool predicted,
                                   int missed_frames) const {
        Parameters md;
        md["version"] = env.version;
        md["coord_space"] = env.coord_space;
        md["model_width"] = env.model_width;
        md["model_height"] = env.model_height;
        if (!env.thresholds.is_null()) {
            md["thresholds"] = env.thresholds;
        }
        md["detections"] = Parameters::array();
        if (det && finiteBox(*det)) {
            Parameters item;
            item["cls"] = det->cls;
            if (det->has_label) {
                item["label"] = det->label;
            }
            item["conf"] = det->conf;
            item["xyxy"] = {det->x1, det->y1, det->x2, det->y2};
            item["track_id"] = track_.track_id;
            item["predicted"] = predicted;
            item["missed_frames"] = missed_frames;
            item["source"] = predicted ? "predicted" : "detected";
            md["detections"].push_back(item);
        }
        md["tracker"] = {
            {"name", "track_ball"},
            {"active", track_.active},
            {"track_id", track_.track_id},
            {"predicted", predicted},
            {"missed_frames", missed_frames}
        };
        return md;
    }

    void maybeLog(const DetectionBox* det, bool predicted, int missed_frames) const {
        if (debug_log_every_n_ <= 0) return;
        if ((frame_counter_ % (uint64_t)debug_log_every_n_) != 0) return;
        if (!det) {
            logstream << "track_ball: frame=" << frame_counter_ << " no active track";
            return;
        }
        logstream << "track_ball: frame=" << frame_counter_
                  << " track_id=" << track_.track_id
                  << " source=" << (predicted ? "predicted" : "detected")
                  << " missed=" << missed_frames
                  << " bbox=[" << det->x1 << "," << det->y1 << "," << det->x2 << "," << det->y2 << "]"
                  << " conf=" << det->conf;
    }

public:
    using NodeSISO::NodeSISO;

    void resetInput() override {
        clearTrack();
        clearHistory();
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm) return;

        if (isEofMarker(frm)) {
            clearTrack();
            clearHistory();
            this->sink_->put(frm);
            return;
        }

        ++frame_counter_;

        MetadataEnvelope env;
        std::vector<DetectionBox> dets;
        (void)parseDetections(frm, env, dets);

        DetectionBox output_det;
        DetectionBox* output_ptr = nullptr;
        bool predicted = false;
        int missed_frames = 0;

        if (!dets.empty()) {
            if (!track_.active) {
                const int match_index = chooseMovingAcquisition(dets);
                if (match_index >= 0) {
                    startTrack(dets[(size_t)match_index]);
                    output_det = track_.box;
                    output_ptr = &output_det;
                }
            } else {
                const DetectionBox reference = predictBox(env);
                const int frame_gap = emit_predicted_ ? 1 : (track_.missed_frames + 1);
                const double frame_span = std::max(env.model_width, env.model_height);
                const int match_index = chooseBestMatch(dets, reference, frame_span, max_center_distance_, frame_gap);
                if (match_index >= 0) {
                    updateTrack(dets[(size_t)match_index], frame_gap);
                    output_det = track_.box;
                    output_ptr = &output_det;
                } else if (emit_predicted_ && track_.missed_frames < max_missed_frames_) {
                    track_.box = predictBox(env);
                    track_.vx *= prediction_decay_;
                    track_.vy *= prediction_decay_;
                    ++track_.missed_frames;
                    output_det = track_.box;
                    output_ptr = &output_det;
                    predicted = true;
                    missed_frames = track_.missed_frames;
                } else {
                    clearTrack();
                }
            }
        } else if (track_.active && emit_predicted_ && track_.missed_frames < max_missed_frames_) {
            track_.box = predictBox(env);
            track_.vx *= prediction_decay_;
            track_.vy *= prediction_decay_;
            ++track_.missed_frames;
            output_det = track_.box;
            output_ptr = &output_det;
            predicted = true;
            missed_frames = track_.missed_frames;
        } else if (track_.active && !emit_predicted_) {
            ++track_.missed_frames;
            if (track_.missed_frames > max_missed_frames_) {
                clearTrack();
            }
        } else {
            clearTrack();
        }

        if (output_ptr) {
            missed_frames = track_.missed_frames;
            pushTrackHistory(*output_ptr, predicted);
        }
        Parameters md = buildOutputMetadata(env, output_ptr, predicted, missed_frames);
        const std::string serialized = md.dump();
        av_dict_set(&frm.raw()->metadata, metadata_key_out_.c_str(), serialized.c_str(), 0);
        maybeLog(output_ptr, predicted, missed_frames);
        prev_prev_dets_ = prev_dets_;
        prev_dets_ = dets;
        this->sink_->put(frm);
    }

    static std::shared_ptr<TrackBall> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<TrackBall>(edges, params);
        if (params.count("metadata_key_in")) r->metadata_key_in_ = params["metadata_key_in"].get<std::string>();
        if (params.count("metadata_key_out")) r->metadata_key_out_ = params["metadata_key_out"].get<std::string>();
        if (params.count("target_label")) r->target_label_ = params["target_label"].get<std::string>();
        if (params.count("target_class")) r->target_class_ = params["target_class"];
        if (params.count("target_labels")) {
            if (!params["target_labels"].is_array()) {
                throw Error("track_ball: target_labels must be a string array");
            }
            const std::list<std::string> labels = jsonToStringList(params["target_labels"]);
            r->target_labels_.assign(labels.begin(), labels.end());
        }
        if (params.count("target_classes")) {
            if (!params["target_classes"].is_array()) {
                throw Error("track_ball: target_classes must be an integer array");
            }
            for (const auto& item : params["target_classes"]) {
                if (!item.is_number_integer()) {
                    throw Error("track_ball: target_classes must be an integer array");
                }
                r->target_classes_.push_back(item.get<int>());
            }
        }
        if (params.count("min_conf")) r->min_conf_ = params["min_conf"];
        if (params.count("max_missed_frames")) r->max_missed_frames_ = params["max_missed_frames"];
        if (params.count("max_center_distance")) r->max_center_distance_ = params["max_center_distance"];
        if (params.count("min_iou_match")) r->min_iou_match_ = params["min_iou_match"];
        if (params.count("match_min_motion")) r->match_min_motion_ = params["match_min_motion"];
        if (params.count("match_max_motion")) r->match_max_motion_ = params["match_max_motion"];
        if (params.count("match_min_cosine_similarity")) r->match_min_cosine_similarity_ = params["match_min_cosine_similarity"];
        if (params.count("match_max_prediction_error")) r->match_max_prediction_error_ = params["match_max_prediction_error"];
        if (params.count("match_max_velocity_delta")) r->match_max_velocity_delta_ = params["match_max_velocity_delta"];
        if (params.count("slow_match_max_prediction_error")) r->slow_match_max_prediction_error_ = params["slow_match_max_prediction_error"];
        if (params.count("slow_match_min_overlap")) r->slow_match_min_overlap_ = params["slow_match_min_overlap"];
        if (params.count("huge_jump_frame_fraction")) r->huge_jump_frame_fraction_ = params["huge_jump_frame_fraction"];
        if (params.count("huge_jump_history_window")) r->huge_jump_history_window_ = params["huge_jump_history_window"];
        if (params.count("history_size")) r->history_size_ = params["history_size"];
        if (params.count("history_motion_window")) r->history_motion_window_ = params["history_motion_window"];
        if (params.count("history_match_min_cosine_similarity")) r->history_match_min_cosine_similarity_ = params["history_match_min_cosine_similarity"];
        if (params.count("history_max_motion_scale")) r->history_max_motion_scale_ = params["history_max_motion_scale"];
        if (params.count("history_max_motion_slack")) r->history_max_motion_slack_ = params["history_max_motion_slack"];
        if (params.count("acquisition_min_motion")) r->acquisition_min_motion_ = params["acquisition_min_motion"];
        if (params.count("acquisition_max_match_distance")) r->acquisition_max_match_distance_ = params["acquisition_max_match_distance"];
        if (params.count("acquisition_min_cosine_similarity")) r->acquisition_min_cosine_similarity_ = params["acquisition_min_cosine_similarity"];
        if (params.count("emit_predicted")) r->emit_predicted_ = params["emit_predicted"];
        if (params.count("prediction_decay")) r->prediction_decay_ = params["prediction_decay"];
        if (params.count("velocity_smoothing")) r->velocity_smoothing_ = params["velocity_smoothing"];
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"];
        return r;
    }
};

DECLNODE(track_ball, TrackBall);
