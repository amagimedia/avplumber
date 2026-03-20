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

struct TrackSample {
    DetectionBox box;
    bool predicted = false;
};

struct HistoryStats {
    bool have_velocity = false;
    bool have_acceleration = false;
    double avg_area = 0.0;
    double last_vx = 0.0;
    double last_vy = 0.0;
    double prev_vx = 0.0;
    double prev_vy = 0.0;
    double avg_step = 0.0;
    double max_step = 0.0;
};

struct Hypothesis {
    int track_id = 0;
    DetectionBox box;
    double vx = 0.0;
    double vy = 0.0;
    double ax = 0.0;
    double ay = 0.0;
    int missed_frames = 0;
    int age = 0;
    int hits = 0;
    double last_conf = 0.0;
    double cumulative_score = 0.0;
    std::vector<TrackSample> history;
};

struct TrackOutput {
    DetectionBox box;
    int track_id = 0;
    bool predicted = false;
    int missed_frames = 0;
    int hits = 0;
    int age = 0;
    double score = -std::numeric_limits<double>::infinity();
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

static double boxArea(const DetectionBox& box) {
    return std::max(0.0, boxWidth(box)) * std::max(0.0, boxHeight(box));
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
    const double uni = boxArea(a) + boxArea(b) - inter;
    return uni > 0.0 ? inter / uni : 0.0;
}

static double centerDistance(const DetectionBox& a, const DetectionBox& b) {
    const double dx = centerX(a) - centerX(b);
    const double dy = centerY(a) - centerY(b);
    return std::sqrt(dx * dx + dy * dy);
}

static double velocityMagnitude(double vx, double vy) {
    return std::sqrt(vx * vx + vy * vy);
}
}

class BallTracker2 : public NodeSISO<av::VideoFrame, av::VideoFrame>, public IInputReset {
private:
    std::string metadata_key_in_ = "yolo_detections_v1";
    std::string metadata_key_out_ = "ball_track_v2";
    std::string target_label_ = "sports ball";
    int target_class_ = -1;
    std::vector<std::string> target_labels_;
    std::vector<int> target_classes_;
    double min_conf_ = 0.10;
    int history_size_ = 120;
    int history_motion_window_ = 12;
    int max_missed_frames_ = 8;
    double max_jump_frame_fraction_ = 0.25;
    double match_max_center_distance_ = 140.0;
    double min_iou_match_ = 0.01;
    double max_acceleration_ = 28.0;
    double max_jerk_ = 28.0;
    double slow_mode_max_prediction_error_ = 12.0;
    double min_track_quality_margin_ = 2.0;
    double max_output_jump_frame_fraction_ = 0.12;
    double output_switch_margin_ = 8.0;
    int min_switch_hits_ = 4;
    double prediction_decay_ = 0.92;
    double velocity_smoothing_ = 0.60;
    int min_confirmed_hits_ = 2;
    int min_track_age_ = 2;
    int max_hypotheses_ = 6;
    int debug_log_every_n_ = 0;
    uint64_t frame_counter_ = 0;
    int next_track_id_ = 1;
    int selected_track_id_ = 0;
    DetectionBox last_output_box_;
    bool last_output_valid_ = false;
    std::vector<Hypothesis> hypotheses_;

    void clearState() {
        hypotheses_.clear();
        selected_track_id_ = 0;
        last_output_box_ = DetectionBox{};
        last_output_valid_ = false;
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

    bool parseDetections(const av::VideoFrame& frm,
                         MetadataEnvelope& env_out,
                         std::vector<DetectionBox>& dets_out) const {
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
            std::sort(dets_out.begin(), dets_out.end(), [](const DetectionBox& a, const DetectionBox& b) {
                return a.conf > b.conf;
            });
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    void trimHistory(std::vector<TrackSample>& history) const {
        const int keep = std::max(1, history_size_);
        if ((int)history.size() > keep) {
            history.erase(history.begin(), history.begin() + ((int)history.size() - keep));
        }
    }

    HistoryStats computeHistoryStats(const Hypothesis& hypothesis) const {
        HistoryStats stats;
        if (hypothesis.history.empty()) {
            return stats;
        }

        const size_t available = hypothesis.history.size();
        const size_t take = (size_t)std::max(1, history_motion_window_);
        const size_t begin = available > take ? available - take : 0;

        for (size_t i = begin; i < available; ++i) {
            stats.avg_area += boxArea(hypothesis.history[i].box);
        }
        stats.avg_area /= std::max<size_t>(1, available - begin);

        if (available - begin < 2) {
            return stats;
        }

        double total_step = 0.0;
        int step_count = 0;
        double prev_vx = 0.0;
        double prev_vy = 0.0;
        bool have_prev_v = false;
        for (size_t i = begin + 1; i < available; ++i) {
            const DetectionBox& prev = hypothesis.history[i - 1].box;
            const DetectionBox& cur = hypothesis.history[i].box;
            const double vx = centerX(cur) - centerX(prev);
            const double vy = centerY(cur) - centerY(prev);
            const double step = velocityMagnitude(vx, vy);
            total_step += step;
            stats.max_step = std::max(stats.max_step, step);
            ++step_count;
            if (have_prev_v) {
                stats.have_acceleration = true;
                stats.prev_vx = prev_vx;
                stats.prev_vy = prev_vy;
            }
            stats.last_vx = vx;
            stats.last_vy = vy;
            have_prev_v = true;
            prev_vx = vx;
            prev_vy = vy;
        }

        stats.have_velocity = step_count > 0;
        stats.avg_step = step_count > 0 ? total_step / step_count : 0.0;
        return stats;
    }

    DetectionBox predictBox(const Hypothesis& hypothesis, const MetadataEnvelope& env) const {
        DetectionBox predicted = shiftedBox(hypothesis.box, hypothesis.vx, hypothesis.vy);
        predicted.conf = std::max(min_conf_, hypothesis.last_conf * std::pow(prediction_decay_, (double)(hypothesis.missed_frames + 1)));
        predicted.cls = hypothesis.box.cls;
        predicted.label = hypothesis.box.label;
        predicted.has_label = hypothesis.box.has_label;
        return clampBoxToCanvas(predicted, env.model_width, env.model_height);
    }

    Hypothesis makeSeedHypothesis(const DetectionBox& det) {
        Hypothesis hypothesis;
        hypothesis.track_id = next_track_id_++;
        hypothesis.box = det;
        hypothesis.last_conf = det.conf;
        hypothesis.hits = 1;
        hypothesis.age = 1;
        hypothesis.cumulative_score = det.conf * 5.0;
        hypothesis.history.push_back(TrackSample{det, false});
        return hypothesis;
    }

    bool updateWithDetection(Hypothesis& hypothesis,
                             const DetectionBox& det,
                             const MetadataEnvelope& env,
                             double& delta_score) const {
        const int frame_gap = hypothesis.missed_frames + 1;
        const double gap = std::max(1, frame_gap);
        const DetectionBox predicted = predictBox(hypothesis, env);
        const double dist_pred = centerDistance(predicted, det);
        const double overlap = iou(predicted, det);
        if (dist_pred > match_max_center_distance_ && overlap < min_iou_match_) {
            return false;
        }

        const double frame_span = std::max(env.model_width, env.model_height);
        const double max_jump = std::max(1.0, max_jump_frame_fraction_ * frame_span * gap);
        const double step_dist = centerDistance(hypothesis.box, det);
        if (step_dist > max_jump) {
            return false;
        }

        const double meas_vx = (centerX(det) - centerX(hypothesis.box)) / gap;
        const double meas_vy = (centerY(det) - centerY(hypothesis.box)) / gap;
        const double velocity_delta = velocityMagnitude(meas_vx - hypothesis.vx, meas_vy - hypothesis.vy);
        const HistoryStats stats = computeHistoryStats(hypothesis);
        if (stats.have_velocity && velocity_delta > max_acceleration_) {
            return false;
        }

        const double jerk = stats.have_acceleration
            ? velocityMagnitude((meas_vx - stats.last_vx) - (stats.last_vx - stats.prev_vx),
                                (meas_vy - stats.last_vy) - (stats.last_vy - stats.prev_vy))
            : 0.0;
        if (stats.have_acceleration && jerk > max_jerk_) {
            return false;
        }

        const bool slow_mode = velocityMagnitude(meas_vx, meas_vy) < std::max(1.0, stats.avg_step * 0.35);
        if (slow_mode && dist_pred > slow_mode_max_prediction_error_ && overlap < 0.10) {
            return false;
        }

        const double avg_area = stats.avg_area > 0.0 ? stats.avg_area : boxArea(hypothesis.box);
        const double area_ratio = avg_area > 0.0 ? std::max(boxArea(det), avg_area) / std::max(1.0, std::min(boxArea(det), avg_area)) : 1.0;
        if (area_ratio > 4.0) {
            return false;
        }

        const double carry = std::clamp(velocity_smoothing_, 0.0, 0.95);
        const double new_ax = meas_vx - hypothesis.vx;
        const double new_ay = meas_vy - hypothesis.vy;
        hypothesis.vx = hypothesis.vx * carry + meas_vx * (1.0 - carry);
        hypothesis.vy = hypothesis.vy * carry + meas_vy * (1.0 - carry);
        hypothesis.ax = hypothesis.ax * carry + new_ax * (1.0 - carry);
        hypothesis.ay = hypothesis.ay * carry + new_ay * (1.0 - carry);
        hypothesis.box = det;
        hypothesis.last_conf = det.conf;
        hypothesis.missed_frames = 0;
        hypothesis.age += frame_gap;
        hypothesis.hits += 1;
        hypothesis.history.push_back(TrackSample{det, false});
        trimHistory(hypothesis.history);

        delta_score = det.conf * 5.0
                    + overlap * 4.0
                    - dist_pred / std::max(1.0, match_max_center_distance_)
                    - velocity_delta / std::max(1.0, max_acceleration_)
                    - jerk / std::max(1.0, max_jerk_)
                    + (slow_mode ? 1.0 : 0.0)
                    - std::max(0.0, area_ratio - 1.5);
        return true;
    }

    bool updateWithPrediction(Hypothesis& hypothesis,
                              const MetadataEnvelope& env,
                              double& delta_score) const {
        if (hypothesis.missed_frames >= max_missed_frames_) {
            return false;
        }
        DetectionBox predicted = predictBox(hypothesis, env);
        if (!finiteBox(predicted)) {
            return false;
        }
        hypothesis.box = predicted;
        hypothesis.vx *= prediction_decay_;
        hypothesis.vy *= prediction_decay_;
        hypothesis.ax *= prediction_decay_;
        hypothesis.ay *= prediction_decay_;
        hypothesis.last_conf = predicted.conf;
        hypothesis.missed_frames += 1;
        hypothesis.age += 1;
        hypothesis.history.push_back(TrackSample{predicted, true});
        trimHistory(hypothesis.history);
        delta_score = -1.5 - hypothesis.missed_frames * 0.75;
        return true;
    }

    bool isDuplicateHypothesis(const Hypothesis& a, const Hypothesis& b) const {
        if (a.track_id != b.track_id) {
            return false;
        }
        if (!finiteBox(a.box) || !finiteBox(b.box)) {
            return false;
        }
        return centerDistance(a.box, b.box) < 4.0 && iou(a.box, b.box) > 0.80 && a.missed_frames == b.missed_frames;
    }

    void pruneHypotheses(std::vector<Hypothesis>& candidates) const {
        std::sort(candidates.begin(), candidates.end(), [](const Hypothesis& a, const Hypothesis& b) {
            if (a.cumulative_score != b.cumulative_score) return a.cumulative_score > b.cumulative_score;
            if (a.hits != b.hits) return a.hits > b.hits;
            return a.last_conf > b.last_conf;
        });

        std::vector<Hypothesis> kept;
        kept.reserve((size_t)std::max(1, max_hypotheses_));
        for (const Hypothesis& candidate : candidates) {
            bool duplicate = false;
            for (const Hypothesis& existing : kept) {
                if (isDuplicateHypothesis(existing, candidate)) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate) continue;
            kept.push_back(candidate);
            if ((int)kept.size() >= std::max(1, max_hypotheses_)) {
                break;
            }
        }
        candidates.swap(kept);
    }

    TrackOutput chooseOutput(const MetadataEnvelope& env) {
        std::vector<TrackOutput> candidates;
        candidates.reserve(hypotheses_.size());
        for (const Hypothesis& hypothesis : hypotheses_) {
            if (hypothesis.hits < std::max(1, min_confirmed_hits_)) continue;
            if (hypothesis.age < std::max(1, min_track_age_)) continue;
            DetectionBox box = hypothesis.box;
            bool predicted = hypothesis.missed_frames > 0;
            if (!finiteBox(box)) {
                box = predictBox(hypothesis, env);
                predicted = true;
            }
            if (!finiteBox(box)) continue;
            const double score = hypothesis.cumulative_score
                               + hypothesis.hits * 4.0
                               + hypothesis.last_conf * 6.0
                               - hypothesis.missed_frames * 10.0;
            TrackOutput candidate;
            candidate.box = box;
            candidate.track_id = hypothesis.track_id;
            candidate.predicted = predicted;
            candidate.missed_frames = hypothesis.missed_frames;
            candidate.hits = hypothesis.hits;
            candidate.age = hypothesis.age;
            candidate.score = score;
            candidates.push_back(candidate);
        }

        if (candidates.empty()) {
            selected_track_id_ = 0;
            last_output_valid_ = false;
            return TrackOutput{};
        }

        std::sort(candidates.begin(), candidates.end(), [](const TrackOutput& a, const TrackOutput& b) {
            return a.score > b.score;
        });

        TrackOutput best = candidates.front();
        TrackOutput second;
        if (candidates.size() > 1) {
            second = candidates[1];
        }

        TrackOutput current;
        bool have_current = false;
        for (const TrackOutput& candidate : candidates) {
            if (candidate.track_id == selected_track_id_) {
                current = candidate;
                have_current = true;
                break;
            }
        }

        const double frame_span = std::max(env.model_width, env.model_height);
        const double max_output_jump = std::max(1.0, frame_span * max_output_jump_frame_fraction_);

        if (best.track_id > 0 && second.track_id > 0
            && (best.score - second.score) < min_track_quality_margin_
            && have_current) {
            best = current;
        }

        if (have_current && best.track_id != current.track_id) {
            const double current_to_best = centerDistance(current.box, best.box);
            const bool huge_switch = current_to_best > max_output_jump;
            const bool best_is_mature = best.hits >= std::max(min_switch_hits_, min_confirmed_hits_);
            const bool decisive_margin = (best.score - current.score) >= output_switch_margin_;

            if (huge_switch && (!best_is_mature || !decisive_margin)) {
                best = current;
            } else if (!huge_switch && (best.score - current.score) < min_track_quality_margin_) {
                best = current;
            }
        }

        if (last_output_valid_ && best.track_id > 0 && best.track_id != selected_track_id_) {
            const double output_jump = centerDistance(last_output_box_, best.box);
            const bool huge_output_jump = output_jump > max_output_jump;
            const bool best_is_mature = best.hits >= std::max(min_switch_hits_, min_confirmed_hits_);
            if (huge_output_jump && !best_is_mature) {
                if (have_current) {
                    best = current;
                }
            }
        }

        selected_track_id_ = best.track_id;
        if (best.track_id > 0 && finiteBox(best.box)) {
            last_output_box_ = best.box;
            last_output_valid_ = true;
        }
        return best;
    }

    Parameters buildOutputMetadata(const MetadataEnvelope& env, const TrackOutput& output) const {
        Parameters md;
        md["version"] = env.version;
        md["coord_space"] = env.coord_space;
        md["model_width"] = env.model_width;
        md["model_height"] = env.model_height;
        if (!env.thresholds.is_null()) {
            md["thresholds"] = env.thresholds;
        }
        md["detections"] = Parameters::array();
        if (output.track_id > 0 && finiteBox(output.box)) {
            Parameters item;
            item["cls"] = output.box.cls;
            if (output.box.has_label) {
                item["label"] = output.box.label;
            }
            item["conf"] = output.box.conf;
            item["xyxy"] = {output.box.x1, output.box.y1, output.box.x2, output.box.y2};
            item["track_id"] = output.track_id;
            item["predicted"] = output.predicted;
            item["missed_frames"] = output.missed_frames;
            item["source"] = output.predicted ? "predicted" : "detected";
            item["hits"] = output.hits;
            item["age"] = output.age;
            md["detections"].push_back(item);
        }
        md["tracker"] = {
            {"name", "ball_tracker2"},
            {"active_hypotheses", (int)hypotheses_.size()},
            {"selected_track_id", output.track_id},
            {"predicted", output.predicted},
            {"missed_frames", output.missed_frames}
        };
        return md;
    }

    void maybeLog(const TrackOutput& output) const {
        if (debug_log_every_n_ <= 0) return;
        if ((frame_counter_ % (uint64_t)debug_log_every_n_) != 0) return;
        if (output.track_id <= 0) {
            logstream << "ball_tracker2: frame=" << frame_counter_ << " no selected track";
            return;
        }
        logstream << "ball_tracker2: frame=" << frame_counter_
                  << " track_id=" << output.track_id
                  << " source=" << (output.predicted ? "predicted" : "detected")
                  << " missed=" << output.missed_frames
                  << " hits=" << output.hits
                  << " score=" << output.score
                  << " bbox=[" << output.box.x1 << "," << output.box.y1 << "," << output.box.x2 << "," << output.box.y2 << "]"
                  << " conf=" << output.box.conf;
    }

public:
    using NodeSISO::NodeSISO;

    void resetInput() override {
        clearState();
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm) return;

        if (isEofMarker(frm)) {
            clearState();
            this->sink_->put(frm);
            return;
        }

        ++frame_counter_;

        MetadataEnvelope env;
        std::vector<DetectionBox> dets;
        (void)parseDetections(frm, env, dets);

        std::vector<Hypothesis> next_hypotheses;
        next_hypotheses.reserve((size_t)(std::max(1, max_hypotheses_) * (2 + dets.size())));

        for (const Hypothesis& hypothesis : hypotheses_) {
            bool matched_any = false;
            for (const DetectionBox& det : dets) {
                Hypothesis candidate = hypothesis;
                double delta_score = 0.0;
                if (!updateWithDetection(candidate, det, env, delta_score)) {
                    continue;
                }
                candidate.cumulative_score += delta_score;
                next_hypotheses.push_back(std::move(candidate));
                matched_any = true;
            }

            if (!matched_any || hypothesis.missed_frames < max_missed_frames_) {
                Hypothesis candidate = hypothesis;
                double delta_score = 0.0;
                if (updateWithPrediction(candidate, env, delta_score)) {
                    candidate.cumulative_score += delta_score;
                    next_hypotheses.push_back(std::move(candidate));
                }
            }
        }

        for (const DetectionBox& det : dets) {
            Hypothesis seed = makeSeedHypothesis(det);
            next_hypotheses.push_back(std::move(seed));
        }

        if (next_hypotheses.empty()) {
            hypotheses_.clear();
        } else {
            pruneHypotheses(next_hypotheses);
            hypotheses_.swap(next_hypotheses);
        }

        const TrackOutput output = chooseOutput(env);
        const Parameters md = buildOutputMetadata(env, output);
        const std::string serialized = md.dump();
        av_dict_set(&frm.raw()->metadata, metadata_key_out_.c_str(), serialized.c_str(), 0);
        maybeLog(output);
        this->sink_->put(frm);
    }

    static std::shared_ptr<BallTracker2> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<BallTracker2>(edges, params);
        if (params.count("metadata_key_in")) r->metadata_key_in_ = params["metadata_key_in"].get<std::string>();
        if (params.count("metadata_key_out")) r->metadata_key_out_ = params["metadata_key_out"].get<std::string>();
        if (params.count("target_label")) r->target_label_ = params["target_label"].get<std::string>();
        if (params.count("target_class")) r->target_class_ = params["target_class"];
        if (params.count("target_labels")) {
            if (!params["target_labels"].is_array()) {
                throw Error("ball_tracker2: target_labels must be a string array");
            }
            const std::list<std::string> labels = jsonToStringList(params["target_labels"]);
            r->target_labels_.assign(labels.begin(), labels.end());
        }
        if (params.count("target_classes")) {
            if (!params["target_classes"].is_array()) {
                throw Error("ball_tracker2: target_classes must be an integer array");
            }
            for (const auto& item : params["target_classes"]) {
                if (!item.is_number_integer()) {
                    throw Error("ball_tracker2: target_classes must be an integer array");
                }
                r->target_classes_.push_back(item.get<int>());
            }
        }
        if (params.count("min_conf")) r->min_conf_ = params["min_conf"];
        if (params.count("history_size")) r->history_size_ = params["history_size"];
        if (params.count("history_motion_window")) r->history_motion_window_ = params["history_motion_window"];
        if (params.count("max_missed_frames")) r->max_missed_frames_ = params["max_missed_frames"];
        if (params.count("max_jump_frame_fraction")) r->max_jump_frame_fraction_ = params["max_jump_frame_fraction"];
        if (params.count("match_max_center_distance")) r->match_max_center_distance_ = params["match_max_center_distance"];
        if (params.count("min_iou_match")) r->min_iou_match_ = params["min_iou_match"];
        if (params.count("max_acceleration")) r->max_acceleration_ = params["max_acceleration"];
        if (params.count("max_jerk")) r->max_jerk_ = params["max_jerk"];
        if (params.count("slow_mode_max_prediction_error")) r->slow_mode_max_prediction_error_ = params["slow_mode_max_prediction_error"];
        if (params.count("min_track_quality_margin")) r->min_track_quality_margin_ = params["min_track_quality_margin"];
        if (params.count("max_output_jump_frame_fraction")) r->max_output_jump_frame_fraction_ = params["max_output_jump_frame_fraction"];
        if (params.count("output_switch_margin")) r->output_switch_margin_ = params["output_switch_margin"];
        if (params.count("min_switch_hits")) r->min_switch_hits_ = params["min_switch_hits"];
        if (params.count("prediction_decay")) r->prediction_decay_ = params["prediction_decay"];
        if (params.count("velocity_smoothing")) r->velocity_smoothing_ = params["velocity_smoothing"];
        if (params.count("min_confirmed_hits")) r->min_confirmed_hits_ = params["min_confirmed_hits"];
        if (params.count("min_track_age")) r->min_track_age_ = params["min_track_age"];
        if (params.count("max_hypotheses")) r->max_hypotheses_ = params["max_hypotheses"];
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"];
        return r;
    }
};

DECLNODE(ball_tracker2, BallTracker2);
