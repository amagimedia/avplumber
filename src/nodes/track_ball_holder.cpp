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
enum class TrackMode {
    None,
    Holder,
    Ball
};

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

struct PairCandidate {
    DetectionBox person;
    DetectionBox ball;
    DetectionBox combined;
    double pair_score = 0.0;
    double output_conf = 0.0;
    std::string hand_side = "unknown";
};

struct TrackState {
    bool active = false;
    int track_id = 0;
    TrackMode mode = TrackMode::None;
    DetectionBox combined;
    DetectionBox person;
    DetectionBox ball;
    double vx = 0.0;
    double vy = 0.0;
    int missed_frames = 0;
    double pair_score = 0.0;
    double output_conf = 0.0;
    std::string hand_side = "unknown";
};

static double clampUnit(double value) {
    return std::max(0.0, std::min(value, 1.0));
}

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

static DetectionBox unionBoxes(const DetectionBox& a,
                               const DetectionBox& b,
                               double pad_x_ratio,
                               double pad_y_ratio,
                               double canvas_w,
                               double canvas_h) {
    DetectionBox out;
    out.x1 = std::min(a.x1, b.x1);
    out.y1 = std::min(a.y1, b.y1);
    out.x2 = std::max(a.x2, b.x2);
    out.y2 = std::max(a.y2, b.y2);
    const double w = std::max(0.0, boxWidth(out));
    const double h = std::max(0.0, boxHeight(out));
    out.x1 -= w * pad_x_ratio;
    out.x2 += w * pad_x_ratio;
    out.y1 -= h * pad_y_ratio;
    out.y2 += h * pad_y_ratio;
    return clampBoxToCanvas(out, canvas_w, canvas_h);
}

static const char* trackModeName(TrackMode mode) {
    switch (mode) {
        case TrackMode::Holder: return "holder";
        case TrackMode::Ball: return "ball";
        default: return "none";
    }
}
}

class TrackBallHolder : public NodeSISO<av::VideoFrame, av::VideoFrame>, public IInputReset {
private:
    std::string metadata_key_in_ = "yolo_detections_v1";
    std::string metadata_key_out_ = "ball_holder_track_v1";
    std::string ball_label_ = "sports ball";
    std::string person_label_ = "person";
    std::string output_label_ = "ball_holder";
    int ball_class_ = -1;
    int person_class_ = -1;
    double min_ball_conf_ = 0.10;
    double min_person_conf_ = 0.25;
    double max_hand_distance_px_ = 180.0;
    double hand_y_ratio_ = 0.35;
    double side_outset_ratio_ = 0.05;
    double upper_y_min_ratio_ = 0.05;
    double upper_y_max_ratio_ = 0.72;
    double expanded_person_margin_x_ratio_ = 0.10;
    double expanded_person_margin_y_ratio_ = 0.05;
    double combined_padding_x_ratio_ = 0.04;
    double combined_padding_y_ratio_ = 0.04;
    int max_missed_frames_ = 4;
    double max_center_distance_ = 220.0;
    double min_iou_match_ = 0.0;
    bool emit_predicted_ = true;
    double prediction_decay_ = 0.85;
    double velocity_smoothing_ = 0.60;
    int debug_log_every_n_ = 0;
    uint64_t frame_counter_ = 0;
    int next_track_id_ = 1;
    TrackState track_;

    void clearTrack() {
        track_ = TrackState{};
    }

    bool matchesTarget(const DetectionBox& det, const std::string& label, int cls) const {
        bool class_match = false;
        bool label_match = false;
        if (cls >= 0 && det.cls == cls) {
            class_match = true;
        }
        if (!label.empty() && det.has_label && det.label == label) {
            label_match = true;
        }
        if (cls < 0 && label.empty()) {
            return true;
        }
        return class_match || label_match;
    }

    bool parseDetections(const av::VideoFrame& frm,
                         MetadataEnvelope& env_out,
                         std::vector<DetectionBox>& person_out,
                         std::vector<DetectionBox>& ball_out) const {
        env_out = MetadataEnvelope{};
        env_out.model_width = frm.width();
        env_out.model_height = frm.height();
        person_out.clear();
        ball_out.clear();

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

                if (!finiteBox(det)) continue;
                if (matchesTarget(det, person_label_, person_class_) && det.conf >= min_person_conf_) {
                    person_out.push_back(det);
                }
                if (matchesTarget(det, ball_label_, ball_class_) && det.conf >= min_ball_conf_) {
                    ball_out.push_back(det);
                }
            }
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    DetectionBox predictedCombined(const MetadataEnvelope& env) const {
        DetectionBox predicted = shiftedBox(track_.combined, track_.vx, track_.vy);
        predicted.conf = std::max(min_ball_conf_, track_.output_conf * std::pow(prediction_decay_, (double)(track_.missed_frames + 1)));
        predicted.label = (track_.mode == TrackMode::Holder) ? output_label_ : ball_label_;
        predicted.has_label = !predicted.label.empty();
        predicted.cls = (track_.mode == TrackMode::Holder) ? -1 : ball_class_;
        return clampBoxToCanvas(predicted, env.model_width, env.model_height);
    }

    DetectionBox predictedPerson(const MetadataEnvelope& env) const {
        return clampBoxToCanvas(shiftedBox(track_.person, track_.vx, track_.vy), env.model_width, env.model_height);
    }

    DetectionBox predictedBall(const MetadataEnvelope& env) const {
        return clampBoxToCanvas(shiftedBox(track_.ball, track_.vx, track_.vy), env.model_width, env.model_height);
    }

    static double safeNormalizedY(const DetectionBox& person, const DetectionBox& ball) {
        const double ph = boxHeight(person);
        if (ph <= 0.0) return -1.0;
        return (centerY(ball) - person.y1) / ph;
    }

    PairCandidate buildPairCandidate(const DetectionBox& person,
                                     const DetectionBox& ball,
                                     const MetadataEnvelope& env) const {
        PairCandidate candidate;
        candidate.person = person;
        candidate.ball = ball;

        const double pw = std::max(1.0, boxWidth(person));
        const double ph = std::max(1.0, boxHeight(person));
        const double bx = centerX(ball);
        const double by = centerY(ball);

        const double upper_norm_y = safeNormalizedY(person, ball);
        if (upper_norm_y < upper_y_min_ratio_ || upper_norm_y > upper_y_max_ratio_) {
            candidate.pair_score = -1.0;
            return candidate;
        }

        const double hand_y = person.y1 + hand_y_ratio_ * ph;
        const double left_hand_x = person.x1 - side_outset_ratio_ * pw;
        const double right_hand_x = person.x2 + side_outset_ratio_ * pw;
        const double left_dist = std::sqrt((bx - left_hand_x) * (bx - left_hand_x) + (by - hand_y) * (by - hand_y));
        const double right_dist = std::sqrt((bx - right_hand_x) * (bx - right_hand_x) + (by - hand_y) * (by - hand_y));

        double hand_dist = left_dist;
        candidate.hand_side = "left";
        if (right_dist < left_dist) {
            hand_dist = right_dist;
            candidate.hand_side = "right";
        }
        if (hand_dist > max_hand_distance_px_) {
            candidate.pair_score = -1.0;
            return candidate;
        }

        const double margin_x = expanded_person_margin_x_ratio_ * pw;
        const double margin_y = expanded_person_margin_y_ratio_ * ph;
        const bool inside_expanded =
            bx >= (person.x1 - margin_x) && bx <= (person.x2 + margin_x) &&
            by >= (person.y1 - margin_y) && by <= (person.y2 + margin_y);

        const double hand_score = 1.0 - clampUnit(hand_dist / std::max(1.0, max_hand_distance_px_));
        const double upper_mid = (upper_y_min_ratio_ + upper_y_max_ratio_) * 0.5;
        const double upper_half_span = std::max(0.05, (upper_y_max_ratio_ - upper_y_min_ratio_) * 0.5);
        const double upper_score = clampUnit(1.0 - std::abs(upper_norm_y - upper_mid) / upper_half_span);
        const double inside_score = inside_expanded ? 1.0 : 0.0;
        const double conf_score = std::sqrt(std::max(0.0, person.conf) * std::max(0.0, ball.conf));

        candidate.combined = unionBoxes(person, ball,
                                        combined_padding_x_ratio_,
                                        combined_padding_y_ratio_,
                                        env.model_width,
                                        env.model_height);
        candidate.combined.cls = -1;
        candidate.combined.label = output_label_;
        candidate.combined.has_label = !output_label_.empty();

        double temporal_score = 0.0;
        if (track_.active) {
            const DetectionBox reference = track_.missed_frames > 0 ? predictedCombined(env) : track_.combined;
            const double dist_score = 1.0 - clampUnit(centerDistance(reference, candidate.combined) / std::max(1.0, max_center_distance_));
            const double overlap_score = iou(reference, candidate.combined);
            temporal_score = 0.5 * dist_score + 0.5 * overlap_score;
        }

        candidate.pair_score = hand_score * 0.40
                             + upper_score * 0.20
                             + inside_score * 0.15
                             + conf_score * 0.15
                             + temporal_score * 0.10;
        candidate.output_conf = clampUnit(conf_score);
        candidate.combined.conf = candidate.output_conf;
        return candidate;
    }

    std::vector<PairCandidate> buildPairCandidates(const std::vector<DetectionBox>& persons,
                                                   const std::vector<DetectionBox>& balls,
                                                   const MetadataEnvelope& env) const {
        std::vector<PairCandidate> result;
        for (const DetectionBox& person : persons) {
            for (const DetectionBox& ball : balls) {
                PairCandidate candidate = buildPairCandidate(person, ball, env);
                if (candidate.pair_score >= 0.0 && finiteBox(candidate.combined)) {
                    result.push_back(candidate);
                }
            }
        }
        return result;
    }

    DetectionBox bestBallByScore(const std::vector<DetectionBox>& balls,
                                 const DetectionBox* reference_box) const {
        DetectionBox best;
        double best_score = -std::numeric_limits<double>::infinity();
        for (const DetectionBox& ball : balls) {
            double score = ball.conf;
            if (reference_box && finiteBox(*reference_box)) {
                const double dist_score = 1.0 - clampUnit(centerDistance(*reference_box, ball) / std::max(1.0, max_center_distance_));
                const double overlap_score = iou(*reference_box, ball);
                score += dist_score * 0.5 + overlap_score * 0.5;
            }
            if (score > best_score) {
                best_score = score;
                best = ball;
            }
        }
        return best;
    }

    int chooseBestBallMatch(const std::vector<DetectionBox>& balls, const DetectionBox& reference_box) const {
        int best_index = -1;
        double best_score = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < balls.size(); ++i) {
            const DetectionBox& ball = balls[i];
            const double dist = centerDistance(reference_box, ball);
            const double overlap = iou(reference_box, ball);
            if (dist > max_center_distance_ && overlap < min_iou_match_) {
                continue;
            }
            const double score = ball.conf * 3.0 + overlap * 2.0 - dist / std::max(1.0, max_center_distance_);
            if (score > best_score) {
                best_score = score;
                best_index = (int)i;
            }
        }
        return best_index;
    }

    const PairCandidate* bestPairByScore(const std::vector<PairCandidate>& candidates) const {
        if (candidates.empty()) return nullptr;
        return &(*std::max_element(candidates.begin(), candidates.end(), [](const PairCandidate& a, const PairCandidate& b) {
            return a.pair_score < b.pair_score;
        }));
    }

    int chooseBestMatch(const std::vector<PairCandidate>& candidates, const DetectionBox& reference_box) const {
        int best_index = -1;
        double best_score = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < candidates.size(); ++i) {
            const PairCandidate& candidate = candidates[i];
            const double dist = centerDistance(reference_box, candidate.combined);
            const double overlap = iou(reference_box, candidate.combined);
            if (dist > max_center_distance_ && overlap < min_iou_match_) {
                continue;
            }
            const double score = candidate.pair_score * 3.0
                               + candidate.output_conf * 2.0
                               + overlap * 2.0
                               - dist / std::max(1.0, max_center_distance_);
            if (score > best_score) {
                best_score = score;
                best_index = (int)i;
            }
        }
        return best_index;
    }

    void startTrack(const PairCandidate& candidate) {
        clearTrack();
        track_.active = true;
        track_.track_id = next_track_id_++;
        track_.mode = TrackMode::Holder;
        track_.combined = candidate.combined;
        track_.person = candidate.person;
        track_.ball = candidate.ball;
        track_.pair_score = candidate.pair_score;
        track_.output_conf = candidate.output_conf;
        track_.hand_side = candidate.hand_side;
    }

    void updateTrack(const PairCandidate& candidate, int frame_gap) {
        const double prev_cx = centerX(track_.combined);
        const double prev_cy = centerY(track_.combined);
        const double next_cx = centerX(candidate.combined);
        const double next_cy = centerY(candidate.combined);
        const double gap = std::max(1, frame_gap);
        const double meas_vx = (next_cx - prev_cx) / gap;
        const double meas_vy = (next_cy - prev_cy) / gap;
        const double carry = std::clamp(velocity_smoothing_, 0.0, 0.95);
        track_.vx = track_.vx * carry + meas_vx * (1.0 - carry);
        track_.vy = track_.vy * carry + meas_vy * (1.0 - carry);
        track_.mode = TrackMode::Holder;
        track_.combined = candidate.combined;
        track_.person = candidate.person;
        track_.ball = candidate.ball;
        track_.pair_score = candidate.pair_score;
        track_.output_conf = candidate.output_conf;
        track_.hand_side = candidate.hand_side;
        track_.missed_frames = 0;
    }

    void startBallTrack(const DetectionBox& ball) {
        clearTrack();
        track_.active = true;
        track_.track_id = next_track_id_++;
        track_.mode = TrackMode::Ball;
        track_.combined = ball;
        track_.combined.label = ball_label_;
        track_.combined.has_label = !ball_label_.empty();
        track_.combined.cls = ball.cls;
        track_.person = DetectionBox{};
        track_.ball = ball;
        track_.pair_score = ball.conf;
        track_.output_conf = ball.conf;
        track_.hand_side = "none";
    }

    void updateBallTrack(const DetectionBox& ball, int frame_gap) {
        const double prev_cx = centerX(track_.ball);
        const double prev_cy = centerY(track_.ball);
        const double next_cx = centerX(ball);
        const double next_cy = centerY(ball);
        const double gap = std::max(1, frame_gap);
        const double meas_vx = (next_cx - prev_cx) / gap;
        const double meas_vy = (next_cy - prev_cy) / gap;
        const double carry = std::clamp(velocity_smoothing_, 0.0, 0.95);
        track_.vx = track_.vx * carry + meas_vx * (1.0 - carry);
        track_.vy = track_.vy * carry + meas_vy * (1.0 - carry);
        track_.mode = TrackMode::Ball;
        track_.ball = ball;
        track_.combined = ball;
        track_.combined.label = ball_label_;
        track_.combined.has_label = !ball_label_.empty();
        track_.combined.cls = ball.cls;
        track_.person = DetectionBox{};
        track_.pair_score = ball.conf;
        track_.output_conf = ball.conf;
        track_.hand_side = "none";
        track_.missed_frames = 0;
    }

    void predictTrack(const MetadataEnvelope& env) {
        track_.combined = predictedCombined(env);
        track_.person = (track_.mode == TrackMode::Holder) ? predictedPerson(env) : DetectionBox{};
        track_.ball = predictedBall(env);
        track_.vx *= prediction_decay_;
        track_.vy *= prediction_decay_;
        track_.output_conf = std::max(min_ball_conf_, track_.output_conf * prediction_decay_);
        ++track_.missed_frames;
    }

    Parameters buildOutputMetadata(const MetadataEnvelope& env, bool predicted, int missed_frames) const {
        Parameters md;
        md["version"] = env.version;
        md["coord_space"] = env.coord_space;
        md["model_width"] = env.model_width;
        md["model_height"] = env.model_height;
        if (!env.thresholds.is_null()) {
            md["thresholds"] = env.thresholds;
        }
        md["detections"] = Parameters::array();
        if (track_.active && finiteBox(track_.combined)) {
            Parameters item;
            item["cls"] = track_.combined.cls;
            item["label"] = (track_.mode == TrackMode::Holder) ? output_label_ : ball_label_;
            item["conf"] = track_.output_conf;
            item["xyxy"] = {track_.combined.x1, track_.combined.y1, track_.combined.x2, track_.combined.y2};
            item["track_id"] = track_.track_id;
            item["predicted"] = predicted;
            item["missed_frames"] = missed_frames;
            item["source"] = predicted ? "predicted" : "detected";
            item["pair_score"] = track_.pair_score;
            item["hand_side"] = track_.hand_side;
            item["tracking_mode"] = trackModeName(track_.mode);
            if (track_.mode == TrackMode::Holder && finiteBox(track_.person)) {
                item["person_xyxy"] = {track_.person.x1, track_.person.y1, track_.person.x2, track_.person.y2};
                item["person_conf"] = track_.person.conf;
            }
            item["ball_xyxy"] = {track_.ball.x1, track_.ball.y1, track_.ball.x2, track_.ball.y2};
            item["ball_conf"] = track_.ball.conf;
            md["detections"].push_back(item);
        }
        md["tracker"] = {
            {"name", "track_ball_holder"},
            {"active", track_.active},
            {"mode", trackModeName(track_.mode)},
            {"track_id", track_.track_id},
            {"predicted", predicted},
            {"missed_frames", missed_frames}
        };
        return md;
    }

    void maybeLog(bool predicted, int missed_frames) const {
        if (debug_log_every_n_ <= 0) return;
        if ((frame_counter_ % (uint64_t)debug_log_every_n_) != 0) return;
        if (!track_.active) {
            logstream << "track_ball_holder: frame=" << frame_counter_ << " no active pair";
            return;
        }
        logstream << "track_ball_holder: frame=" << frame_counter_
                  << " track_id=" << track_.track_id
                  << " mode=" << trackModeName(track_.mode)
                  << " source=" << (predicted ? "predicted" : "detected")
                  << " missed=" << missed_frames
                  << " hand=" << track_.hand_side
                  << " pair_score=" << track_.pair_score
                  << " bbox=[" << track_.combined.x1 << "," << track_.combined.y1
                  << "," << track_.combined.x2 << "," << track_.combined.y2 << "]";
    }

public:
    using NodeSISO::NodeSISO;

    void resetInput() override {
        clearTrack();
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm) return;

        if (isEofMarker(frm)) {
            clearTrack();
            this->sink_->put(frm);
            return;
        }

        ++frame_counter_;

        MetadataEnvelope env;
        std::vector<DetectionBox> persons;
        std::vector<DetectionBox> balls;
        (void)parseDetections(frm, env, persons, balls);
        const std::vector<PairCandidate> candidates = buildPairCandidates(persons, balls, env);

        bool predicted = false;
        int missed_frames = 0;

        if (!candidates.empty()) {
            if (!track_.active) {
                const PairCandidate* best = bestPairByScore(candidates);
                if (best) {
                    startTrack(*best);
                }
            } else {
                const DetectionBox reference = (track_.missed_frames > 0 && track_.mode == TrackMode::Holder) ? predictedCombined(env) : track_.combined;
                const int match_index = chooseBestMatch(candidates, reference);
                if (match_index >= 0) {
                    updateTrack(candidates[(size_t)match_index], track_.missed_frames + 1);
                } else {
                    const DetectionBox reference_ball = finiteBox(track_.ball) ? ((track_.missed_frames > 0) ? predictedBall(env) : track_.ball) : reference;
                    const int ball_index = balls.empty() ? -1 : chooseBestBallMatch(balls, reference_ball);
                    if (ball_index >= 0) {
                        updateBallTrack(balls[(size_t)ball_index], track_.missed_frames + 1);
                    } else if (emit_predicted_ && track_.missed_frames < max_missed_frames_) {
                        predictTrack(env);
                        predicted = true;
                    } else {
                        clearTrack();
                    }
                }
            }
        } else if (!balls.empty()) {
            if (!track_.active) {
                startBallTrack(bestBallByScore(balls, nullptr));
            } else {
                const DetectionBox reference = finiteBox(track_.ball)
                    ? ((track_.missed_frames > 0) ? predictedBall(env) : track_.ball)
                    : ((track_.missed_frames > 0) ? predictedCombined(env) : track_.combined);
                const int ball_index = chooseBestBallMatch(balls, reference);
                if (ball_index >= 0) {
                    updateBallTrack(balls[(size_t)ball_index], track_.missed_frames + 1);
                } else if (emit_predicted_ && track_.missed_frames < max_missed_frames_) {
                    predictTrack(env);
                    predicted = true;
                } else {
                    clearTrack();
                }
            }
        } else if (track_.active && emit_predicted_ && track_.missed_frames < max_missed_frames_) {
            predictTrack(env);
            predicted = true;
        } else if (track_.active && !emit_predicted_) {
            ++track_.missed_frames;
            if (track_.missed_frames > max_missed_frames_) {
                clearTrack();
            }
        } else {
            clearTrack();
        }

        if (track_.active) {
            missed_frames = track_.missed_frames;
        }
        const Parameters md = buildOutputMetadata(env, predicted, missed_frames);
        const std::string serialized = md.dump();
        av_dict_set(&frm.raw()->metadata, metadata_key_out_.c_str(), serialized.c_str(), 0);
        maybeLog(predicted, missed_frames);
        this->sink_->put(frm);
    }

    static std::shared_ptr<TrackBallHolder> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<TrackBallHolder>(edges, params);
        if (params.count("metadata_key_in")) r->metadata_key_in_ = params["metadata_key_in"].get<std::string>();
        if (params.count("metadata_key_out")) r->metadata_key_out_ = params["metadata_key_out"].get<std::string>();
        if (params.count("ball_label")) r->ball_label_ = params["ball_label"].get<std::string>();
        if (params.count("person_label")) r->person_label_ = params["person_label"].get<std::string>();
        if (params.count("output_label")) r->output_label_ = params["output_label"].get<std::string>();
        if (params.count("ball_class")) r->ball_class_ = params["ball_class"];
        if (params.count("person_class")) r->person_class_ = params["person_class"];
        if (params.count("min_ball_conf")) r->min_ball_conf_ = params["min_ball_conf"];
        if (params.count("min_person_conf")) r->min_person_conf_ = params["min_person_conf"];
        if (params.count("max_hand_distance_px")) r->max_hand_distance_px_ = params["max_hand_distance_px"];
        if (params.count("hand_y_ratio")) r->hand_y_ratio_ = params["hand_y_ratio"];
        if (params.count("side_outset_ratio")) r->side_outset_ratio_ = params["side_outset_ratio"];
        if (params.count("upper_y_min_ratio")) r->upper_y_min_ratio_ = params["upper_y_min_ratio"];
        if (params.count("upper_y_max_ratio")) r->upper_y_max_ratio_ = params["upper_y_max_ratio"];
        if (params.count("expanded_person_margin_x_ratio")) r->expanded_person_margin_x_ratio_ = params["expanded_person_margin_x_ratio"];
        if (params.count("expanded_person_margin_y_ratio")) r->expanded_person_margin_y_ratio_ = params["expanded_person_margin_y_ratio"];
        if (params.count("combined_padding_x_ratio")) r->combined_padding_x_ratio_ = params["combined_padding_x_ratio"];
        if (params.count("combined_padding_y_ratio")) r->combined_padding_y_ratio_ = params["combined_padding_y_ratio"];
        if (params.count("max_missed_frames")) r->max_missed_frames_ = params["max_missed_frames"];
        if (params.count("max_center_distance")) r->max_center_distance_ = params["max_center_distance"];
        if (params.count("min_iou_match")) r->min_iou_match_ = params["min_iou_match"];
        if (params.count("emit_predicted")) r->emit_predicted_ = params["emit_predicted"];
        if (params.count("prediction_decay")) r->prediction_decay_ = params["prediction_decay"];
        if (params.count("velocity_smoothing")) r->velocity_smoothing_ = params["velocity_smoothing"];
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"];
        return r;
    }
};

DECLNODE(track_ball_holder, TrackBallHolder);
