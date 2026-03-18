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
    double min_conf_ = 0.10;
    int max_missed_frames_ = 4;
    double max_center_distance_ = 160.0;
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

    bool detectionMatchesTarget(const DetectionBox& det) const {
        bool class_match = false;
        bool label_match = false;
        if (target_class_ >= 0 && det.cls == target_class_) {
            class_match = true;
        }
        if (!target_label_.empty() && det.has_label && det.label == target_label_) {
            label_match = true;
        }
        if (target_class_ < 0 && target_label_.empty()) {
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

    int chooseBestMatch(const std::vector<DetectionBox>& dets, const DetectionBox& reference_box) const {
        int best_index = -1;
        double best_score = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < dets.size(); ++i) {
            const DetectionBox& det = dets[i];
            const double dist = centerDistance(reference_box, det);
            const double overlap = iou(reference_box, det);
            if (dist > max_center_distance_ && overlap < min_iou_match_) {
                continue;
            }
            const double score = det.conf * 4.0 + overlap * 2.0 - dist / std::max(1.0, max_center_distance_);
            if (score > best_score) {
                best_score = score;
                best_index = (int)i;
            }
        }
        return best_index;
    }

    DetectionBox highestConfidence(const std::vector<DetectionBox>& dets) const {
        return *std::max_element(dets.begin(), dets.end(), [](const DetectionBox& a, const DetectionBox& b) {
            return a.conf < b.conf;
        });
    }

    void startTrack(const DetectionBox& det) {
        clearTrack();
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
        std::vector<DetectionBox> dets;
        (void)parseDetections(frm, env, dets);

        DetectionBox output_det;
        DetectionBox* output_ptr = nullptr;
        bool predicted = false;
        int missed_frames = 0;

        if (!dets.empty()) {
            if (!track_.active) {
                startTrack(highestConfidence(dets));
                output_det = track_.box;
                output_ptr = &output_det;
            } else {
                DetectionBox reference = track_.missed_frames > 0 ? predictBox(env) : track_.box;
                const int match_index = chooseBestMatch(dets, reference);
                if (match_index >= 0) {
                    const int frame_gap = track_.missed_frames + 1;
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
        }
        Parameters md = buildOutputMetadata(env, output_ptr, predicted, missed_frames);
        const std::string serialized = md.dump();
        av_dict_set(&frm.raw()->metadata, metadata_key_out_.c_str(), serialized.c_str(), 0);
        maybeLog(output_ptr, predicted, missed_frames);
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
        if (params.count("min_conf")) r->min_conf_ = params["min_conf"];
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

DECLNODE(track_ball, TrackBall);
