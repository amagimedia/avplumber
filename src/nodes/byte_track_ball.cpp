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
    int track_id = 0;
    DetectionBox box;
    double vx = 0.0;
    double vy = 0.0;
    int missed_frames = 0;
    int hits = 0;
    int age = 0;
    double last_conf = 0.0;
    bool confirmed = false;
};

struct TrackOutput {
    DetectionBox box;
    int track_id = 0;
    bool predicted = false;
    int missed_frames = 0;
    bool confirmed = false;
    int hits = 0;
};

struct MatchCandidate {
    size_t track_index = 0;
    size_t det_index = 0;
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

class ByteTrackBall : public NodeSISO<av::VideoFrame, av::VideoFrame>, public IInputReset {
private:
    std::string metadata_key_in_ = "yolo_detections_v1";
    std::string metadata_key_out_ = "ball_track_v1";
    std::string target_label_ = "sports ball";
    int target_class_ = -1;
    std::vector<std::string> target_labels_;
    std::vector<int> target_classes_;
    double min_conf_ = 0.10;
    double high_conf_thresh_ = 0.35;
    double new_track_conf_thresh_ = 0.35;
    double match_iou_thresh_ = 0.05;
    double low_match_iou_thresh_ = 0.01;
    double match_max_center_distance_ = 120.0;
    double low_match_max_center_distance_ = 160.0;
    int max_time_lost_ = 8;
    int min_confirmed_hits_ = 2;
    int max_tracks_ = 8;
    bool emit_predicted_ = true;
    double prediction_decay_ = 0.92;
    double velocity_smoothing_ = 0.60;
    int debug_log_every_n_ = 0;
    uint64_t frame_counter_ = 0;
    int next_track_id_ = 1;
    std::vector<TrackState> tracks_;

    void clearTracks() {
        tracks_.clear();
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
            return true;
        } catch (const std::exception&) {
            return false;
        }
    }

    DetectionBox predictBox(const TrackState& track, const MetadataEnvelope& env) const {
        DetectionBox predicted = shiftedBox(track.box, track.vx, track.vy);
        predicted.conf = std::max(min_conf_, track.last_conf * std::pow(prediction_decay_, (double)(track.missed_frames + 1)));
        predicted.cls = track.box.cls;
        predicted.label = track.box.label;
        predicted.has_label = track.box.has_label;
        return clampBoxToCanvas(predicted, env.model_width, env.model_height);
    }

    void updateTrack(TrackState& track, const DetectionBox& det, int frame_gap) const {
        const double prev_cx = centerX(track.box);
        const double prev_cy = centerY(track.box);
        const double next_cx = centerX(det);
        const double next_cy = centerY(det);
        const double gap = std::max(1, frame_gap);
        const double meas_vx = (next_cx - prev_cx) / gap;
        const double meas_vy = (next_cy - prev_cy) / gap;
        const double carry = std::clamp(velocity_smoothing_, 0.0, 0.95);
        track.vx = track.vx * carry + meas_vx * (1.0 - carry);
        track.vy = track.vy * carry + meas_vy * (1.0 - carry);
        track.box = det;
        track.missed_frames = 0;
        track.hits += 1;
        track.age += 1;
        track.last_conf = det.conf;
        if (track.hits >= std::max(1, min_confirmed_hits_)) {
            track.confirmed = true;
        }
    }

    void startTrack(const DetectionBox& det) {
        TrackState track;
        track.track_id = next_track_id_++;
        track.box = det;
        track.last_conf = det.conf;
        track.hits = 1;
        track.age = 1;
        track.confirmed = track.hits >= std::max(1, min_confirmed_hits_);
        tracks_.push_back(track);
    }

    bool canMatch(const DetectionBox& predicted,
                  const DetectionBox& det,
                  double iou_thresh,
                  double max_center_distance,
                  double& score_out) const {
        const double overlap = iou(predicted, det);
        const double dist = centerDistance(predicted, det);
        if (overlap < iou_thresh && dist > max_center_distance) {
            return false;
        }
        score_out = overlap * 4.0 + det.conf - dist / std::max(1.0, max_center_distance);
        return true;
    }

    std::vector<MatchCandidate> buildMatches(const std::vector<size_t>& track_indices,
                                             const std::vector<DetectionBox>& dets,
                                             const MetadataEnvelope& env,
                                             double iou_thresh,
                                             double max_center_distance,
                                             const std::vector<bool>& track_available,
                                             const std::vector<bool>& det_available) const {
        std::vector<MatchCandidate> candidates;
        for (size_t track_index : track_indices) {
            if (track_index >= tracks_.size() || !track_available[track_index]) continue;
            const DetectionBox predicted = predictBox(tracks_[track_index], env);
            for (size_t det_index = 0; det_index < dets.size(); ++det_index) {
                if (!det_available[det_index]) continue;
                double score = 0.0;
                if (!canMatch(predicted, dets[det_index], iou_thresh, max_center_distance, score)) {
                    continue;
                }
                candidates.push_back(MatchCandidate{track_index, det_index, score});
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const MatchCandidate& a, const MatchCandidate& b) {
            return a.score > b.score;
        });
        return candidates;
    }

    void applyMatches(const std::vector<MatchCandidate>& candidates,
                      const std::vector<DetectionBox>& dets,
                      std::vector<bool>& track_available,
                      std::vector<bool>& det_available) {
        for (const MatchCandidate& candidate : candidates) {
            if (!track_available[candidate.track_index]) continue;
            if (!det_available[candidate.det_index]) continue;
            TrackState& track = tracks_[candidate.track_index];
            updateTrack(track, dets[candidate.det_index], track.missed_frames + 1);
            track_available[candidate.track_index] = false;
            det_available[candidate.det_index] = false;
        }
    }

    void ageTrack(TrackState& track, const MetadataEnvelope& env) const {
        track.age += 1;
        track.missed_frames += 1;
        if (emit_predicted_) {
            track.box = predictBox(track, env);
            track.vx *= prediction_decay_;
            track.vy *= prediction_decay_;
            track.last_conf = track.box.conf;
        }
    }

    void pruneTracks() {
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [&](const TrackState& track) {
            return track.missed_frames > max_time_lost_;
        }), tracks_.end());

        if ((int)tracks_.size() <= std::max(1, max_tracks_)) {
            return;
        }

        std::sort(tracks_.begin(), tracks_.end(), [](const TrackState& a, const TrackState& b) {
            const double sa = (a.confirmed ? 1000.0 : 0.0) + a.hits * 10.0 + a.last_conf - a.missed_frames * 20.0;
            const double sb = (b.confirmed ? 1000.0 : 0.0) + b.hits * 10.0 + b.last_conf - b.missed_frames * 20.0;
            return sa > sb;
        });
        tracks_.resize((size_t)std::max(1, max_tracks_));
    }

    TrackOutput chooseOutput(const MetadataEnvelope& env) const {
        TrackOutput output;
        double best_score = -std::numeric_limits<double>::infinity();
        for (const TrackState& track : tracks_) {
            if (!track.confirmed && track.hits < std::max(1, min_confirmed_hits_)) {
                continue;
            }
            DetectionBox out_box = track.box;
            bool predicted = track.missed_frames > 0;
            if (!emit_predicted_ && predicted) {
                continue;
            }
            if (!finiteBox(out_box)) {
                out_box = predictBox(track, env);
                predicted = true;
            }
            if (!finiteBox(out_box)) continue;
            const double score = (track.confirmed ? 1000.0 : 0.0)
                               + track.hits * 15.0
                               + track.last_conf * 10.0
                               - track.missed_frames * 25.0;
            if (score > best_score) {
                best_score = score;
                output.box = out_box;
                output.track_id = track.track_id;
                output.predicted = predicted;
                output.missed_frames = track.missed_frames;
                output.confirmed = track.confirmed;
                output.hits = track.hits;
            }
        }
        return output;
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
            item["confirmed"] = output.confirmed;
            item["hits"] = output.hits;
            md["detections"].push_back(item);
        }
        md["tracker"] = {
            {"name", "byte_track_ball"},
            {"active_tracks", (int)tracks_.size()},
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
            logstream << "byte_track_ball: frame=" << frame_counter_ << " no selected track";
            return;
        }
        logstream << "byte_track_ball: frame=" << frame_counter_
                  << " track_id=" << output.track_id
                  << " source=" << (output.predicted ? "predicted" : "detected")
                  << " missed=" << output.missed_frames
                  << " hits=" << output.hits
                  << " bbox=[" << output.box.x1 << "," << output.box.y1 << "," << output.box.x2 << "," << output.box.y2 << "]"
                  << " conf=" << output.box.conf;
    }

public:
    using NodeSISO::NodeSISO;

    void resetInput() override {
        clearTracks();
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm) return;

        if (isEofMarker(frm)) {
            clearTracks();
            this->sink_->put(frm);
            return;
        }

        ++frame_counter_;

        MetadataEnvelope env;
        std::vector<DetectionBox> dets;
        (void)parseDetections(frm, env, dets);

        std::vector<DetectionBox> high_dets;
        std::vector<DetectionBox> low_dets;
        high_dets.reserve(dets.size());
        low_dets.reserve(dets.size());
        for (const DetectionBox& det : dets) {
            if (det.conf >= high_conf_thresh_) {
                high_dets.push_back(det);
            } else {
                low_dets.push_back(det);
            }
        }

        std::vector<bool> track_available(tracks_.size(), true);
        std::vector<bool> high_available(high_dets.size(), true);
        std::vector<bool> low_available(low_dets.size(), true);

        std::vector<size_t> confirmed_tracks;
        std::vector<size_t> tentative_tracks;
        for (size_t i = 0; i < tracks_.size(); ++i) {
            if (tracks_[i].confirmed) {
                confirmed_tracks.push_back(i);
            } else {
                tentative_tracks.push_back(i);
            }
        }

        applyMatches(buildMatches(confirmed_tracks, high_dets, env, match_iou_thresh_, match_max_center_distance_, track_available, high_available),
                     high_dets, track_available, high_available);
        applyMatches(buildMatches(tentative_tracks, high_dets, env, match_iou_thresh_, match_max_center_distance_, track_available, high_available),
                     high_dets, track_available, high_available);

        std::vector<size_t> unmatched_tracks;
        for (size_t i = 0; i < tracks_.size(); ++i) {
            if (track_available[i]) {
                unmatched_tracks.push_back(i);
            }
        }
        applyMatches(buildMatches(unmatched_tracks, low_dets, env, low_match_iou_thresh_, low_match_max_center_distance_, track_available, low_available),
                     low_dets, track_available, low_available);

        for (size_t i = 0; i < tracks_.size(); ++i) {
            if (track_available[i]) {
                ageTrack(tracks_[i], env);
            }
        }

        for (size_t i = 0; i < high_dets.size(); ++i) {
            if (!high_available[i]) continue;
            if (high_dets[i].conf < new_track_conf_thresh_) continue;
            startTrack(high_dets[i]);
        }

        pruneTracks();

        const TrackOutput output = chooseOutput(env);
        const Parameters md = buildOutputMetadata(env, output);
        const std::string serialized = md.dump();
        av_dict_set(&frm.raw()->metadata, metadata_key_out_.c_str(), serialized.c_str(), 0);
        maybeLog(output);
        this->sink_->put(frm);
    }

    static std::shared_ptr<ByteTrackBall> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<ByteTrackBall>(edges, params);
        if (params.count("metadata_key_in")) r->metadata_key_in_ = params["metadata_key_in"].get<std::string>();
        if (params.count("metadata_key_out")) r->metadata_key_out_ = params["metadata_key_out"].get<std::string>();
        if (params.count("target_label")) r->target_label_ = params["target_label"].get<std::string>();
        if (params.count("target_class")) r->target_class_ = params["target_class"];
        if (params.count("target_labels")) {
            if (!params["target_labels"].is_array()) {
                throw Error("byte_track_ball: target_labels must be a string array");
            }
            const std::list<std::string> labels = jsonToStringList(params["target_labels"]);
            r->target_labels_.assign(labels.begin(), labels.end());
        }
        if (params.count("target_classes")) {
            if (!params["target_classes"].is_array()) {
                throw Error("byte_track_ball: target_classes must be an integer array");
            }
            for (const auto& item : params["target_classes"]) {
                if (!item.is_number_integer()) {
                    throw Error("byte_track_ball: target_classes must be an integer array");
                }
                r->target_classes_.push_back(item.get<int>());
            }
        }
        if (params.count("min_conf")) r->min_conf_ = params["min_conf"];
        if (params.count("high_conf_thresh")) r->high_conf_thresh_ = params["high_conf_thresh"];
        if (params.count("new_track_conf_thresh")) r->new_track_conf_thresh_ = params["new_track_conf_thresh"];
        if (params.count("match_iou_thresh")) r->match_iou_thresh_ = params["match_iou_thresh"];
        if (params.count("low_match_iou_thresh")) r->low_match_iou_thresh_ = params["low_match_iou_thresh"];
        if (params.count("match_max_center_distance")) r->match_max_center_distance_ = params["match_max_center_distance"];
        if (params.count("low_match_max_center_distance")) r->low_match_max_center_distance_ = params["low_match_max_center_distance"];
        if (params.count("max_time_lost")) r->max_time_lost_ = params["max_time_lost"];
        if (params.count("min_confirmed_hits")) r->min_confirmed_hits_ = params["min_confirmed_hits"];
        if (params.count("max_tracks")) r->max_tracks_ = params["max_tracks"];
        if (params.count("emit_predicted")) r->emit_predicted_ = params["emit_predicted"];
        if (params.count("prediction_decay")) r->prediction_decay_ = params["prediction_decay"];
        if (params.count("velocity_smoothing")) r->velocity_smoothing_ = params["velocity_smoothing"];
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"];
        return r;
    }
};

DECLNODE(byte_track_ball, ByteTrackBall);
