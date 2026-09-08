#include "../../node_common.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include <bytetrack/BYTETracker.h>

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>
#include <vector>

namespace {

bool iequals(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i])) return false;
    }
    return true;
}

float computeIoU(const std::vector<float>& a_tlbr, float bx1, float by1, float bx2, float by2) {
    float ix1 = std::max(a_tlbr[0], bx1);
    float iy1 = std::max(a_tlbr[1], by1);
    float ix2 = std::min(a_tlbr[2], bx2);
    float iy2 = std::min(a_tlbr[3], by2);
    float iw = std::max(0.0f, ix2 - ix1);
    float ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float area_a = (a_tlbr[2] - a_tlbr[0]) * (a_tlbr[3] - a_tlbr[1]);
    float area_b = (bx2 - bx1) * (by2 - by1);
    float u = area_a + area_b - inter;
    return (u > 0) ? inter / u : 0.0f;
}

} // anonymous namespace

class ObjectTracker : public NodeSISO<av::VideoFrame, av::VideoFrame>, public IInputReset, public ReportsFinishByFlag {

    std::string metadata_key_ = "yolo_detections";
    std::vector<std::string> target_labels_;
    int target_class_ = -1;
    bool label_case_sensitive_ = false;
    double min_conf_ = 0.01;
    int frame_rate_ = 30;

    // Shot-aware reset
    std::string camera_shot_metadata_key_;  // empty = disabled
    bool require_wide_shot_ = false;
    int track_buffer_ = 30;
    float track_thresh_ = 0.5f;
    float high_thresh_ = 0.6f;
    float match_thresh_ = 0.8f;
    float low_match_thresh_ = 0.5f;
    bool predict_on_empty_ = true;
    bool emit_lost_tracks_ = false;
    int debug_log_every_n_ = 0;

    std::unique_ptr<bytetrack::BYTETracker> tracker_;
    uint64_t frame_counter_ = 0;

    // Statistics
    uint64_t stat_total_frames_ = 0;
    uint64_t stat_resets_ = 0;              // shot transition resets
    uint64_t stat_total_target_dets_ = 0;   // raw model detections matching target labels
    uint64_t stat_matched_ = 0;             // detections matched to a track (direct or IoU)
    uint64_t stat_unmatched_ = 0;           // detections with track_id=-1
    uint64_t stat_predicted_ = 0;           // predicted boxes (no detection, predict_on_empty)
    uint64_t stat_lost_emitted_ = 0;        // lost tracks emitted
    int stat_max_track_id_ = 0;             // highest track_id assigned

    void initTracker() {
        tracker_ = std::make_unique<bytetrack::BYTETracker>(frame_rate_, track_buffer_);
        tracker_->track_thresh = track_thresh_;
        tracker_->high_thresh = high_thresh_;
        tracker_->match_thresh = match_thresh_;
        tracker_->low_match_thresh = low_match_thresh_;
    }

    void resetTrackerState() {
        initTracker();
    }

    void resetState() {
        frame_counter_ = 0;
        resetTrackerState();
    }

    bool matchesTarget(const Parameters& det) const {
        if (target_class_ >= 0 && det.contains("cls")) {
            if (det["cls"].get<int>() == target_class_) return true;
        }
        if (!target_labels_.empty() && det.contains("label")) {
            std::string lbl = det["label"].get<std::string>();
            for (const auto& tl : target_labels_) {
                if (label_case_sensitive_) {
                    if (lbl == tl) return true;
                } else {
                    if (iequals(lbl, tl)) return true;
                }
            }
        }
        return target_class_ < 0 && target_labels_.empty();
    }

    bool readShotInfo(const av::VideoFrame& frm, std::string& shot_type_out, bool& transition_out) const {
        shot_type_out.clear();
        transition_out = false;
        if (camera_shot_metadata_key_.empty()) return false;

        const AVFrame* raw = frm.raw();
        if (!raw || !raw->metadata) return false;

        AVDictionaryEntry* shot_entry = av_dict_get(raw->metadata, camera_shot_metadata_key_.c_str(), nullptr, 0);
        if (!shot_entry || !shot_entry->value) return false;

        try {
            Parameters shot_md = Parameters::parse(shot_entry->value);
            shot_type_out = shot_md.value("camera_shot_type", std::string());
            transition_out = shot_md.value("camera_shot_transition", false);
            return true;
        } catch (...) {
            shot_type_out.clear();
            transition_out = false;
            return false;
        }
    }

    void stripTrackAnnotation(Parameters& det) const {
        det.erase("track_id");
        det.erase("tracklet_len");
        det.erase("track_state");
        det.erase("predicted");
        det.erase("velocity");
        det.erase("predicted_xyxy");
    }

    Parameters buildTrackAnnotation(const bytetrack::STrack& track, bool predicted) const {
        Parameters ann;
        ann["track_id"] = track.track_id;
        ann["tracklet_len"] = track.tracklet_len;
        ann["track_state"] = (track.state == bytetrack::TrackState::Lost) ? "lost" :
                             (track.state == bytetrack::TrackState::New) ? "new" : "tracked";
        ann["predicted"] = predicted;

        // Velocity from Kalman state: mean[4]=dx, mean[5]=dy
        Parameters vel = Parameters::array();
        vel.push_back((float)track.mean[4]);
        vel.push_back((float)track.mean[5]);
        ann["velocity"] = vel;

        // Predicted xyxy from track's tlbr
        Parameters pred_xyxy = Parameters::array();
        pred_xyxy.push_back(track.tlbr[0]);
        pred_xyxy.push_back(track.tlbr[1]);
        pred_xyxy.push_back(track.tlbr[2]);
        pred_xyxy.push_back(track.tlbr[3]);
        ann["predicted_xyxy"] = pred_xyxy;

        return ann;
    }

    Parameters buildPredictedDetection(const bytetrack::STrack& track) const {
        Parameters det;
        det["track_id"] = track.track_id;
        det["tracklet_len"] = track.tracklet_len;
        det["track_state"] = "lost";
        det["predicted"] = true;
        det["conf"] = (double)track.score;

        Parameters xyxy = Parameters::array();
        xyxy.push_back(track.tlbr[0]);
        xyxy.push_back(track.tlbr[1]);
        xyxy.push_back(track.tlbr[2]);
        xyxy.push_back(track.tlbr[3]);
        det["xyxy"] = xyxy;

        Parameters vel = Parameters::array();
        vel.push_back((float)track.mean[4]);
        vel.push_back((float)track.mean[5]);
        det["velocity"] = vel;
        det["predicted_xyxy"] = xyxy;

        return det;
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;
    bool consumeEofIfPresent() override {
        return false;
    }

    void resetInput() override { resetState(); }

    ~ObjectTracker() {
        uint64_t total_output = stat_matched_ + stat_unmatched_ + stat_predicted_ + stat_lost_emitted_;
        if (stat_total_frames_ == 0) return;

        logstream << "object_tracker: === tracking summary ===";
        logstream << "  total frames:        " << stat_total_frames_;
        logstream << "  shot resets:         " << stat_resets_;
        logstream << "  raw target dets:     " << stat_total_target_dets_;
        if (stat_total_frames_ > 0) {
            logstream << "  avg dets/frame:      " << (double)stat_total_target_dets_ / stat_total_frames_;
        }
        logstream << "  matched to tracks:   " << stat_matched_;
        logstream << "  unmatched (id=-1):   " << stat_unmatched_;
        logstream << "  predicted (no det):  " << stat_predicted_;
        logstream << "  lost emitted:        " << stat_lost_emitted_;
        logstream << "  total output dets:   " << total_output;
        logstream << "  max track_id:        " << stat_max_track_id_;
        if (stat_total_target_dets_ > 0) {
            logstream << "  match rate:          " << (100.0 * stat_matched_ / stat_total_target_dets_) << "%";
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

        // Shot-aware reset: reset tracker on any shot transition so IDs restart from 1.
        std::string shot_type;
        bool shot_transition = false;
        const bool have_shot_info = readShotInfo(frm, shot_type, shot_transition);
        if (shot_transition) {
            resetTrackerState();
            stat_resets_++;
            if (debug_log_every_n_ > 0) {
                logstream << "object_tracker: frame=" << frame_counter_
                          << " reset (shot transition to " << (shot_type.empty() ? std::string("?") : shot_type) << ")";
            }
        }

        if (!tracker_) initTracker();

        // Parse metadata
        const AVFrame* raw = frm.raw();
        bool has_metadata = false;
        Parameters md;

        if (raw && raw->metadata) {
            AVDictionaryEntry* entry = av_dict_get(raw->metadata, metadata_key_.c_str(), nullptr, 0);
            if (entry && entry->value) {
                try {
                    md = Parameters::parse(entry->value);
                    has_metadata = true;
                } catch (...) {}
            }
        }

        // Split detections
        std::vector<Parameters> target_dets;
        std::vector<int> target_indices;  // index into original detections array
        Parameters passthrough_dets = Parameters::array();

        if (has_metadata && md.contains("detections") && md["detections"].is_array()) {
            int idx = 0;
            for (const auto& item : md["detections"]) {
                if (!item.is_object()) {
                    passthrough_dets.push_back(item);
                    idx++;
                    continue;
                }
                double conf = item.value("conf", 0.0);
                if (matchesTarget(item) && conf >= min_conf_) {
                    target_dets.push_back(item);
                    target_indices.push_back(idx);
                } else {
                    passthrough_dets.push_back(item);
                }
                idx++;
            }
        }

        const bool shot_suppressed = require_wide_shot_ && (!have_shot_info || shot_type != "wide");
        if (shot_suppressed) {
            resetTrackerState();

            if (has_metadata) {
                Parameters out_md = md;
                Parameters out_dets = passthrough_dets;
                for (auto det_out : target_dets) {
                    stripTrackAnnotation(det_out);
                    out_dets.push_back(det_out);
                }
                out_md["detections"] = out_dets;
                av_dict_set(&frm.raw()->metadata, metadata_key_.c_str(), out_md.dump().c_str(), 0);
            }

            stat_total_frames_++;
            stat_total_target_dets_ += target_dets.size();

            if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
                logstream << "object_tracker: frame=" << frame_counter_
                          << " suppressed shot_type=" << (shot_type.empty() ? std::string("<missing>") : shot_type)
                          << " targets=" << target_dets.size();
            }

            this->sink_->put(frm);
            return;
        }

        // Convert target detections to ByteTrack Objects
        std::vector<bytetrack::Object> objects;
        for (size_t i = 0; i < target_dets.size(); i++) {
            const auto& det = target_dets[i];
            if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) continue;
            float x1 = det["xyxy"][0].get<float>();
            float y1 = det["xyxy"][1].get<float>();
            float x2 = det["xyxy"][2].get<float>();
            float y2 = det["xyxy"][3].get<float>();

            bytetrack::Object obj;
            obj.x = x1;
            obj.y = y1;
            obj.w = x2 - x1;
            obj.h = y2 - y1;
            obj.label = det.value("cls", 0);
            obj.prob = det.value("conf", 0.0f);
            obj.detection_index = (int)i;
            objects.push_back(obj);
        }

        // Run tracker (always, to keep frame_id in sync)
        auto output_tracks = tracker_->update(objects);

        // Build output detections
        Parameters out_dets = passthrough_dets;

        stat_total_frames_++;
        stat_total_target_dets_ += target_dets.size();

        if (!target_dets.empty()) {
            // Map output tracks back to source detections via detection_index
            std::vector<bool> matched(target_dets.size(), false);

            for (const auto& track : output_tracks) {
                int det_idx = track.detection_index;
                if (det_idx >= 0 && det_idx < (int)target_dets.size()) {
                    // Direct index mapping
                    Parameters det_out = target_dets[det_idx];
                    Parameters ann = buildTrackAnnotation(track, false);
                    for (auto it = ann.begin(); it != ann.end(); ++it) {
                        det_out[it.key()] = it.value();
                    }
                    out_dets.push_back(det_out);
                    matched[det_idx] = true;
                    stat_matched_++;
                    if (track.track_id > stat_max_track_id_) stat_max_track_id_ = track.track_id;
                } else {
                    // Fallback: IoU match if index not available
                    float best_iou = 0.0f;
                    int best_j = -1;
                    for (size_t j = 0; j < target_dets.size(); j++) {
                        if (matched[j]) continue;
                        const auto& d = target_dets[j];
                        if (!d.contains("xyxy") || d["xyxy"].size() < 4) continue;
                        float iou = computeIoU(track.tlbr,
                                               d["xyxy"][0].get<float>(), d["xyxy"][1].get<float>(),
                                               d["xyxy"][2].get<float>(), d["xyxy"][3].get<float>());
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_j = (int)j;
                        }
                    }
                    if (best_j >= 0 && best_iou > 0.3f) {
                        Parameters det_out = target_dets[best_j];
                        Parameters ann = buildTrackAnnotation(track, false);
                        for (auto it = ann.begin(); it != ann.end(); ++it) {
                            det_out[it.key()] = it.value();
                        }
                        out_dets.push_back(det_out);
                        matched[best_j] = true;
                        stat_matched_++;
                        if (track.track_id > stat_max_track_id_) stat_max_track_id_ = track.track_id;
                    }
                }
            }

            // Unmatched target detections: pass through with track_id = -1
            for (size_t j = 0; j < target_dets.size(); j++) {
                if (!matched[j]) {
                    Parameters det_out = target_dets[j];
                    det_out["track_id"] = -1;
                    det_out["predicted"] = false;
                    out_dets.push_back(det_out);
                    stat_unmatched_++;
                }
            }

            // Optionally emit lost tracks
            if (emit_lost_tracks_) {
                for (const auto& lost : tracker_->get_lost_stracks()) {
                    out_dets.push_back(buildPredictedDetection(lost));
                    stat_lost_emitted_++;
                }
            }
        } else if (predict_on_empty_) {
            // No target detections: emit predicted boxes from tracked stracks
            for (const auto& track : tracker_->get_tracked_stracks()) {
                if (track.is_activated) {
                    out_dets.push_back(buildPredictedDetection(track));
                    stat_predicted_++;
                }
            }
            if (emit_lost_tracks_) {
                for (const auto& lost : tracker_->get_lost_stracks()) {
                    out_dets.push_back(buildPredictedDetection(lost));
                    stat_lost_emitted_++;
                }
            }
        } else if (!has_metadata) {
            // No metadata and predict_on_empty is false: pass frame unchanged
            this->sink_->put(frm);
            return;
        }

        // Write output metadata
        Parameters out_md;
        if (has_metadata) {
            out_md["coord_space"] = md.value("coord_space", std::string("model"));
            out_md["model_width"] = md.value("model_width", (double)frm.width());
            out_md["model_height"] = md.value("model_height", (double)frm.height());
            if (md.contains("models")) out_md["models"] = md["models"];
        }
        out_md["detections"] = out_dets;

        std::string serialized = out_md.dump();
        av_dict_set(&frm.raw()->metadata, metadata_key_.c_str(), serialized.c_str(), 0);

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "object_tracker: frame=" << frame_counter_
                      << " targets=" << target_dets.size()
                      << " tracked=" << output_tracks.size()
                      << " lost=" << tracker_->get_lost_stracks().size();
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<ObjectTracker> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<ObjectTracker>(edges, params);
        r->auto_eof_ = false;

        if (params.count("camera_shot_metadata_key")) r->camera_shot_metadata_key_ = params["camera_shot_metadata_key"].get<std::string>();
        if (params.count("require_wide_shot")) r->require_wide_shot_ = params["require_wide_shot"].get<bool>();
        if (params.count("metadata_key")) r->metadata_key_ = params["metadata_key"].get<std::string>();
        if (params.count("target_labels")) {
            if (!params["target_labels"].is_array()) throw Error("object_tracker: target_labels must be a string array");
            r->target_labels_.clear();
            for (const auto& l : params["target_labels"]) {
                r->target_labels_.push_back(l.get<std::string>());
            }
        }
        if (params.count("target_class")) r->target_class_ = params["target_class"];
        if (params.count("label_case_sensitive")) r->label_case_sensitive_ = params["label_case_sensitive"];
        if (params.count("min_conf")) r->min_conf_ = params["min_conf"];
        if (params.count("frame_rate")) r->frame_rate_ = params["frame_rate"];
        if (params.count("track_buffer")) r->track_buffer_ = params["track_buffer"];
        if (params.count("track_thresh")) r->track_thresh_ = params["track_thresh"];
        if (params.count("high_thresh")) r->high_thresh_ = params["high_thresh"];
        if (params.count("match_thresh")) r->match_thresh_ = params["match_thresh"];
        if (params.count("low_match_thresh")) r->low_match_thresh_ = params["low_match_thresh"];
        if (params.count("predict_on_empty")) r->predict_on_empty_ = params["predict_on_empty"];
        if (params.count("emit_lost_tracks")) r->emit_lost_tracks_ = params["emit_lost_tracks"];
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"];

        r->initTracker();

        return r;
    }
};

DECLNODE(object_tracker, ObjectTracker)
