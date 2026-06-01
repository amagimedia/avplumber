#include "../../node_common.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include "reframe_metadata_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

class BallTrackValidator : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
    using DetectionBox = avp_sport_reframe::DetectionBox;

    std::string raw_metadata_key_ = "raw_yolo_ball";
    std::string tracked_metadata_key_ = "tracked_yolo_ball";
    std::string scene_diff_metadata_key_ = "scene_diff";
    std::string output_metadata_key_ = "validated_yolo_ball";
    std::string ball_label_ = "basketball";

    double min_conf_ = 0.01;
    double scene_diff_mean_abs_reset_ = 60.0;
    double max_jump_px_per_frame_ = 110.0;
    int min_confirm_frames_ = 2;
    int max_coast_frames_ = 6;
    int debug_log_every_n_ = 0;

    uint64_t frame_counter_ = 0;
    uint64_t stat_accepted_ = 0;
    uint64_t stat_rejected_ = 0;
    uint64_t stat_scene_resets_ = 0;

    DetectionBox last_accepted_;
    bool have_last_accepted_ = false;
    uint64_t last_accepted_frame_ = 0;

    DetectionBox pending_;
    bool have_pending_ = false;
    int pending_frames_ = 0;

    struct BallCandidate {
        DetectionBox box;
        Parameters item;
        bool present = false;
        bool predicted = false;
        std::string source;
        int coast_age = 0;
    };

    struct MetadataPayload {
        Parameters md;
        bool present = false;
        double model_w = 0.0;
        double model_h = 0.0;
    };

    static double centerDistance(const DetectionBox& a, const DetectionBox& b) {
        const double dx = avp_sport_reframe::centerX(a) - avp_sport_reframe::centerX(b);
        const double dy = avp_sport_reframe::centerY(a) - avp_sport_reframe::centerY(b);
        return std::sqrt(dx * dx + dy * dy);
    }

    MetadataPayload parsePayload(const AVFrame* raw, const std::string& key, int frame_w, int frame_h) const {
        MetadataPayload out;
        out.model_w = (double)frame_w;
        out.model_h = (double)frame_h;
        if (!raw || !raw->metadata || key.empty()) return out;
        AVDictionaryEntry* entry = av_dict_get(raw->metadata, key.c_str(), nullptr, 0);
        if (!entry || !entry->value) return out;
        try {
            out.md = Parameters::parse(entry->value);
            out.present = out.md.is_object();
            if (out.present) {
                out.model_w = out.md.value("model_width", out.model_w);
                out.model_h = out.md.value("model_height", out.model_h);
            }
        } catch (...) {
            out.present = false;
        }
        return out;
    }

    bool parseCandidate(const MetadataPayload& payload, BallCandidate& out) const {
        if (!payload.present || !payload.md.contains("detections") || !payload.md["detections"].is_array()) {
            return false;
        }

        bool found = false;
        BallCandidate best;
        best.box.conf = -1.0;
        for (const auto& item : payload.md["detections"]) {
            if (!item.is_object()) continue;
            if (!item.contains("xyxy") || !item["xyxy"].is_array() || item["xyxy"].size() < 4) continue;
            DetectionBox box;
            box.cls = item.value("cls", -1);
            box.conf = item.value("conf", 0.0);
            box.model_index = item.value("model_index", -1);
            box.x1 = item["xyxy"][0].get<double>();
            box.y1 = item["xyxy"][1].get<double>();
            box.x2 = item["xyxy"][2].get<double>();
            box.y2 = item["xyxy"][3].get<double>();
            if (item.contains("label") && item["label"].is_string()) {
                box.label = item["label"].get<std::string>();
                box.has_label = true;
            }
            if (item.contains("track_id") && !item["track_id"].is_null()) {
                try {
                    box.track_id = item["track_id"].get<int>();
                    box.has_track_id = box.track_id >= 0;
                } catch (...) {
                    box.track_id = -1;
                    box.has_track_id = false;
                }
            }
            if (box.x2 < box.x1) std::swap(box.x1, box.x2);
            if (box.y2 < box.y1) std::swap(box.y1, box.y2);
            if (!avp_sport_reframe::finiteBox(box)) continue;
            if (!avp_sport_reframe::labelMatches(box, ball_label_)) continue;
            if (box.conf < min_conf_) continue;
            if (!found || box.conf > best.box.conf) {
                best.box = box;
                best.item = item;
                best.present = true;
                best.predicted = item.value("predicted", false);
                best.source = item.value("source", std::string());
                best.coast_age = item.value("coast_streak", item.value("missed_frames", 0));
                found = true;
            }
        }
        if (!found) return false;
        out = best;
        return true;
    }

    bool isSamePending(const DetectionBox& box) const {
        if (!have_pending_) return false;
        return centerDistance(box, pending_) <= std::max(8.0, max_jump_px_per_frame_);
    }

    void resetState() {
        have_last_accepted_ = false;
        last_accepted_ = DetectionBox{};
        last_accepted_frame_ = 0;
        have_pending_ = false;
        pending_ = DetectionBox{};
        pending_frames_ = 0;
    }

    bool validateCandidate(const BallCandidate& cand,
                           bool raw_present,
                           bool scene_reset,
                           double& jump_px,
                           int& gap_frames,
                           std::string& state,
                           std::string& reason) {
        jump_px = 0.0;
        gap_frames = 0;
        if (scene_reset) {
            resetState();
            state = "rejected";
            reason = "scene_reset";
            return false;
        }
        if (!cand.present) {
            have_pending_ = false;
            pending_frames_ = 0;
            state = "rejected";
            reason = "no_tracked_ball";
            return false;
        }
        if (cand.coast_age > max_coast_frames_) {
            state = "rejected";
            reason = "max_coast_exceeded";
            return false;
        }

        bool jump_ok = true;
        if (have_last_accepted_) {
            gap_frames = (int)std::max<uint64_t>(1, frame_counter_ - last_accepted_frame_);
            jump_px = centerDistance(cand.box, last_accepted_);
            jump_ok = jump_px <= max_jump_px_per_frame_ * (double)gap_frames;
        }

        const bool tracked_from_real = !cand.predicted && cand.source != "coasted";
        if (!have_last_accepted_ || !jump_ok) {
            if (tracked_from_real && raw_present && isSamePending(cand.box)) {
                ++pending_frames_;
            } else {
                pending_ = cand.box;
                have_pending_ = true;
                pending_frames_ = tracked_from_real && raw_present ? 1 : 0;
            }

            if (pending_frames_ < std::max(1, min_confirm_frames_)) {
                state = "pending";
                reason = !have_last_accepted_ ? "min_confirm" : "jump_confirm";
                return false;
            }
        }

        last_accepted_ = cand.box;
        have_last_accepted_ = true;
        last_accepted_frame_ = frame_counter_;
        have_pending_ = false;
        pending_frames_ = 0;
        state = cand.predicted ? "accepted_coasted" : "accepted";
        reason = "ok";
        return true;
    }

    Parameters buildOutput(const MetadataPayload& tracked_payload,
                           const BallCandidate& cand,
                           bool accepted,
                           const std::string& state,
                           const std::string& reason,
                           bool raw_present,
                           bool scene_reset,
                           double jump_px,
                           int gap_frames) const {
        Parameters out;
        out["coord_space"] = tracked_payload.present
            ? tracked_payload.md.value("coord_space", std::string("model"))
            : std::string("model");
        out["model_width"] = tracked_payload.model_w;
        out["model_height"] = tracked_payload.model_h;
        if (tracked_payload.present && tracked_payload.md.contains("models")) out["models"] = tracked_payload.md["models"];
        out["detections"] = Parameters::array();
        if (accepted && tracked_payload.present && tracked_payload.md.contains("trail")) {
            out["trail"] = tracked_payload.md["trail"];
        } else {
            out["trail"] = Parameters::array();
        }
        out["validation_state"] = state;
        out["reject_reason"] = reason;
        out["accepted"] = accepted;
        out["raw_present"] = raw_present;
        out["tracked_present"] = cand.present;
        out["scene_reset"] = scene_reset;
        out["tracked_source"] = cand.source;
        out["coast_age_frames"] = cand.coast_age;
        out["pending_frames"] = pending_frames_;
        out["jump_px"] = jump_px;
        out["gap_frames"] = gap_frames;
        out["strategy"] = "confirm_jump_coast_scene";

        if (accepted && cand.present) {
            Parameters item = cand.item;
            item["validated"] = true;
            item["validation_state"] = state;
            item["reject_reason"] = reason;
            item["jump_px"] = jump_px;
            item["gap_frames"] = gap_frames;
            item["coast_age_frames"] = cand.coast_age;
            out["detections"].push_back(item);
        }
        return out;
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    bool consumeEofIfPresent() override {
        return false;
    }

    ~BallTrackValidator() {
        if (frame_counter_ == 0) return;
        logstream << "ball_track_validator: === summary ===";
        logstream << "  total frames:    " << frame_counter_;
        logstream << "  accepted:        " << stat_accepted_;
        logstream << "  rejected:        " << stat_rejected_;
        logstream << "  scene resets:    " << stat_scene_resets_;
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
        AVFrame* raw = frm.raw();
        const bool scene_reset = raw && avp_sport_reframe::sceneResetRequested(raw, scene_diff_metadata_key_, scene_diff_mean_abs_reset_);
        if (scene_reset) ++stat_scene_resets_;

        MetadataPayload tracked_payload = parsePayload(raw, tracked_metadata_key_, frm.width(), frm.height());
        MetadataPayload raw_payload = parsePayload(raw, raw_metadata_key_, frm.width(), frm.height());
        BallCandidate tracked;
        BallCandidate raw_ball;
        const bool tracked_present = parseCandidate(tracked_payload, tracked);
        const bool raw_present = parseCandidate(raw_payload, raw_ball);
        (void)tracked_present;

        double jump_px = 0.0;
        int gap_frames = 0;
        std::string state;
        std::string reason;
        const bool accepted = validateCandidate(tracked, raw_present, scene_reset, jump_px, gap_frames, state, reason);
        if (accepted) ++stat_accepted_;
        else ++stat_rejected_;

        Parameters out = buildOutput(tracked_payload, tracked, accepted, state, reason,
                                     raw_present, scene_reset, jump_px, gap_frames);
        av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), out.dump().c_str(), 0);

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "ball_track_validator: frame=" << frame_counter_
                      << " accepted=" << (accepted ? 1 : 0)
                      << " state=" << state
                      << " reason=" << reason
                      << " raw=" << (raw_present ? 1 : 0)
                      << " tracked=" << (tracked.present ? 1 : 0)
                      << " src=" << tracked.source
                      << " coast_age=" << tracked.coast_age
                      << " pending=" << pending_frames_
                      << " jump=" << jump_px
                      << " gap=" << gap_frames
                      << " scene_reset=" << (scene_reset ? 1 : 0);
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<BallTrackValidator> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<BallTrackValidator>(edges, params);
        r->auto_eof_ = false;

        if (params.count("raw_metadata_key")) r->raw_metadata_key_ = params["raw_metadata_key"].get<std::string>();
        if (params.count("tracked_metadata_key")) r->tracked_metadata_key_ = params["tracked_metadata_key"].get<std::string>();
        if (params.count("scene_diff_metadata_key")) r->scene_diff_metadata_key_ = params["scene_diff_metadata_key"].get<std::string>();
        if (params.count("output_metadata_key")) r->output_metadata_key_ = params["output_metadata_key"].get<std::string>();
        if (params.count("ball_label")) r->ball_label_ = params["ball_label"].get<std::string>();
        if (params.count("min_conf")) r->min_conf_ = params["min_conf"].get<double>();
        if (params.count("scene_diff_mean_abs_reset")) r->scene_diff_mean_abs_reset_ = params["scene_diff_mean_abs_reset"].get<double>();
        if (params.count("max_jump_px_per_frame")) r->max_jump_px_per_frame_ = params["max_jump_px_per_frame"].get<double>();
        if (params.count("min_confirm_frames")) r->min_confirm_frames_ = params["min_confirm_frames"].get<int>();
        if (params.count("max_coast_frames")) r->max_coast_frames_ = params["max_coast_frames"].get<int>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();

        r->min_conf_ = std::max(0.0, r->min_conf_);
        r->max_jump_px_per_frame_ = std::max(1.0, r->max_jump_px_per_frame_);
        r->min_confirm_frames_ = std::max(1, r->min_confirm_frames_);
        r->max_coast_frames_ = std::max(0, r->max_coast_frames_);
        return r;
    }
};

DECLNODE(ball_track_validator, BallTrackValidator)
