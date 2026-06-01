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
#include <vector>

class BallTrackValidator : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
    using DetectionBox = avp_sport_reframe::DetectionBox;

    std::string raw_metadata_key_ = "raw_yolo_ball";
    std::string tracked_metadata_key_ = "tracked_yolo_ball";
    std::string scene_diff_metadata_key_ = "scene_diff";
    std::string player_metadata_key_ = "yolo_players";
    std::string output_metadata_key_ = "validated_yolo_ball";
    std::string ball_label_ = "basketball";
    std::vector<std::string> player_labels_ = {"Player"};

    double min_conf_ = 0.01;
    double min_player_conf_ = 0.25;
    double scene_diff_mean_abs_reset_ = 60.0;
    double max_jump_px_per_frame_ = 110.0;
    double player_near_distance_px_ = 90.0;
    double player_far_distance_px_ = 220.0;
    double player_jump_relax_ = 1.35;
    double raw_fallback_min_conf_ = 0.04;
    double raw_fallback_same_ball_px_ = 80.0;
    double raw_fallback_max_jump_px_per_frame_ = 150.0;
    int min_confirm_frames_ = 2;
    int max_coast_frames_ = 6;
    int raw_fallback_confirm_frames_ = 2;
    bool player_aware_enabled_ = true;
    bool raw_fallback_enabled_ = true;
    int debug_log_every_n_ = 0;

    uint64_t frame_counter_ = 0;
    uint64_t stat_accepted_ = 0;
    uint64_t stat_rejected_ = 0;
    uint64_t stat_scene_resets_ = 0;
    uint64_t stat_player_rejects_ = 0;
    uint64_t stat_raw_fallbacks_ = 0;

    DetectionBox last_accepted_;
    bool have_last_accepted_ = false;
    uint64_t last_accepted_frame_ = 0;

    DetectionBox pending_;
    bool have_pending_ = false;
    int pending_frames_ = 0;

    DetectionBox raw_pending_;
    bool have_raw_pending_ = false;
    uint64_t raw_pending_frame_ = 0;
    int raw_pending_frames_ = 0;

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

    struct PlayerContext {
        int count = 0;
        double nearest_distance = std::numeric_limits<double>::max();
        int nearest_track_id = -1;
        double nearest_conf = 0.0;
        bool near_player = false;
        bool far_from_players = false;
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

    bool matchesPlayerLabel(const DetectionBox& d) const {
        if (!d.has_label) return false;
        for (const auto& label : player_labels_) {
            if (d.label == label) return true;
        }
        return false;
    }

    std::vector<DetectionBox> validPlayers(const AVFrame* raw, double& model_w, double& model_h) const {
        std::vector<DetectionBox> parsed;
        std::vector<DetectionBox> out;
        if (!player_aware_enabled_) return out;
        if (!avp_sport_reframe::parseDetections(raw, player_metadata_key_, parsed, model_w, model_h)) return out;
        for (const auto& p : parsed) {
            if (!matchesPlayerLabel(p)) continue;
            if (p.conf < min_player_conf_) continue;
            out.push_back(p);
        }
        return out;
    }

    PlayerContext playerContextFor(const BallCandidate& cand,
                                   const std::vector<DetectionBox>& players) const {
        PlayerContext ctx;
        ctx.count = (int)players.size();
        if (!player_aware_enabled_ || !cand.present || players.empty()) return ctx;

        const double bx = avp_sport_reframe::centerX(cand.box);
        const double by = avp_sport_reframe::centerY(cand.box);
        for (const auto& p : players) {
            const double dist = avp_sport_reframe::pointToBoxDistance(bx, by, p);
            if (dist < ctx.nearest_distance) {
                ctx.nearest_distance = dist;
                ctx.nearest_track_id = p.has_track_id ? p.track_id : -1;
                ctx.nearest_conf = p.conf;
            }
        }
        ctx.near_player = ctx.nearest_distance <= player_near_distance_px_;
        ctx.far_from_players = ctx.nearest_distance > player_far_distance_px_;
        return ctx;
    }

    bool isSamePending(const DetectionBox& box) const {
        if (!have_pending_) return false;
        return centerDistance(box, pending_) <= std::max(8.0, max_jump_px_per_frame_);
    }

    bool isSameRawPending(const DetectionBox& box) const {
        if (!have_raw_pending_) return false;
        const uint64_t gap = std::max<uint64_t>(1, frame_counter_ - raw_pending_frame_);
        const double gate = std::max(raw_fallback_same_ball_px_,
                                     raw_fallback_max_jump_px_per_frame_ * (double)gap);
        return centerDistance(box, raw_pending_) <= gate;
    }

    void resetRawFallback() {
        have_raw_pending_ = false;
        raw_pending_ = DetectionBox{};
        raw_pending_frame_ = 0;
        raw_pending_frames_ = 0;
    }

    void resetState() {
        have_last_accepted_ = false;
        last_accepted_ = DetectionBox{};
        last_accepted_frame_ = 0;
        have_pending_ = false;
        pending_ = DetectionBox{};
        pending_frames_ = 0;
        resetRawFallback();
    }

    bool validateCandidate(const BallCandidate& cand,
                           bool raw_present,
                           bool scene_reset,
                           const PlayerContext& player_ctx,
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
        if (player_aware_enabled_ && cand.predicted && player_ctx.count > 0 && player_ctx.far_from_players) {
            ++stat_player_rejects_;
            state = "rejected";
            reason = "coast_far_from_players";
            return false;
        }

        bool jump_ok = true;
        if (have_last_accepted_) {
            gap_frames = (int)std::max<uint64_t>(1, frame_counter_ - last_accepted_frame_);
            jump_px = centerDistance(cand.box, last_accepted_);
            double allowed_jump = max_jump_px_per_frame_ * (double)gap_frames;
            if (player_aware_enabled_ && player_ctx.near_player && raw_present &&
                !cand.predicted && cand.source != "coasted") {
                allowed_jump *= player_jump_relax_;
            }
            jump_ok = jump_px <= allowed_jump;
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

            int confirm_needed = std::max(1, min_confirm_frames_);
            if (player_aware_enabled_ && player_ctx.near_player && tracked_from_real && raw_present) {
                confirm_needed = 1;
            } else if (player_aware_enabled_ && player_ctx.count > 0 && player_ctx.far_from_players) {
                confirm_needed += 1;
            }

            if (pending_frames_ < confirm_needed) {
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
        resetRawFallback();
        state = cand.predicted ? "accepted_coasted" : "accepted";
        reason = "ok";
        return true;
    }

    bool tryRawFallback(const BallCandidate& raw_ball,
                        const MetadataPayload& raw_payload,
                        const PlayerContext& raw_player_ctx,
                        bool scene_reset,
                        double& jump_px,
                        int& gap_frames,
                        std::string& state,
                        std::string& reason,
                        BallCandidate& accepted_raw,
                        MetadataPayload& accepted_payload) {
        if (!raw_fallback_enabled_ || scene_reset || !raw_ball.present ||
            raw_ball.box.conf < raw_fallback_min_conf_) {
            resetRawFallback();
            return false;
        }

        jump_px = 0.0;
        gap_frames = 0;
        bool jump_plausible = true;
        if (have_last_accepted_) {
            gap_frames = (int)std::max<uint64_t>(1, frame_counter_ - last_accepted_frame_);
            jump_px = centerDistance(raw_ball.box, last_accepted_);
            jump_plausible = jump_px <= raw_fallback_max_jump_px_per_frame_ * (double)gap_frames;
        }

        const bool player_plausible = !player_aware_enabled_ ||
            raw_player_ctx.count == 0 ||
            raw_player_ctx.near_player;
        if (!jump_plausible && !player_plausible) {
            resetRawFallback();
            state = "rejected";
            reason = "raw_fallback_not_plausible";
            return false;
        }

        if (isSameRawPending(raw_ball.box)) {
            ++raw_pending_frames_;
        } else {
            raw_pending_ = raw_ball.box;
            raw_pending_frame_ = frame_counter_;
            raw_pending_frames_ = 1;
        }

        if (raw_pending_frames_ < std::max(1, raw_fallback_confirm_frames_)) {
            state = "pending";
            reason = "raw_fallback_confirm";
            return false;
        }

        accepted_raw = raw_ball;
        accepted_raw.source = "raw_fallback";
        accepted_raw.predicted = false;
        accepted_raw.coast_age = 0;
        accepted_raw.item["source"] = "raw_fallback";
        accepted_raw.item["predicted"] = false;
        accepted_raw.item["raw_fallback"] = true;
        accepted_payload = raw_payload;

        last_accepted_ = accepted_raw.box;
        have_last_accepted_ = true;
        last_accepted_frame_ = frame_counter_;
        have_pending_ = false;
        pending_frames_ = 0;
        resetRawFallback();
        ++stat_raw_fallbacks_;

        state = "accepted_raw_fallback";
        reason = "raw_fallback";
        return true;
    }

    Parameters buildOutput(const MetadataPayload& source_payload,
                           const BallCandidate& cand,
                           bool accepted,
                           const std::string& state,
                           const std::string& reason,
                           bool raw_present,
                           bool tracked_present,
                           bool scene_reset,
                           const PlayerContext& player_ctx,
                           double jump_px,
                           int gap_frames) const {
        Parameters out;
        out["coord_space"] = source_payload.present
            ? source_payload.md.value("coord_space", std::string("model"))
            : std::string("model");
        out["model_width"] = source_payload.model_w;
        out["model_height"] = source_payload.model_h;
        if (source_payload.present && source_payload.md.contains("models")) out["models"] = source_payload.md["models"];
        out["detections"] = Parameters::array();
        if (accepted && source_payload.present && source_payload.md.contains("trail")) {
            out["trail"] = source_payload.md["trail"];
        } else {
            out["trail"] = Parameters::array();
        }
        out["validation_state"] = state;
        out["reject_reason"] = reason;
        out["accepted"] = accepted;
        out["raw_present"] = raw_present;
        out["tracked_present"] = tracked_present;
        out["scene_reset"] = scene_reset;
        out["tracked_source"] = cand.source;
        out["coast_age_frames"] = cand.coast_age;
        out["pending_frames"] = pending_frames_;
        out["raw_fallback_pending_frames"] = raw_pending_frames_;
        out["nearest_player_distance_px"] = std::isfinite(player_ctx.nearest_distance) ? player_ctx.nearest_distance : -1.0;
        out["nearest_player_track_id"] = player_ctx.nearest_track_id;
        out["nearest_player_conf"] = player_ctx.nearest_conf;
        out["near_player"] = player_ctx.near_player;
        out["far_from_players"] = player_ctx.far_from_players;
        out["player_count"] = player_ctx.count;
        out["jump_px"] = jump_px;
        out["gap_frames"] = gap_frames;
        out["strategy"] = "confirm_jump_coast_scene_player_raw_fallback";

        if (accepted && cand.present) {
            Parameters item = cand.item;
            item["validated"] = true;
            item["validation_state"] = state;
            item["reject_reason"] = reason;
            item["jump_px"] = jump_px;
            item["gap_frames"] = gap_frames;
            item["coast_age_frames"] = cand.coast_age;
            item["nearest_player_distance_px"] = std::isfinite(player_ctx.nearest_distance) ? player_ctx.nearest_distance : -1.0;
            item["nearest_player_track_id"] = player_ctx.nearest_track_id;
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
        logstream << "  player rejects:  " << stat_player_rejects_;
        logstream << "  raw fallbacks:   " << stat_raw_fallbacks_;
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
        double player_model_w = tracked_payload.model_w;
        double player_model_h = tracked_payload.model_h;
        std::vector<DetectionBox> players = validPlayers(raw, player_model_w, player_model_h);
        BallCandidate tracked;
        BallCandidate raw_ball;
        const bool tracked_present = parseCandidate(tracked_payload, tracked);
        const bool raw_present = parseCandidate(raw_payload, raw_ball);
        PlayerContext tracked_player_ctx = playerContextFor(tracked, players);
        PlayerContext raw_player_ctx = playerContextFor(raw_ball, players);

        double jump_px = 0.0;
        int gap_frames = 0;
        std::string state;
        std::string reason;
        BallCandidate output_candidate = tracked;
        MetadataPayload output_payload = tracked_payload;
        PlayerContext output_player_ctx = tracked_player_ctx;
        bool accepted = validateCandidate(tracked, raw_present, scene_reset, tracked_player_ctx,
                                          jump_px, gap_frames, state, reason);
        if (!accepted) {
            double raw_jump_px = 0.0;
            int raw_gap_frames = 0;
            std::string raw_state;
            std::string raw_reason;
            BallCandidate fallback_candidate;
            MetadataPayload fallback_payload;
            if (tryRawFallback(raw_ball, raw_payload, raw_player_ctx, scene_reset,
                               raw_jump_px, raw_gap_frames, raw_state, raw_reason,
                               fallback_candidate, fallback_payload)) {
                accepted = true;
                output_candidate = fallback_candidate;
                output_payload = fallback_payload;
                output_player_ctx = raw_player_ctx;
                jump_px = raw_jump_px;
                gap_frames = raw_gap_frames;
                state = raw_state;
                reason = raw_reason;
            } else if (!raw_state.empty()) {
                state = raw_state;
                reason = raw_reason;
                jump_px = raw_jump_px;
                gap_frames = raw_gap_frames;
                output_player_ctx = raw_player_ctx;
            }
        }
        if (accepted) ++stat_accepted_;
        else ++stat_rejected_;

        Parameters out = buildOutput(output_payload, output_candidate, accepted, state, reason,
                                     raw_present, tracked_present, scene_reset, output_player_ctx,
                                     jump_px, gap_frames);
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
                      << " raw_pending=" << raw_pending_frames_
                      << " jump=" << jump_px
                      << " gap=" << gap_frames
                      << " near_player=" << (output_player_ctx.near_player ? 1 : 0)
                      << " nearest_player_dist=" << (std::isfinite(output_player_ctx.nearest_distance) ? output_player_ctx.nearest_distance : -1.0)
                      << " nearest_player_id=" << output_player_ctx.nearest_track_id
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
        if (params.count("player_metadata_key")) r->player_metadata_key_ = params["player_metadata_key"].get<std::string>();
        if (params.count("output_metadata_key")) r->output_metadata_key_ = params["output_metadata_key"].get<std::string>();
        if (params.count("ball_label")) r->ball_label_ = params["ball_label"].get<std::string>();
        if (params.count("player_labels")) {
            r->player_labels_.clear();
            for (const auto& l : params["player_labels"]) r->player_labels_.push_back(l.get<std::string>());
        }
        if (params.count("min_conf")) r->min_conf_ = params["min_conf"].get<double>();
        if (params.count("min_player_conf")) r->min_player_conf_ = params["min_player_conf"].get<double>();
        if (params.count("scene_diff_mean_abs_reset")) r->scene_diff_mean_abs_reset_ = params["scene_diff_mean_abs_reset"].get<double>();
        if (params.count("max_jump_px_per_frame")) r->max_jump_px_per_frame_ = params["max_jump_px_per_frame"].get<double>();
        if (params.count("player_near_distance_px")) r->player_near_distance_px_ = params["player_near_distance_px"].get<double>();
        if (params.count("player_far_distance_px")) r->player_far_distance_px_ = params["player_far_distance_px"].get<double>();
        if (params.count("player_jump_relax")) r->player_jump_relax_ = params["player_jump_relax"].get<double>();
        if (params.count("raw_fallback_min_conf")) r->raw_fallback_min_conf_ = params["raw_fallback_min_conf"].get<double>();
        if (params.count("raw_fallback_same_ball_px")) r->raw_fallback_same_ball_px_ = params["raw_fallback_same_ball_px"].get<double>();
        if (params.count("raw_fallback_max_jump_px_per_frame")) r->raw_fallback_max_jump_px_per_frame_ = params["raw_fallback_max_jump_px_per_frame"].get<double>();
        if (params.count("min_confirm_frames")) r->min_confirm_frames_ = params["min_confirm_frames"].get<int>();
        if (params.count("max_coast_frames")) r->max_coast_frames_ = params["max_coast_frames"].get<int>();
        if (params.count("raw_fallback_confirm_frames")) r->raw_fallback_confirm_frames_ = params["raw_fallback_confirm_frames"].get<int>();
        if (params.count("player_aware_enabled")) r->player_aware_enabled_ = params["player_aware_enabled"].get<bool>();
        if (params.count("raw_fallback_enabled")) r->raw_fallback_enabled_ = params["raw_fallback_enabled"].get<bool>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();

        r->min_conf_ = std::max(0.0, r->min_conf_);
        r->min_player_conf_ = std::max(0.0, r->min_player_conf_);
        r->max_jump_px_per_frame_ = std::max(1.0, r->max_jump_px_per_frame_);
        r->player_near_distance_px_ = std::max(1.0, r->player_near_distance_px_);
        r->player_far_distance_px_ = std::max(r->player_near_distance_px_, r->player_far_distance_px_);
        r->player_jump_relax_ = std::max(1.0, r->player_jump_relax_);
        r->raw_fallback_min_conf_ = std::max(0.0, r->raw_fallback_min_conf_);
        r->raw_fallback_same_ball_px_ = std::max(1.0, r->raw_fallback_same_ball_px_);
        r->raw_fallback_max_jump_px_per_frame_ = std::max(1.0, r->raw_fallback_max_jump_px_per_frame_);
        r->min_confirm_frames_ = std::max(1, r->min_confirm_frames_);
        r->max_coast_frames_ = std::max(0, r->max_coast_frames_);
        r->raw_fallback_confirm_frames_ = std::max(1, r->raw_fallback_confirm_frames_);
        return r;
    }
};

DECLNODE(ball_track_validator, BallTrackValidator)
