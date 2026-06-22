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
#include <unordered_map>
#include <vector>

class ProbableBallHandler : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
    using DetectionBox = avp_sport_reframe::DetectionBox;

    std::string ball_metadata_key_ = "yolo_ball";
    std::string player_metadata_key_ = "yolo_players";
    std::string handler_metadata_key_ = "ball_handler";
    std::string output_metadata_key_ = "probable_ball_handler";
    std::string ball_label_ = "basketball";
    std::vector<std::string> player_labels_ = {"Player"};
    std::string output_label_ = "ProbableBallHandler";
    std::string camera_shot_metadata_key_;
    std::string scene_diff_metadata_key_ = "scene_diff";

    double min_ball_conf_ = 0.04;
    double min_player_conf_ = 0.25;
    double max_distance_px_ = 120.0;
    double dropout_radius_rel_height_ = 9.0 / 32.0;
    double min_bbox_frac_in_circle_ = 0.05;
    double inv_d2_epsilon_px_ = 12.0;
    double previous_handler_bonus_ = 1.35;
    double score_decay_per_frame_ = 0.985;
    double min_output_conf_ = 0.02;
    double scene_diff_mean_abs_reset_ = 60.0;
    int tagging_window_frames_ = 12;
    int max_dropout_frames_ = 50;
    int max_candidates_ = 3;
    bool require_track_id_ = true;
    bool require_wide_shot_ = true;
    int debug_log_every_n_ = 0;

    uint64_t frame_counter_ = 0;
    bool have_last_ball_ = false;
    DetectionBox last_ball_;
    int dropout_age_ = 0;
    int last_handler_track_id_ = -1;
    std::unordered_map<int, double> sticky_scores_;

    uint64_t stat_frames_with_ball_ = 0;
    uint64_t stat_frames_with_probable_ = 0;
    uint64_t stat_scene_resets_ = 0;

    bool matchesPlayerLabel(const std::string& label) const {
        for (const auto& want : player_labels_) {
            if (label == want) return true;
        }
        return false;
    }

    void resetDropoutState(bool reset_last_ball) {
        dropout_age_ = 0;
        sticky_scores_.clear();
        if (reset_last_ball) {
            have_last_ball_ = false;
            last_handler_track_id_ = -1;
        }
    }

    void resetAllState() {
        frame_counter_ = 0;
        have_last_ball_ = false;
        last_ball_ = DetectionBox();
        dropout_age_ = 0;
        last_handler_track_id_ = -1;
        sticky_scores_.clear();
    }

    DetectionBox bestBall(const std::vector<DetectionBox>& balls) const {
        DetectionBox best;
        best.conf = -1.0;
        for (const auto& b : balls) {
            if (!avp_sport_reframe::labelMatches(b, ball_label_)) continue;
            if (b.conf < min_ball_conf_) continue;
            if (b.conf > best.conf) best = b;
        }
        return best;
    }

    int confirmedHandlerTrackId(const AVFrame* raw) const {
        double model_w = 960.0;
        double model_h = 544.0;
        std::vector<DetectionBox> handlers;
        if (!avp_sport_reframe::parseDetections(raw, handler_metadata_key_, handlers, model_w, model_h)) return -1;
        for (const auto& h : handlers) {
            if (h.has_label && h.label == "BallHandler" && h.has_track_id) return h.track_id;
        }
        return -1;
    }

    std::vector<DetectionBox> validPlayers(const AVFrame* raw, double& model_w, double& model_h) const {
        std::vector<DetectionBox> parsed;
        std::vector<DetectionBox> out;
        if (!avp_sport_reframe::parseDetections(raw, player_metadata_key_, parsed, model_w, model_h)) return out;
        for (const auto& p : parsed) {
            if (!p.has_label || !matchesPlayerLabel(p.label)) continue;
            if (p.conf < min_player_conf_) continue;
            if (require_track_id_ && !p.has_track_id) continue;
            out.push_back(p);
        }
        return out;
    }

    struct Candidate {
        DetectionBox player;
        double score = 0.0;
        double raw_inv_d2 = 0.0;
        double distance_px = 0.0;
        double circle_fraction = 0.0;
        bool sticky = false;
    };

    std::vector<Candidate> scoreCurrentPlayers(const std::vector<DetectionBox>& players, double model_h) const {
        std::vector<Candidate> candidates;
        if (!have_last_ball_) return candidates;

        const double bx = avp_sport_reframe::centerX(last_ball_);
        const double by = avp_sport_reframe::centerY(last_ball_);
        const double radius = std::max(1.0, model_h * dropout_radius_rel_height_);
        double sum = 0.0;

        for (const auto& p : players) {
            const double dist = avp_sport_reframe::pointToBoxDistance(bx, by, p);
            const double frac = avp_sport_reframe::boxAreaFractionInsideCircle(p, bx, by, radius);
            if (dist > max_distance_px_ && frac < min_bbox_frac_in_circle_) continue;

            const double d_eff = std::max(dist, inv_d2_epsilon_px_);
            double raw = 1.0 / (d_eff * d_eff);
            raw *= std::sqrt(std::max(0.0, p.conf));
            if (p.has_track_id && p.track_id == last_handler_track_id_) raw *= previous_handler_bonus_;
            if (!(raw > 0.0) || !std::isfinite(raw)) continue;

            Candidate c;
            c.player = p;
            c.raw_inv_d2 = raw;
            c.distance_px = dist;
            c.circle_fraction = frac;
            candidates.push_back(c);
            sum += raw;
        }

        if (sum > 0.0) {
            for (auto& c : candidates) c.score = c.raw_inv_d2 / sum;
        }
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            if (a.score != b.score) return a.score > b.score;
            return a.player.track_id < b.player.track_id;
        });
        return candidates;
    }

    std::vector<Candidate> buildOutputCandidates(const std::vector<DetectionBox>& players,
                                                 const std::vector<Candidate>& current) const {
        std::unordered_map<int, Candidate> current_by_id;
        for (const auto& c : current) {
            if (c.player.has_track_id) current_by_id[c.player.track_id] = c;
        }

        std::vector<Candidate> out;
        const double decay = std::pow(score_decay_per_frame_, std::max(0, dropout_age_ - 1));
        for (const auto& p : players) {
            if (!p.has_track_id) continue;
            auto it = sticky_scores_.find(p.track_id);
            if (it == sticky_scores_.end()) continue;
            Candidate c;
            c.player = p;
            c.score = it->second * decay;
            c.sticky = true;
            auto cur_it = current_by_id.find(p.track_id);
            if (cur_it != current_by_id.end()) {
                c.distance_px = cur_it->second.distance_px;
                c.circle_fraction = cur_it->second.circle_fraction;
                c.raw_inv_d2 = cur_it->second.raw_inv_d2;
            }
            if (c.score >= min_output_conf_) out.push_back(c);
        }

        if (out.empty() && dropout_age_ <= tagging_window_frames_) {
            for (auto c : current) {
                if (c.score >= min_output_conf_) out.push_back(c);
            }
        }

        std::sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) {
            if (a.score != b.score) return a.score > b.score;
            return a.player.track_id < b.player.track_id;
        });
        if ((int)out.size() > max_candidates_) out.resize((size_t)max_candidates_);
        return out;
    }

    void writeOutput(av::VideoFrame& frm,
                     double model_w,
                     double model_h,
                     const std::vector<Candidate>& candidates,
                     const std::string& state,
                     bool reset) {
        Parameters out = avp_sport_reframe::emptyYoloMetadata(model_w, model_h);
        out["state"] = state;
        out["dropout_age_frames"] = dropout_age_;
        out["last_ball_available"] = have_last_ball_;
        out["reset"] = reset;
        if (have_last_ball_) {
            out["last_ball_xyxy"] = {last_ball_.x1, last_ball_.y1, last_ball_.x2, last_ball_.y2};
        }

        int rank = 0;
        for (const auto& c : candidates) {
            Parameters det;
            det["label"] = output_label_;
            det["conf"] = avp_sport_reframe::clampDouble(c.score, 0.0, 1.0);
            det["xyxy"] = {c.player.x1, c.player.y1, c.player.x2, c.player.y2};
            if (c.player.has_track_id) det["track_id"] = c.player.track_id;
            det["candidate_rank"] = rank++;
            det["score_inv_d2"] = c.score;
            det["raw_inv_d2"] = c.raw_inv_d2;
            det["distance_px"] = c.distance_px;
            det["bbox_frac_in_last_ball_circle"] = c.circle_fraction;
            det["dropout_age_frames"] = dropout_age_;
            det["source"] = c.sticky ? "sticky_dropout_candidate" : "tagging_window_candidate";
            out["detections"].push_back(det);
        }

        av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), out.dump().c_str(), 0);
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    bool consumeEofIfPresent() override {
        return false;
    }

    ~ProbableBallHandler() {
        if (frame_counter_ == 0) return;
        logstream << "probable_ball_handler: === summary ===";
        logstream << "  total frames:            " << frame_counter_;
        logstream << "  frames with ball:        " << stat_frames_with_ball_;
        logstream << "  frames with probable:    " << stat_frames_with_probable_;
        logstream << "  scene/non-wide resets:   " << stat_scene_resets_;
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (isEofMarker(frm)) {
            resetAllState();
            this->sink_->put(frm);
            this->finished_ = true;
            return;
        }
        if (!frm) return;

        ++frame_counter_;
        const AVFrame* raw = frm.raw();
        double model_w = 960.0;
        double model_h = 544.0;
        bool reset = false;

        if (raw && avp_sport_reframe::sceneResetRequested(raw, scene_diff_metadata_key_, scene_diff_mean_abs_reset_)) {
            resetDropoutState(true);
            reset = true;
            ++stat_scene_resets_;
        }

        if (raw && require_wide_shot_ &&
            !avp_sport_reframe::cameraShotIsWide(raw, camera_shot_metadata_key_, true)) {
            resetDropoutState(true);
            reset = true;
            ++stat_scene_resets_;
        }

        std::vector<DetectionBox> balls;
        avp_sport_reframe::parseDetections(raw, ball_metadata_key_, balls, model_w, model_h);
        DetectionBox ball = bestBall(balls);
        const bool have_ball = ball.conf >= min_ball_conf_;

        const int handler_id = confirmedHandlerTrackId(raw);
        if (handler_id >= 0) last_handler_track_id_ = handler_id;

        if (have_ball) {
            last_ball_ = ball;
            have_last_ball_ = true;
            resetDropoutState(false);
            ++stat_frames_with_ball_;
            writeOutput(frm, model_w, model_h, {}, "ball_visible", reset);
            this->sink_->put(frm);
            return;
        }

        std::vector<Candidate> out_candidates;
        std::string state = have_last_ball_ ? "ball_dropout" : "no_ball_history";
        if (have_last_ball_) {
            dropout_age_++;
            if (dropout_age_ <= max_dropout_frames_) {
                std::vector<DetectionBox> players = validPlayers(raw, model_w, model_h);
                std::vector<Candidate> current = scoreCurrentPlayers(players, model_h);

                if (dropout_age_ <= tagging_window_frames_) {
                    for (const auto& c : current) {
                        if (!c.player.has_track_id) continue;
                        double& sticky = sticky_scores_[c.player.track_id];
                        sticky = std::max(sticky, c.score);
                    }
                }
                out_candidates = buildOutputCandidates(players, current);
            } else {
                state = "dropout_expired";
                resetDropoutState(true);
            }
        }

        if (!out_candidates.empty()) ++stat_frames_with_probable_;
        writeOutput(frm, model_w, model_h, out_candidates, state, reset);

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "probable_ball_handler: frame=" << frame_counter_
                      << " state=" << state
                      << " dropout_age=" << dropout_age_
                      << " sticky_ids=" << sticky_scores_.size()
                      << " emitted=" << out_candidates.size()
                      << " last_handler=" << last_handler_track_id_;
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<ProbableBallHandler> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<ProbableBallHandler>(edges, params);
        r->auto_eof_ = false;

        if (params.count("ball_metadata_key")) r->ball_metadata_key_ = params["ball_metadata_key"].get<std::string>();
        if (params.count("player_metadata_key")) r->player_metadata_key_ = params["player_metadata_key"].get<std::string>();
        if (params.count("handler_metadata_key")) r->handler_metadata_key_ = params["handler_metadata_key"].get<std::string>();
        if (params.count("output_metadata_key")) r->output_metadata_key_ = params["output_metadata_key"].get<std::string>();
        if (params.count("ball_label")) r->ball_label_ = params["ball_label"].get<std::string>();
        if (params.count("player_labels")) {
            r->player_labels_.clear();
            for (const auto& l : params["player_labels"]) r->player_labels_.push_back(l.get<std::string>());
        }
        if (params.count("output_label")) r->output_label_ = params["output_label"].get<std::string>();
        if (params.count("camera_shot_metadata_key")) r->camera_shot_metadata_key_ = params["camera_shot_metadata_key"].get<std::string>();
        if (params.count("scene_diff_metadata_key")) r->scene_diff_metadata_key_ = params["scene_diff_metadata_key"].get<std::string>();
        if (params.count("min_ball_conf")) r->min_ball_conf_ = params["min_ball_conf"].get<double>();
        if (params.count("min_player_conf")) r->min_player_conf_ = params["min_player_conf"].get<double>();
        if (params.count("max_distance_px")) r->max_distance_px_ = params["max_distance_px"].get<double>();
        if (params.count("dropout_radius_rel_height")) r->dropout_radius_rel_height_ = params["dropout_radius_rel_height"].get<double>();
        if (params.count("min_bbox_frac_in_circle")) r->min_bbox_frac_in_circle_ = params["min_bbox_frac_in_circle"].get<double>();
        if (params.count("inv_d2_epsilon_px")) r->inv_d2_epsilon_px_ = params["inv_d2_epsilon_px"].get<double>();
        if (params.count("previous_handler_bonus")) r->previous_handler_bonus_ = params["previous_handler_bonus"].get<double>();
        if (params.count("score_decay_per_frame")) r->score_decay_per_frame_ = params["score_decay_per_frame"].get<double>();
        if (params.count("min_output_conf")) r->min_output_conf_ = params["min_output_conf"].get<double>();
        if (params.count("scene_diff_mean_abs_reset")) r->scene_diff_mean_abs_reset_ = params["scene_diff_mean_abs_reset"].get<double>();
        if (params.count("tagging_window_frames")) r->tagging_window_frames_ = params["tagging_window_frames"].get<int>();
        if (params.count("max_dropout_frames")) r->max_dropout_frames_ = params["max_dropout_frames"].get<int>();
        if (params.count("max_candidates")) r->max_candidates_ = params["max_candidates"].get<int>();
        if (params.count("require_track_id")) r->require_track_id_ = params["require_track_id"].get<bool>();
        if (params.count("require_wide_shot")) r->require_wide_shot_ = params["require_wide_shot"].get<bool>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();

        r->tagging_window_frames_ = std::max(1, r->tagging_window_frames_);
        r->max_dropout_frames_ = std::max(r->tagging_window_frames_, r->max_dropout_frames_);
        r->max_candidates_ = std::max(1, r->max_candidates_);
        r->score_decay_per_frame_ = avp_sport_reframe::clampDouble(r->score_decay_per_frame_, 0.0, 1.0);
        return r;
    }
};

DECLNODE(probable_ball_handler, ProbableBallHandler)
