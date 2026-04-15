#include "../../node_common.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include <cmath>
#include <limits>
#include <string>
#include <vector>

class BallHandler : public NodeSISO<av::VideoFrame, av::VideoFrame> {

    std::string ball_metadata_key_ = "yolo_ball";
    std::string player_metadata_key_ = "yolo_players";
    std::string output_metadata_key_ = "ball_handler";
    std::string ball_label_ = "basketball";
    std::vector<std::string> player_labels_ = {"Player"};
    float max_distance_px_ = 35.0f;       // max ball-to-player edge distance in model space (~1 ball width)
    float player_min_conf_ = 0.25f;
    int hysteresis_frames_ = 12;          // hold handler tag for N frames (~400ms) — survives dribble bounce
    std::string shot_metadata_key_;       // optional: read shot_info for stats
    int debug_log_every_n_ = 0;

    // State
    int current_handler_track_id_ = -1;
    int handler_hold_counter_ = 0;
    uint64_t frame_counter_ = 0;

    // Stats
    uint64_t stat_frames_with_handler_ = 0;
    uint64_t stat_frames_with_ball_ = 0;
    uint64_t stat_handler_changes_ = 0;
    uint64_t stat_wide_frames_ = 0;
    uint64_t stat_handler_in_wide_ = 0;
    int stat_last_handler_id_ = -1;

    struct Detection {
        float x1, y1, x2, y2;
        float conf;
        std::string label;
        int track_id;
        bool has_track_id;
    };

    static float centerX(const Detection& d) { return (d.x1 + d.x2) * 0.5f; }
    static float centerY(const Detection& d) { return (d.y1 + d.y2) * 0.5f; }
    static float footY(const Detection& d) { return d.y2; }  // bottom of bbox

    bool matchesPlayerLabel(const std::string& label) const {
        for (const auto& pl : player_labels_) {
            if (label == pl) return true;
        }
        return false;
    }

    // Distance from ball center to nearest point on player bbox
    static float ballToPlayerDist(const Detection& ball, const Detection& player) {
        float bx = centerX(ball);
        float by = centerY(ball);
        float dx = std::max({0.0f, player.x1 - bx, bx - player.x2});
        float dy = std::max({0.0f, player.y1 - by, by - player.y2});
        return std::sqrt(dx * dx + dy * dy);
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    ~BallHandler() {
        if (frame_counter_ == 0) return;
        logstream << "ball_handler: === summary ===";
        logstream << "  total frames:          " << frame_counter_;
        logstream << "  wide frames:           " << stat_wide_frames_
                  << " (" << (100.0 * stat_wide_frames_ / frame_counter_) << "%)";
        logstream << "  frames with ball:      " << stat_frames_with_ball_
                  << " (" << (100.0 * stat_frames_with_ball_ / frame_counter_) << "%)";
        logstream << "  frames with handler:   " << stat_frames_with_handler_
                  << " (" << (100.0 * stat_frames_with_handler_ / frame_counter_) << "%)";
        if (stat_wide_frames_ > 0) {
            logstream << "  handler in wide:       " << stat_handler_in_wide_
                      << " (" << (100.0 * stat_handler_in_wide_ / stat_wide_frames_) << "% of wide)";
        }
        logstream << "  handler changes:       " << stat_handler_changes_;
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm) return;

        if (isEofMarker(frm)) {
            current_handler_track_id_ = -1;
            handler_hold_counter_ = 0;
            frame_counter_ = 0;
            this->sink_->put(frm);
            return;
        }

        ++frame_counter_;

        const AVFrame* raw = frm.raw();
        if (!raw || !raw->metadata) {
            this->sink_->put(frm);
            return;
        }

        // Read shot type for stats
        bool is_wide = true;
        if (!shot_metadata_key_.empty()) {
            AVDictionaryEntry* shot_entry = av_dict_get(raw->metadata, shot_metadata_key_.c_str(), nullptr, 0);
            if (shot_entry && shot_entry->value) {
                try {
                    Parameters shot_md = Parameters::parse(shot_entry->value);
                    is_wide = shot_md.value("shot_type", std::string()) == "wide";
                } catch (...) {}
            }
        }
        if (is_wide) stat_wide_frames_++;

        // Skip handler detection during non-wide shots
        if (!is_wide) {
            current_handler_track_id_ = -1;
            handler_hold_counter_ = 0;
            this->sink_->put(frm);
            return;
        }

        // Parse ball detection
        Detection ball_det{};
        bool have_ball = false;
        float model_w = 960.0f, model_h = 544.0f;
        {
            AVDictionaryEntry* entry = av_dict_get(raw->metadata, ball_metadata_key_.c_str(), nullptr, 0);
            if (entry && entry->value) {
                try {
                    Parameters md = Parameters::parse(entry->value);
                    model_w = md.value("model_width", 960.0);
                    model_h = md.value("model_height", 544.0);
                    if (md.contains("detections") && md["detections"].is_array()) {
                        for (const auto& det : md["detections"]) {
                            if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) continue;
                            std::string lbl = det.value("label", std::string());
                            if (!ball_label_.empty() && lbl != ball_label_) continue;
                            float conf = det.value("conf", 0.0f);
                            ball_det.x1 = det["xyxy"][0].get<float>();
                            ball_det.y1 = det["xyxy"][1].get<float>();
                            ball_det.x2 = det["xyxy"][2].get<float>();
                            ball_det.y2 = det["xyxy"][3].get<float>();
                            ball_det.conf = conf;
                            ball_det.label = lbl;
                            have_ball = true;
                            break;  // take first (highest conf from tracker)
                        }
                    }
                } catch (...) {}
            }
        }

        // Parse player detections
        std::vector<Detection> players;
        {
            AVDictionaryEntry* entry = av_dict_get(raw->metadata, player_metadata_key_.c_str(), nullptr, 0);
            if (entry && entry->value) {
                try {
                    Parameters md = Parameters::parse(entry->value);
                    if (md.contains("detections") && md["detections"].is_array()) {
                        for (const auto& det : md["detections"]) {
                            if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) continue;
                            std::string lbl = det.value("label", std::string());
                            if (!matchesPlayerLabel(lbl)) continue;
                            float conf = det.value("conf", 0.0f);
                            if (conf < player_min_conf_) continue;

                            Detection p;
                            p.x1 = det["xyxy"][0].get<float>();
                            p.y1 = det["xyxy"][1].get<float>();
                            p.x2 = det["xyxy"][2].get<float>();
                            p.y2 = det["xyxy"][3].get<float>();
                            p.conf = conf;
                            p.label = lbl;
                            p.track_id = det.value("track_id", -1);
                            p.has_track_id = det.contains("track_id") && p.track_id >= 0;
                            players.push_back(p);
                        }
                    }
                } catch (...) {}
            }
        }

        // Find nearest player to ball
        int best_idx = -1;
        float best_dist = std::numeric_limits<float>::max();
        if (have_ball) {
            for (int i = 0; i < (int)players.size(); i++) {
                float dist = ballToPlayerDist(ball_det, players[i]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = i;
                }
            }
        }

        // Determine handler with hysteresis
        bool found_handler = (best_idx >= 0 && best_dist <= max_distance_px_);
        int new_handler_track_id = found_handler && players[best_idx].has_track_id
                                   ? players[best_idx].track_id : -1;

        if (found_handler) {
            current_handler_track_id_ = new_handler_track_id;
            handler_hold_counter_ = hysteresis_frames_;
        } else if (handler_hold_counter_ > 0) {
            handler_hold_counter_--;
            // Keep current_handler_track_id_ during hold
            // Try to find the held player by track_id
            if (current_handler_track_id_ >= 0) {
                found_handler = false;
                for (int i = 0; i < (int)players.size(); i++) {
                    if (players[i].has_track_id && players[i].track_id == current_handler_track_id_) {
                        best_idx = i;
                        found_handler = true;
                        break;
                    }
                }
                if (!found_handler) {
                    // Player disappeared, drop hold
                    handler_hold_counter_ = 0;
                    current_handler_track_id_ = -1;
                }
            }
        } else {
            current_handler_track_id_ = -1;
        }

        bool emit_handler = (current_handler_track_id_ >= 0 && best_idx >= 0);

        // Stats
        if (have_ball) stat_frames_with_ball_++;
        if (emit_handler) {
            stat_frames_with_handler_++;
            if (is_wide) stat_handler_in_wide_++;
            if (current_handler_track_id_ != stat_last_handler_id_) {
                stat_handler_changes_++;
                stat_last_handler_id_ = current_handler_track_id_;
            }
        }

        // Write output metadata with single BallHandler detection
        Parameters out_md;
        out_md["coord_space"] = "model";
        out_md["model_width"] = model_w;
        out_md["model_height"] = model_h;
        out_md["detections"] = Parameters::array();

        if (emit_handler) {
            const auto& p = players[best_idx];
            Parameters det;
            det["label"] = "BallHandler";
            det["conf"] = p.conf;
            det["xyxy"] = {p.x1, p.y1, p.x2, p.y2};
            det["track_id"] = p.track_id;
            det["ball_distance"] = best_dist;
            out_md["detections"].push_back(det);
        }

        av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), out_md.dump().c_str(), 0);

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "ball_handler: frame=" << frame_counter_
                      << " ball=" << (have_ball ? "yes" : "no")
                      << " players=" << players.size()
                      << " handler=" << (emit_handler ? std::to_string(current_handler_track_id_) : "none")
                      << " dist=" << (best_idx >= 0 ? best_dist : -1.0f)
                      << " hold=" << handler_hold_counter_;
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<BallHandler> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<BallHandler>(edges, params);
        r->auto_eof_ = false;

        if (params.count("ball_metadata_key")) r->ball_metadata_key_ = params["ball_metadata_key"].get<std::string>();
        if (params.count("player_metadata_key")) r->player_metadata_key_ = params["player_metadata_key"].get<std::string>();
        if (params.count("output_metadata_key")) r->output_metadata_key_ = params["output_metadata_key"].get<std::string>();
        if (params.count("ball_label")) r->ball_label_ = params["ball_label"].get<std::string>();
        if (params.count("player_labels")) {
            r->player_labels_.clear();
            for (const auto& l : params["player_labels"]) r->player_labels_.push_back(l.get<std::string>());
        }
        if (params.count("max_distance_px")) r->max_distance_px_ = params["max_distance_px"];
        if (params.count("player_min_conf")) r->player_min_conf_ = params["player_min_conf"];
        if (params.count("hysteresis_frames")) r->hysteresis_frames_ = params["hysteresis_frames"];
        if (params.count("shot_metadata_key")) r->shot_metadata_key_ = params["shot_metadata_key"].get<std::string>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"];

        return r;
    }
};

DECLNODE(ball_handler, BallHandler)
