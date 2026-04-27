#include "../../node_common.hpp"

extern "C" {
#include <libavutil/dict.h>
}

#include <cstdint>
#include <string>
#include <unordered_map>

class PossessionTracker : public NodeSISO<av::VideoFrame, av::VideoFrame> {
    std::string handler_metadata_key_ = "ball_handler";
    std::string player_metadata_key_ = "yolo_players";
    std::string ball_metadata_key_ = "yolo_ball";
    std::string shot_events_metadata_key_ = "shot_events";
    std::string output_metadata_key_ = "possession_state";
    int hold_frames_ = 12;
    int change_confirm_frames_ = 12;
    bool clear_team_on_loose_ball_ = false;
    int debug_log_every_n_ = 0;

    uint64_t frame_counter_ = 0;
    int possession_id_ = 0;
    int frames_in_possession_ = 0;
    int current_handler_id_ = -1;
    int hold_remaining_ = 0;

    std::string current_team_;
    std::string last_controlled_team_;
    std::string candidate_team_;
    int candidate_frames_ = 0;

    std::unordered_map<int, std::string> track_team_cache_;

    static Parameters tryParse(const AVFrame* raw, const std::string& key) {
        if (!raw || !raw->metadata) return {};
        AVDictionaryEntry* entry = av_dict_get(raw->metadata, key.c_str(), nullptr, 0);
        if (!entry || !entry->value) return {};
        try { return Parameters::parse(entry->value); } catch (...) { return {}; }
    }

    template <typename T>
    static T getOr(const Parameters& obj, const char* key, const T& fallback) {
        if (!obj.is_object() || !obj.contains(key) || obj[key].is_null()) return fallback;
        try { return obj[key].get<T>(); } catch (...) { return fallback; }
    }

    static int parseHandlerTrackId(const Parameters& handler_md) {
        if (!handler_md.contains("detections") || !handler_md["detections"].is_array()) return -1;
        for (const auto& det : handler_md["detections"]) {
            if (!det.is_object()) continue;
            if (getOr<std::string>(det, "label", "") == "BallHandler") {
                return getOr<int>(det, "track_id", -1);
            }
        }
        return -1;
    }

    std::string resolvePlayerTeam(const Parameters& players_md, int track_id) {
        if (track_id < 0) return "";
        if (players_md.contains("detections") && players_md["detections"].is_array()) {
            for (const auto& det : players_md["detections"]) {
                if (!det.is_object()) continue;
                if (getOr<int>(det, "track_id", -1) != track_id) continue;
                if (getOr<std::string>(det, "label", "") != "Player") break;
                std::string team_ab = getOr<std::string>(det, "team_ab", "");
                if (team_ab == "A" || team_ab == "B") {
                    auto ins = track_team_cache_.emplace(track_id, team_ab);
                    return ins.first->second;
                }
                break;
            }
        }
        auto it = track_team_cache_.find(track_id);
        if (it != track_team_cache_.end()) return it->second;
        return "";
    }

    static bool ballDetected(const Parameters& ball_md) {
        return ball_md.contains("detections") && ball_md["detections"].is_array() && !ball_md["detections"].empty();
    }

    static bool shotInFlight(const Parameters& shot_events_md) {
        return getOr<bool>(shot_events_md, "in_flight", false);
    }

    void observeCandidateTeam(const std::string& team) {
        if (candidate_team_ == team) {
            ++candidate_frames_;
        } else {
            candidate_team_ = team;
            candidate_frames_ = 1;
        }
    }

    bool confirmCandidateTeam(const std::string& team) const {
        return candidate_team_ == team && candidate_frames_ >= change_confirm_frames_;
    }

    void resetState() {
        frame_counter_ = 0;
        possession_id_ = 0;
        frames_in_possession_ = 0;
        current_handler_id_ = -1;
        hold_remaining_ = 0;
        current_team_.clear();
        last_controlled_team_.clear();
        candidate_team_.clear();
        candidate_frames_ = 0;
        track_team_cache_.clear();
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (!frm) return;

        if (isEofMarker(frm)) {
            resetState();
            this->sink_->put(frm);
            return;
        }

        ++frame_counter_;

        const AVFrame* raw = frm.raw();
        Parameters handler_md = tryParse(raw, handler_metadata_key_);
        Parameters players_md = tryParse(raw, player_metadata_key_);
        Parameters ball_md = tryParse(raw, ball_metadata_key_);
        Parameters shot_events_md = tryParse(raw, shot_events_metadata_key_);

        const int handler_track_id = parseHandlerTrackId(handler_md);
        const std::string handler_team = resolvePlayerTeam(players_md, handler_track_id);
        const bool controlled = handler_track_id >= 0;
        const bool have_controlled_team = controlled && !handler_team.empty();
        const bool ball_visible = ballDetected(ball_md);
        const bool in_flight = shotInFlight(shot_events_md);

        std::string ball_state = "dead_or_unknown";
        std::string possessing_team;
        int out_possession_id = possession_id_;
        int out_frames_in_possession = frames_in_possession_;

        if (controlled) {
            ball_state = "controlled";
            current_handler_id_ = handler_track_id;
            hold_remaining_ = hold_frames_;

            if (have_controlled_team) {
                if (handler_team == current_team_) {
                    last_controlled_team_ = current_team_;
                    possessing_team = current_team_;
                    candidate_team_.clear();
                    candidate_frames_ = 0;
                    ++frames_in_possession_;
                    out_possession_id = possession_id_;
                    out_frames_in_possession = frames_in_possession_;
                } else {
                    observeCandidateTeam(handler_team);

                    if (!current_team_.empty()) {
                        possessing_team = current_team_;
                        out_possession_id = possession_id_;
                        out_frames_in_possession = frames_in_possession_;
                    }

                    if (confirmCandidateTeam(handler_team)) {
                        current_team_ = handler_team;
                        last_controlled_team_ = current_team_;
                        possessing_team = current_team_;
                        ++possession_id_;
                        frames_in_possession_ = candidate_frames_;
                        out_possession_id = possession_id_;
                        out_frames_in_possession = frames_in_possession_;
                    }
                }
            } else {
                candidate_team_.clear();
                candidate_frames_ = 0;
            }
        } else if (in_flight) {
            ball_state = "shot_in_air";
            current_handler_id_ = -1;
            candidate_team_.clear();
            candidate_frames_ = 0;

            possessing_team = !current_team_.empty() ? current_team_ : last_controlled_team_;
            if (!possessing_team.empty()) {
                ++frames_in_possession_;
                out_possession_id = possession_id_;
                out_frames_in_possession = frames_in_possession_;
            }
        } else if (ball_visible) {
            ball_state = "loose";
            current_handler_id_ = -1;
            candidate_team_.clear();
            candidate_frames_ = 0;

            if (hold_remaining_ > 0) {
                --hold_remaining_;
                possessing_team = !current_team_.empty() ? current_team_ : last_controlled_team_;
                if (!possessing_team.empty()) {
                    ++frames_in_possession_;
                    out_possession_id = possession_id_;
                    out_frames_in_possession = frames_in_possession_;
                }
            } else if (clear_team_on_loose_ball_) {
                current_team_.clear();
                out_possession_id = possession_id_;
                out_frames_in_possession = 0;
            } else {
                possessing_team = !current_team_.empty() ? current_team_ : last_controlled_team_;
                out_possession_id = possession_id_;
                out_frames_in_possession = frames_in_possession_;
            }
        } else {
            current_handler_id_ = -1;
            candidate_team_.clear();
            candidate_frames_ = 0;

            if (hold_remaining_ > 0) {
                --hold_remaining_;
                possessing_team = !current_team_.empty() ? current_team_ : last_controlled_team_;
                if (!possessing_team.empty()) {
                    out_possession_id = possession_id_;
                    out_frames_in_possession = frames_in_possession_;
                }
            } else if (clear_team_on_loose_ball_) {
                current_team_.clear();
                out_possession_id = possession_id_;
                out_frames_in_possession = 0;
            } else {
                possessing_team = !current_team_.empty() ? current_team_ : last_controlled_team_;
                out_possession_id = possession_id_;
                out_frames_in_possession = frames_in_possession_;
            }
        }

        Parameters out_md;
        out_md["ball_state"] = ball_state;
        out_md["possession_id"] = out_possession_id;
        out_md["frames_in_possession"] = out_frames_in_possession;
        if (!possessing_team.empty()) out_md["possessing_team"] = possessing_team;
        if (controlled) {
            out_md["handler_id"] = handler_track_id;
            if (!handler_team.empty()) out_md["handler_team"] = handler_team;
        }

        av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), out_md.dump().c_str(), 0);

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "possession_tracker: frame=" << frame_counter_
                      << " state=" << ball_state
                      << " team=" << (possessing_team.empty() ? "none" : possessing_team)
                      << " handler=" << (controlled ? std::to_string(handler_track_id) : "none")
                      << " possession_id=" << out_possession_id
                      << " frames=" << out_frames_in_possession;
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<PossessionTracker> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<PossessionTracker>(edges, params);

        if (params.count("handler_metadata_key")) r->handler_metadata_key_ = params["handler_metadata_key"].get<std::string>();
        if (params.count("player_metadata_key")) r->player_metadata_key_ = params["player_metadata_key"].get<std::string>();
        if (params.count("ball_metadata_key")) r->ball_metadata_key_ = params["ball_metadata_key"].get<std::string>();
        if (params.count("shot_events_metadata_key")) r->shot_events_metadata_key_ = params["shot_events_metadata_key"].get<std::string>();
        if (params.count("output_metadata_key")) r->output_metadata_key_ = params["output_metadata_key"].get<std::string>();
        if (params.count("hold_frames")) r->hold_frames_ = params["hold_frames"].get<int>();
        if (params.count("change_confirm_frames")) r->change_confirm_frames_ = params["change_confirm_frames"].get<int>();
        if (params.count("clear_team_on_loose_ball")) r->clear_team_on_loose_ball_ = params["clear_team_on_loose_ball"].get<bool>();
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();

        return r;
    }
};

DECLNODE(possession_tracker, PossessionTracker)
