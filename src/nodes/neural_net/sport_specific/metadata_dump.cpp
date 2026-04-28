#include "../../node_common.hpp"
#include "../common/yolo_side_data.hpp"

extern "C" {
#include <libavutil/dict.h>
#include <libavutil/frame.h>
}

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class MetadataDump : public NodeSISO<av::VideoFrame, av::VideoFrame>, public ReportsFinishByFlag {
    std::string player_metadata_key_ = "yolo_players";
    std::string ball_metadata_key_ = "yolo_ball";
    std::string ball_handler_metadata_key_ = "ball_handler";
    std::string possession_metadata_key_ = "possession_state";
    std::string court_zone_metadata_key_ = "court_zone";
    std::string camera_shot_metadata_key_ = "camera_shot_info";
    std::string shot_events_metadata_key_ = "shot_events";
    std::string scoreboard_metadata_key_ = "scoreboard";
    std::string game_state_metadata_key_ = "game_state";
    std::string viewport_metadata_key_ = "smoothed_crop_viewport_v1";
    std::string player_seg_metadata_key_ = "yolo_players_seg";
    std::string court_seg_metadata_key_ = "yolo_seg";
    std::string output_metadata_key_ = "frame_dump";
    std::string output_file_;
    std::string output_file_court_;
    std::string output_file_outlines_;
    std::string output_file_trail_;
    std::string output_file_events_;
    std::string output_file_summary_;
    int summary_update_every_n_ = 1;
    std::string video_label_;
    int fps_ = 0;
    int schema_ = 3;
    int dump_every_n_ = 1;
    int debug_log_every_n_ = 0;
    int shot_result_wait_frames_ = 25;
    int score_change_confirm_frames_ = 6;
    int score_change_max_delta_ = 4;
    bool include_ocr_game_state_ = false;
    bool emit_ocr_clock_events_ = false;

    int court_seg_slot_ = 0;
    int player_seg_slot_ = 1;
    float mask_threshold_ = 0.5f;
    int contour_simplify_step_ = 2;

    uint64_t frame_counter_ = 0;
    std::string cached_json_;

    // Scale state — set on first frame
    bool header_written_ = false;
    int source_w_ = 0;
    int source_h_ = 0;
    float model_w_ = 960.0f;
    float model_h_ = 544.0f;
    float scale_x_ = 1.0f;
    float scale_y_ = 1.0f;

    // Event-detection state
    int prev_total_releases_ = 0;
    int prev_total_arrivals_ = 0;
    int prev_handler_id_ = -1;
    int last_known_handler_id_ = -1;
    std::string last_known_handler_team_;

    int pending_new_handler_id_ = -1;
    std::string pending_new_handler_team_;
    int pending_new_handler_frames_ = 0;
    int possession_change_confirm_frames_ = 6;

    // Clock-movement tracking
    int last_game_clock_sec_ = -1;
    float last_game_clock_change_t_ = -1.0f;
    bool clock_currently_stopped_ = false;
    float clock_stopped_threshold_s_ = 2.5f;

    // Clip-summary counters
    std::map<std::string, int> camera_shot_frames_;
    std::map<std::string, int> possessions_by_team_;
    int shots_made_ = 0;
    int shots_missed_ = 0;
    int shots_unknown_ = 0;
    int shot_attempts_total_ = 0;
    int score_events_total_ = 0;
    int score_events_2pt_ = 0;
    int score_events_3pt_ = 0;
    int scoreboard_verified_shots_ = 0;
    int scoreboard_unverified_shots_ = 0;
    int detector_score_disagreements_ = 0;
    int score_ocr_regressions_ = 0;
    int score_ocr_rejected_ = 0;
    int first_game_clock_sec_ = -1;
    int latest_game_clock_sec_ = -1;

    // Game-shape aggregates derived from confirmed score changes.
    int lead_changes_ = 0;
    int largest_lead_ = 0;
    std::string current_leader_;       // "A", "B", or "" while tied
    int longest_run_points_ = 0;
    std::string current_run_side_;     // "team_a" / "team_b"
    int current_run_points_ = 0;
    uint64_t total_possession_frames_ = 0;
    std::string prev_possessing_team_;
    int prev_period_num_ = -1;
    int prev_shot_clock_sec_ = -1;
    std::string prev_camera_shot_;

    int team_identity_evidence_[2][2] = {{0, 0}, {0, 0}}; // visual A/B x scoreboard team_a/team_b
    bool team_identity_locked_ = false;
    bool team_identity_just_locked_ = false;
    int team_identity_lock_frame_ = -1;
    std::string visual_to_scoreboard_[2];
    std::string scoreboard_to_visual_[2];
    int team_identity_lock_min_evidence_ = 2;
    int team_identity_lock_min_margin_ = 1;

    std::string locked_scoreboard_team_[2];
    std::string scoreboard_team_candidate_[2];
    int scoreboard_team_candidate_hits_ = 0;
    int scoreboard_team_name_lock_hits_ = 1;

    struct ScoreState {
        int a = -1;
        int b = -1;
        int period = -1;
        int clock_sec = -1;
        int shot_clock_sec = -1;
    };

    bool have_confirmed_score_ = false;
    ScoreState confirmed_score_;
    ScoreState score_candidate_;
    int score_candidate_frames_ = 0;
    uint64_t score_candidate_first_frame_ = 0;
    int64_t score_candidate_first_pts_ = 0;

    struct PendingRelease {
        uint64_t frame = 0;
        int64_t pts = 0;
        int player_id = -1;
        std::string shooting_team;
        std::string attempt_type;
        int attempt_points = 0;
        ScoreState score_at_release;
    };
    bool have_last_release_ = false;
    PendingRelease last_release_;

    struct PendingShot {
        uint64_t release_frame = 0;
        uint64_t arrival_frame = 0;
        int64_t arrival_pts = 0;
        int player_id = -1;
        std::string shooting_team;
        std::string attempt_type_detected;
        int points_detected = 0;
        ScoreState score_at_release;
        bool detector_result_seen = false;
        std::string detector_result;
        std::string detector_source;
        float detector_confidence = -1.0f;
        int frames_waited = 0;
    };
    bool have_pending_shot_ = false;
    PendingShot pending_shot_;

    struct ShotRecord {
        int release_frame = -1;
        int arrival_frame = -1;
        int hoop_dist = -1;
        int release_hoop_dist = -1;
        std::string attempt_type;   // "2pt" | "3pt" | "unknown"
        int game_clock_sec = -1;    // at release
        int period_num = -1;        // at release
        std::string result;
        std::string result_source;
        int points = 0;
        bool scoreboard_verified = false;
        int points_detected = 0;
        std::string attempt_type_detected;
        int points_scoreboard = 0;
        std::string attempt_type_scoreboard;
        std::string score_delta_side;
        std::string visual_team;
        std::string scoreboard_team_abbrev;
        std::string scoreboard_team_name;
        int score_before_a = -1;
        int score_before_b = -1;
        int score_after_a = -1;
        int score_after_b = -1;
        std::string detector_result;
        bool detector_disagreement = false;
        bool attempt_type_corrected_by_scoreboard = false;
    };

    struct ShotResultData {
        std::string result;
        std::string source;
        int points = 0;
        std::string attempt_type;
        bool scoreboard_verified = false;
        int points_detected = 0;
        std::string attempt_type_detected;
        int points_scoreboard = 0;
        std::string attempt_type_scoreboard;
        std::string score_delta_side;
        std::string visual_team;
        std::string scoreboard_team_abbrev;
        std::string scoreboard_team_name;
        ScoreState score_before;
        ScoreState score_after;
        std::string detector_result;
        bool detector_disagreement = false;
        bool attempt_type_corrected_by_scoreboard = false;
    };

    struct Touch {
        int track_id = -1;
        int start_frame = 0;
        int frames = 0;
    };

    struct PossessionAcc {
        int possession_id = -1;
        std::string team;
        uint64_t start_frame = 0;
        int64_t start_pts = 0;
        uint64_t end_frame = 0;
        int64_t end_pts = 0;
        int total_frames = 0;
        int frames_controlled = 0;
        int frames_loose = 0;
        int frames_in_flight = 0;
        std::vector<int> handler_ids;
        std::vector<Touch> touches;
        std::vector<std::string> zones_visited;
        std::map<std::string, int> zone_frames;
        std::vector<ShotRecord> shots;
        // Game-state snapshots — populated from OCR'd game state on the
        // first/last frame of the possession. -1 means "not observed".
        int period_at_start = -1;
        int period_at_end = -1;
        bool period_changed_during = false;
        int clock_at_start = -1;
        int clock_at_end = -1;
        int shot_clock_at_end = -1;
        int score_a_at_start = -1;
        int score_b_at_start = -1;
        int score_a_at_end = -1;
        int score_b_at_end = -1;
    };
    bool poss_active_ = false;
    PossessionAcc poss_acc_;
    bool have_deferred_poss_ = false;
    PossessionAcc deferred_poss_;
    std::string output_file_possessions_;
    std::string output_file_pbp_;
    bool emit_handler_change_events_ = false;
    // Frames at which a brief same-team handler flicker is folded back into
    // the prior touch instead of opening a new one. Default tuned for 25 fps:
    // 6 frames ≈ 240 ms — shorter than any real pass.
    int touch_merge_gap_frames_ = 6;

    struct JsonFileWriter {
        std::ofstream out;
        bool opened = false;

        bool open(const std::string& path) {
            if (path.empty()) return false;
            out.open(path, std::ios::out | std::ios::trunc);
            if (!out) return false;
            opened = true;
            return true;
        }
        void writeLine(const std::string& json) {
            if (!opened) return;
            out << json << '\n';
            out.flush();
        }
        void close() {
            if (!opened) return;
            out.close();
            opened = false;
        }
    };

    JsonFileWriter file_main_;
    JsonFileWriter file_court_;
    JsonFileWriter file_outlines_;
    JsonFileWriter file_trail_;
    JsonFileWriter file_events_;
    JsonFileWriter file_possessions_;
    JsonFileWriter file_pbp_;

    static Parameters tryParse(const AVFrame* raw, const std::string& key) {
        if (!raw || !raw->metadata) return {};
        AVDictionaryEntry* entry = av_dict_get(raw->metadata, key.c_str(), nullptr, 0);
        if (!entry || !entry->value) return {};
        try { return Parameters::parse(entry->value); } catch (...) { return {}; }
    }

    static int ri(float v) { return (int)std::round(v); }

    int scaleX(float v) const { return ri(v * scale_x_); }
    int scaleY(float v) const { return ri(v * scale_y_); }
    int scaleDist(float v) const { return ri(v * 0.5f * (scale_x_ + scale_y_)); }

    template <typename T>
    static T getOr(const Parameters& obj, const char* key, const T& fallback) {
        if (!obj.is_object() || !obj.contains(key) || obj[key].is_null()) return fallback;
        try {
            return obj[key].get<T>();
        } catch (...) {
            return fallback;
        }
    }

    static bool haveScore(const ScoreState& s) {
        return s.a >= 0 && s.b >= 0;
    }

    static bool sameScore(const ScoreState& a, const ScoreState& b) {
        return a.a == b.a && a.b == b.b;
    }

    static std::string attemptTypeForPoints(int points) {
        if (points == 2) return "2pt";
        if (points == 3) return "3pt";
        return "unknown";
    }

    static int visualTeamIndex(const std::string& team) {
        if (team == "A") return 0;
        if (team == "B") return 1;
        return -1;
    }

    static int scoreboardSideIndex(const std::string& side) {
        if (side == "team_a") return 0;
        if (side == "team_b") return 1;
        return -1;
    }

    static std::string visualTeamName(int idx) {
        if (idx == 0) return "A";
        if (idx == 1) return "B";
        return "?";
    }

    static std::string scoreboardSideName(int idx) {
        if (idx == 0) return "team_a";
        if (idx == 1) return "team_b";
        return "";
    }

    static bool isKnownNbaAbbrev(const std::string& abbrev) {
        static const std::unordered_set<std::string> nba = {
            "ATL", "BOS", "BKN", "BRK", "CHA", "CHI", "CLE", "DAL", "DEN", "DET",
            "GS", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN",
            "NOP", "NO", "NY", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC",
            "SA", "SAS", "TOR", "UTA", "WAS"
        };
        return nba.count(abbrev) > 0;
    }

    static std::string nbaFullName(const std::string& abbrev) {
        static const std::unordered_map<std::string, std::string> names = {
            {"ATL", "Atlanta Hawks"}, {"BOS", "Boston Celtics"}, {"BKN", "Brooklyn Nets"},
            {"CHA", "Charlotte Hornets"}, {"CHI", "Chicago Bulls"}, {"CLE", "Cleveland Cavaliers"},
            {"DAL", "Dallas Mavericks"}, {"DEN", "Denver Nuggets"}, {"DET", "Detroit Pistons"},
            {"GSW", "Golden State Warriors"}, {"HOU", "Houston Rockets"}, {"IND", "Indiana Pacers"},
            {"LAC", "LA Clippers"}, {"LAL", "Los Angeles Lakers"}, {"MEM", "Memphis Grizzlies"},
            {"MIA", "Miami Heat"}, {"MIL", "Milwaukee Bucks"}, {"MIN", "Minnesota Timberwolves"},
            {"NOP", "New Orleans Pelicans"}, {"NYK", "New York Knicks"}, {"OKC", "Oklahoma City Thunder"},
            {"ORL", "Orlando Magic"}, {"PHI", "Philadelphia 76ers"}, {"PHX", "Phoenix Suns"},
            {"POR", "Portland Trail Blazers"}, {"SAC", "Sacramento Kings"}, {"SAS", "San Antonio Spurs"},
            {"TOR", "Toronto Raptors"}, {"UTA", "Utah Jazz"}, {"WAS", "Washington Wizards"}
        };
        auto it = names.find(abbrev);
        return it == names.end() ? std::string() : it->second;
    }

    static std::string cleanNbaAbbrev(const Parameters& obj, const char* key) {
        if (!obj.is_object() || !obj.contains(key) || !obj[key].is_string()) return {};
        std::string s = obj[key].get<std::string>();
        std::string up;
        up.reserve(s.size());
        for (char c : s) {
            if (std::isalpha((unsigned char)c)) up.push_back((char)std::toupper((unsigned char)c));
        }
        return isKnownNbaAbbrev(up) ? up : std::string();
    }

    std::string scoreboardTeamAbbrevForSide(const std::string& side) const {
        int idx = scoreboardSideIndex(side);
        if (idx < 0) return {};
        return locked_scoreboard_team_[idx];
    }

    std::string scoreboardTeamNameForSide(const std::string& side) const {
        return nbaFullName(scoreboardTeamAbbrevForSide(side));
    }

    void maybeLockScoreboardTeamNames(const Parameters& game_state_md) {
        std::string a = cleanNbaAbbrev(game_state_md, "team_a_abbrev");
        std::string b = cleanNbaAbbrev(game_state_md, "team_b_abbrev");
        if (a.empty() || b.empty() || a == b) return;
        if (!locked_scoreboard_team_[0].empty() && !locked_scoreboard_team_[1].empty()) return;

        if (scoreboard_team_candidate_[0] == a && scoreboard_team_candidate_[1] == b) {
            ++scoreboard_team_candidate_hits_;
        } else {
            scoreboard_team_candidate_[0] = a;
            scoreboard_team_candidate_[1] = b;
            scoreboard_team_candidate_hits_ = 1;
        }

        if (scoreboard_team_candidate_hits_ >= scoreboard_team_name_lock_hits_) {
            locked_scoreboard_team_[0] = a;
            locked_scoreboard_team_[1] = b;
        }
    }

    int teamIdentityEvidenceTotal() const {
        int total = 0;
        for (int v = 0; v < 2; ++v) {
            for (int s = 0; s < 2; ++s) total += team_identity_evidence_[v][s];
        }
        return total;
    }

    Parameters teamIdentityJson(bool include_evidence = true) const {
        Parameters out = Parameters::object();
        out["locked"] = team_identity_locked_;
        out["source"] = "scoreboard_verified_shots";
        const bool scoreboard_names_locked =
            !locked_scoreboard_team_[0].empty() && !locked_scoreboard_team_[1].empty();
        out["scoreboard_team_names_locked"] = scoreboard_names_locked;
        if (team_identity_locked_) {
            Parameters v2s = Parameters::object();
            Parameters s2v = Parameters::object();
            Parameters visual = Parameters::object();
            Parameters scoreboard = Parameters::object();
            for (int v = 0; v < 2; ++v) {
                const std::string vt = visualTeamName(v);
                const std::string side = visual_to_scoreboard_[v];
                if (side.empty()) continue;
                v2s[vt] = side;
                Parameters vo = Parameters::object();
                vo["scoreboard_side"] = side;
                const std::string abbr = scoreboardTeamAbbrevForSide(side);
                if (!abbr.empty()) {
                    vo["abbrev"] = abbr;
                    const std::string full = nbaFullName(abbr);
                    if (!full.empty()) vo["name"] = full;
                }
                visual[vt] = vo;
            }
            for (int s = 0; s < 2; ++s) {
                const std::string side = scoreboardSideName(s);
                const std::string vt = scoreboard_to_visual_[s];
                if (vt.empty()) continue;
                s2v[side] = vt;
                Parameters so = Parameters::object();
                so["visual_team"] = vt;
                if (!locked_scoreboard_team_[s].empty()) {
                    so["abbrev"] = locked_scoreboard_team_[s];
                    const std::string full = nbaFullName(locked_scoreboard_team_[s]);
                    if (!full.empty()) so["name"] = full;
                }
                scoreboard[side] = so;
            }
            if (!v2s.empty()) out["visual_to_scoreboard"] = v2s;
            if (!s2v.empty()) out["scoreboard_to_visual"] = s2v;
            if (!visual.empty()) out["visual"] = visual;
            if (!scoreboard.empty()) out["scoreboard"] = scoreboard;
            if (team_identity_lock_frame_ >= 0) out["locked_frame"] = team_identity_lock_frame_;
        }
        if (!team_identity_locked_ && scoreboard_names_locked) {
            Parameters scoreboard = Parameters::object();
            for (int s = 0; s < 2; ++s) {
                Parameters so = Parameters::object();
                so["abbrev"] = locked_scoreboard_team_[s];
                const std::string full = nbaFullName(locked_scoreboard_team_[s]);
                if (!full.empty()) so["name"] = full;
                scoreboard[scoreboardSideName(s)] = so;
            }
            out["scoreboard"] = scoreboard;
        }
        if (include_evidence) {
            Parameters ev = Parameters::object();
            for (int v = 0; v < 2; ++v) {
                Parameters row = Parameters::object();
                row["team_a"] = team_identity_evidence_[v][0];
                row["team_b"] = team_identity_evidence_[v][1];
                ev[visualTeamName(v)] = row;
            }
            out["evidence"] = ev;
            out["evidence_total"] = teamIdentityEvidenceTotal();
        }
        return out;
    }

    void maybeLockTeamIdentity() {
        if (team_identity_locked_) return;
        const int normal = team_identity_evidence_[0][0] + team_identity_evidence_[1][1];
        const int swapped = team_identity_evidence_[0][1] + team_identity_evidence_[1][0];
        const bool normal_ready =
            normal >= team_identity_lock_min_evidence_ &&
            (normal - swapped) >= team_identity_lock_min_margin_;
        const bool swapped_ready =
            swapped >= team_identity_lock_min_evidence_ &&
            (swapped - normal) >= team_identity_lock_min_margin_;

        if (!normal_ready && !swapped_ready) return;

        if (normal_ready) {
            visual_to_scoreboard_[0] = "team_a";
            visual_to_scoreboard_[1] = "team_b";
            scoreboard_to_visual_[0] = "A";
            scoreboard_to_visual_[1] = "B";
        } else {
            visual_to_scoreboard_[0] = "team_b";
            visual_to_scoreboard_[1] = "team_a";
            scoreboard_to_visual_[0] = "B";
            scoreboard_to_visual_[1] = "A";
        }
        team_identity_locked_ = true;
        team_identity_just_locked_ = true;
        team_identity_lock_frame_ = (int)frame_counter_;
    }

    void observeTeamIdentityEvidence(const std::string& visual_team, const std::string& side) {
        const int v = visualTeamIndex(visual_team);
        const int s = scoreboardSideIndex(side);
        if (v < 0 || s < 0) return;
        ++team_identity_evidence_[v][s];
        maybeLockTeamIdentity();
    }

    static Parameters scoreJson(const ScoreState& s) {
        Parameters out;
        if (s.a >= 0) out["team_a"] = s.a;
        if (s.b >= 0) out["team_b"] = s.b;
        if (s.period > 0) out["period"] = s.period;
        if (s.clock_sec >= 0) out["period_clock_remaining_sec"] = s.clock_sec;
        if (s.shot_clock_sec >= 0) out["shot_clock_sec"] = s.shot_clock_sec;
        return out;
    }

    static bool readScoreState(const Parameters& frame_json, ScoreState& out) {
        if (!frame_json.contains("game") || !frame_json["game"].is_object()) return false;
        const auto& gs = frame_json["game"];
        if (!gs.contains("score_a") || !gs.contains("score_b")) return false;
        try {
            out.a = gs["score_a"].get<int>();
            out.b = gs["score_b"].get<int>();
        } catch (...) {
            return false;
        }
        out.period = gs.value("period", -1);
        out.clock_sec = gs.value("period_clock_remaining_sec", gs.value("clock_sec", -1));
        out.shot_clock_sec = gs.value("shot_clock_sec", -1);
        return haveScore(out);
    }

    static bool isKnownZone(const std::string& zone) {
        return !zone.empty() && zone != "unknown";
    }

    static std::string bestCourtZone(const Parameters& court_zone_md, std::string* source = nullptr) {
        const std::string handler_zone = getOr<std::string>(court_zone_md, "handler_zone", std::string());
        const std::string ball_zone = getOr<std::string>(court_zone_md, "ball_zone", std::string());
        if (isKnownZone(handler_zone)) {
            if (source) *source = "handler";
            return handler_zone;
        }
        if (isKnownZone(ball_zone)) {
            if (source) *source = "ball";
            return ball_zone;
        }
        if (source) source->clear();
        return {};
    }

    static std::string dominantZone(const std::map<std::string, int>& zone_frames) {
        std::string best;
        int best_count = 0;
        for (const auto& kv : zone_frames) {
            if (kv.second > best_count) {
                best = kv.first;
                best_count = kv.second;
            }
        }
        return best_count >= 5 ? best : std::string();
    }

    Parameters scaleBox(const Parameters& det) const {
        if (!det.contains("xyxy") || !det["xyxy"].is_array() || det["xyxy"].size() < 4) return nullptr;
        return Parameters::array({
            scaleX(det["xyxy"][0].get<float>()),
            scaleY(det["xyxy"][1].get<float>()),
            scaleX(det["xyxy"][2].get<float>()),
            scaleY(det["xyxy"][3].get<float>())
        });
    }

    struct MaskInfo {
        int num_masks;
        int w;
        int h;
        const float* data;
    };

    static bool readCpuMasks(const AVFrame* raw, int slot, MaskInfo& out) {
        if (!raw) return false;
        AVFrameSideData* sd = av_frame_get_side_data(raw, yoloSegCpuSideDataType(slot));
        if (!sd || !sd->buf || sd->buf->size < 16) return false;
        const uint32_t* header = (const uint32_t*)sd->buf->data;
        out.num_masks = (int)header[0];
        out.w = (int)header[1];
        out.h = (int)header[2];
        size_t expected = 16 + (size_t)out.num_masks * (size_t)out.w * (size_t)out.h * sizeof(float);
        if ((size_t)sd->buf->size < expected) return false;
        out.data = (const float*)(sd->buf->data + 16);
        return out.num_masks > 0 && out.w > 0 && out.h > 0;
    }

    static Parameters dedupeContour(Parameters& pts) {
        if (pts.size() <= 1) return pts;
        Parameters deduped = Parameters::array();
        deduped.push_back(pts[0]);
        for (size_t i = 1; i < pts.size(); ++i) {
            if (pts[i] != pts[i - 1]) deduped.push_back(pts[i]);
        }
        if (deduped.size() > 1 && deduped.back() == deduped.front()) deduped.erase(deduped.size() - 1);
        return deduped;
    }

    Parameters traceContour(const float* mask, int w, int h, float scale_x, float scale_y) const {
        return traceContourRegion(mask, w, h, 0, 0, w, h, scale_x, scale_y);
    }

    Parameters traceContourRegion(const float* mask, int w, int h,
                                  int rx0, int ry0, int rx1, int ry1,
                                  float scale_x, float scale_y) const {
        Parameters pts = Parameters::array();
        int step = std::max(1, contour_simplify_step_);
        rx0 = std::max(0, rx0); ry0 = std::max(0, ry0);
        rx1 = std::min(w, rx1); ry1 = std::min(h, ry1);
        if (rx0 >= rx1 || ry0 >= ry1) return pts;

        for (int x = rx0; x < rx1; x += step) {
            for (int y = ry0; y < ry1; ++y) {
                if (mask[y * w + x] >= mask_threshold_) {
                    pts.push_back(Parameters::array({ri(x * scale_x), ri(y * scale_y)}));
                    break;
                }
            }
        }
        for (int y = ry0; y < ry1; y += step) {
            for (int x = rx1 - 1; x >= rx0; --x) {
                if (mask[y * w + x] >= mask_threshold_) {
                    pts.push_back(Parameters::array({ri(x * scale_x), ri(y * scale_y)}));
                    break;
                }
            }
        }
        for (int x = rx1 - 1; x >= rx0; x -= step) {
            for (int y = ry1 - 1; y >= ry0; --y) {
                if (mask[y * w + x] >= mask_threshold_) {
                    pts.push_back(Parameters::array({ri(x * scale_x), ri(y * scale_y)}));
                    break;
                }
            }
        }
        for (int y = ry1 - 1; y >= ry0; y -= step) {
            for (int x = rx0; x < rx1; ++x) {
                if (mask[y * w + x] >= mask_threshold_) {
                    pts.push_back(Parameters::array({ri(x * scale_x), ri(y * scale_y)}));
                    break;
                }
            }
        }

        return dedupeContour(pts);
    }

    struct DumpResult {
        Parameters main;
        Parameters court;
        Parameters outlines;
        Parameters trail;
    };

    Parameters buildHeader() const {
        Parameters h;
        h["type"] = "header";
        h["schema"] = schema_;
        h["source_w"] = source_w_;
        h["source_h"] = source_h_;
        h["model_w"] = model_w_;
        h["model_h"] = model_h_;
        h["coord_space"] = "source";
        if (fps_ > 0) h["fps"] = fps_;
        if (!video_label_.empty()) h["video"] = video_label_;
        return h;
    }

    DumpResult buildDump(const av::VideoFrame& frm) {
        const AVFrame* raw = frm.raw();
        DumpResult result;
        Parameters& out = result.main;

        int64_t pts = frm.pts().timestamp();
        out["type"] = "frame";
        out["frame"] = frame_counter_;
        out["pts"] = pts;
        if (fps_ > 0) out["t"] = std::round((float)frame_counter_ / (float)fps_ * 1000.0f) / 1000.0f;

        auto players_md = tryParse(raw, player_metadata_key_);
        auto ball_md = tryParse(raw, ball_metadata_key_);
        auto handler_md = tryParse(raw, ball_handler_metadata_key_);
        auto possession_md = tryParse(raw, possession_metadata_key_);
        auto court_zone_md = tryParse(raw, court_zone_metadata_key_);
        auto shot_md = tryParse(raw, camera_shot_metadata_key_);
        auto shot_events_md = tryParse(raw, shot_events_metadata_key_);
        auto game_state_md = tryParse(raw, game_state_metadata_key_);
        auto viewport_md = tryParse(raw, viewport_metadata_key_);
        auto player_seg_md = tryParse(raw, player_seg_metadata_key_);
        auto court_seg_md = tryParse(raw, court_seg_metadata_key_);

        // Camera shot type + transition
        if (shot_md.contains("camera_shot_type")) out["camera_shot"] = shot_md["camera_shot_type"];
        if (shot_md.contains("camera_shot_transition") && shot_md["camera_shot_transition"].get<bool>())
            out["camera_shot_transition"] = true;

        // Shot attempt state. Keep only fields useful to downstream reasoning;
        // counters are retained because event synthesis uses their increments.
        if (shot_events_md.is_object() && !shot_events_md.is_null()) {
            Parameters shot;
            const bool in_flight = getOr<bool>(shot_events_md, "in_flight", false);
            const bool shot_result = getOr<bool>(shot_events_md, "shot_result", false);
            const int total_releases = getOr<int>(shot_events_md, "total_releases", 0);
            const int total_arrivals = getOr<int>(shot_events_md, "total_hoop_arrivals", 0);
            if (in_flight) shot["in_flight"] = true;
            if (total_releases > 0) shot["total_releases"] = total_releases;
            if (total_arrivals > 0) shot["total_hoop_arrivals"] = total_arrivals;
            if (shot_result) shot["result_event"] = true;
            if (shot_events_md.contains("result")) shot["result"] = shot_events_md["result"];
            if (shot_events_md.contains("result_conf")) {
                shot["confidence"] = std::round(shot_events_md["result_conf"].get<float>() * 1000.0f) / 1000.0f;
            }
            if (shot_events_md.contains("result_vx")) shot["result_vx"] = shot_events_md["result_vx"];
            if (shot_events_md.contains("result_vy")) shot["result_vy"] = shot_events_md["result_vy"];
            if (shot_events_md.contains("attempt_type")) shot["attempt_type"] = shot_events_md["attempt_type"];
            if (shot_events_md.contains("attempt_points")) shot["attempt_points"] = shot_events_md["attempt_points"];
            if (shot_events_md.contains("points")) shot["points"] = shot_events_md["points"];
            if (shot_events_md.contains("result_source")) shot["result_source"] = shot_events_md["result_source"];
            if (shot_events_md.contains("ball_hoop_dist")) {
                shot["ball_hoop_dist"] = scaleDist(shot_events_md["ball_hoop_dist"].get<float>());
            }
            if (!shot.empty()) out["shot"] = shot;
        }

        // Possession state
        if (possession_md.is_object() && !possession_md.is_null()) {
            Parameters possession_out;
            if (possession_md.contains("possessing_team")) possession_out["team"] = possession_md["possessing_team"];
            if (possession_md.contains("handler_id")) possession_out["handler_id"] = possession_md["handler_id"];
            if (possession_md.contains("handler_team")) possession_out["handler_team"] = possession_md["handler_team"];
            if (possession_md.contains("ball_state")) possession_out["state"] = possession_md["ball_state"];
            if (possession_md.contains("frames_in_possession")) possession_out["frames_in_possession"] = possession_md["frames_in_possession"];
            if (possession_md.contains("possession_id")) possession_out["possession_id"] = possession_md["possession_id"];
            if (!possession_out.empty()) out["possession"] = possession_out;
        }

        // Parsed OCR game state is intentionally opt-in. Current OCR is noisy enough
        // that exposing it by default makes the semantic dump harder to interpret.
        if (include_ocr_game_state_ && game_state_md.is_object() && !game_state_md.is_null()) {
            maybeLockScoreboardTeamNames(game_state_md);
            Parameters game_out;
            if (!locked_scoreboard_team_[0].empty()) {
                game_out["team_a"] = locked_scoreboard_team_[0];
                const std::string full = nbaFullName(locked_scoreboard_team_[0]);
                if (!full.empty()) game_out["team_a_name"] = full;
            } else if (game_state_md.contains("team_a_abbrev")) {
                game_out["team_a"] = game_state_md["team_a_abbrev"];
                if (game_state_md.contains("team_a_name")) game_out["team_a_name"] = game_state_md["team_a_name"];
            }
            if (!locked_scoreboard_team_[1].empty()) {
                game_out["team_b"] = locked_scoreboard_team_[1];
                const std::string full = nbaFullName(locked_scoreboard_team_[1]);
                if (!full.empty()) game_out["team_b_name"] = full;
            } else if (game_state_md.contains("team_b_abbrev")) {
                game_out["team_b"] = game_state_md["team_b_abbrev"];
                if (game_state_md.contains("team_b_name")) game_out["team_b_name"] = game_state_md["team_b_name"];
            }
            if (game_state_md.contains("score_a")) game_out["score_a"] = game_state_md["score_a"];
            if (game_state_md.contains("score_b")) game_out["score_b"] = game_state_md["score_b"];
            if (game_state_md.contains("score_margin")) game_out["score_margin"] = game_state_md["score_margin"];
            if (game_state_md.contains("leading_team")) game_out["leading_team"] = game_state_md["leading_team"];
            if (game_state_md.contains("leading_team_name")) game_out["leading_team_name"] = game_state_md["leading_team_name"];
            if (game_out.contains("score_a") && game_out.contains("score_b")) {
                const int a = game_out["score_a"].get<int>();
                const int b = game_out["score_b"].get<int>();
                if (a > b && !locked_scoreboard_team_[0].empty()) {
                    game_out["leading_team"] = locked_scoreboard_team_[0];
                    const std::string full = nbaFullName(locked_scoreboard_team_[0]);
                    if (!full.empty()) game_out["leading_team_name"] = full;
                } else if (b > a && !locked_scoreboard_team_[1].empty()) {
                    game_out["leading_team"] = locked_scoreboard_team_[1];
                    const std::string full = nbaFullName(locked_scoreboard_team_[1]);
                    if (!full.empty()) game_out["leading_team_name"] = full;
                } else if (a == b) {
                    game_out["leading_team"] = "TIE";
                    game_out.erase("leading_team_name");
                }
            }
            if (game_state_md.contains("period_num")) game_out["period"] = game_state_md["period_num"];
            if (game_state_md.contains("game_clock_sec")) {
                game_out["clock_sec"] = game_state_md["game_clock_sec"];
                game_out["period_clock_remaining_sec"] = game_state_md["game_clock_sec"];
            }
            if (game_state_md.contains("period_clock_remaining_sec")) {
                game_out["period_clock_remaining_sec"] = game_state_md["period_clock_remaining_sec"];
            }
            if (game_state_md.contains("shot_clock_sec")) game_out["shot_clock_sec"] = game_state_md["shot_clock_sec"];
            if (!game_out.empty()) out["game"] = game_out;
        }

        if (team_identity_locked_ || teamIdentityEvidenceTotal() > 0 ||
            !locked_scoreboard_team_[0].empty() || !locked_scoreboard_team_[1].empty()) {
            out["team_identity"] = teamIdentityJson(false);
        }

        // Court zone — concise semantic form. Detailed court geometry stays in the
        // court sidecar; the main frame stream only keeps the coarse basketball zone.
        if (court_zone_md.is_object() && !court_zone_md.is_null()) {
            Parameters zone_out;
            std::string source;
            const std::string zone = bestCourtZone(court_zone_md, &source);
            if (isKnownZone(zone)) {
                zone_out["zone"] = zone;
                if (!source.empty()) zone_out["source"] = source;
                if (court_zone_md.contains("hoop_side")) zone_out["hoop_side"] = court_zone_md["hoop_side"];
            }
            if (!zone_out.empty()) out["court_zone"] = zone_out;
        }

        // Ball handler track_id
        int handler_track_id = -1;
        if (handler_md.contains("detections") && handler_md["detections"].is_array()) {
            for (const auto& det : handler_md["detections"]) {
                if (!det.is_object()) continue;
                if (getOr<std::string>(det, "label", std::string()) == "BallHandler") {
                    handler_track_id = getOr<int>(det, "track_id", -1);
                    break;
                }
            }
        }

        // Player seg masks — source-space contour scaling
        MaskInfo player_masks = {};
        bool have_player_masks = readCpuMasks(raw, player_seg_slot_, player_masks);

        std::vector<int> seg_det_to_mask;
        if (have_player_masks && player_seg_md.contains("detections") && player_seg_md["detections"].is_array()) {
            int mask_idx = 0;
            for (size_t i = 0; i < player_seg_md["detections"].size(); ++i) {
                if (mask_idx < player_masks.num_masks) {
                    seg_det_to_mask.push_back(mask_idx);
                    ++mask_idx;
                } else {
                    seg_det_to_mask.push_back(-1);
                }
            }
        }

        struct SegMatch { int seg_det_idx; std::string team_ab; };
        std::vector<std::pair<int, SegMatch>> track_to_seg;
        if (player_seg_md.contains("detections") && player_seg_md["detections"].is_array()) {
            if (players_md.contains("detections") && players_md["detections"].is_array()) {
                for (size_t si = 0; si < player_seg_md["detections"].size(); ++si) {
                    const auto& seg_det = player_seg_md["detections"][si];
                    if (!seg_det.contains("xyxy") || !seg_det["xyxy"].is_array() || seg_det["xyxy"].size() < 4) continue;
                    float sx1 = seg_det["xyxy"][0].get<float>(), sy1 = seg_det["xyxy"][1].get<float>();
                    float sx2 = seg_det["xyxy"][2].get<float>(), sy2 = seg_det["xyxy"][3].get<float>();

                    float best_iou = 0.15f;
                    int best_track = -1;
                    for (size_t pi = 0; pi < players_md["detections"].size(); ++pi) {
                        const auto& p = players_md["detections"][pi];
                        if (getOr<std::string>(p, "label", std::string()) != "Player") continue;
                        if (!p.contains("xyxy") || !p["xyxy"].is_array() || p["xyxy"].size() < 4) continue;
                        float px1 = p["xyxy"][0].get<float>(), py1 = p["xyxy"][1].get<float>();
                        float px2 = p["xyxy"][2].get<float>(), py2 = p["xyxy"][3].get<float>();
                        float ix1 = std::max(sx1, px1), iy1 = std::max(sy1, py1);
                        float ix2 = std::min(sx2, px2), iy2 = std::min(sy2, py2);
                        float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
                        float inter = iw * ih;
                        float ua = (sx2 - sx1) * (sy2 - sy1) + (px2 - px1) * (py2 - py1) - inter;
                        float iou = (ua > 0.0f) ? inter / ua : 0.0f;
                        if (iou > best_iou) { best_iou = iou; best_track = getOr<int>(p, "track_id", -1); }
                    }
                    if (best_track >= 0) track_to_seg.push_back({best_track, {(int)si, getOr<std::string>(seg_det, "team_ab", std::string("?"))}});
                }
            }
        }

        // Mask→source scale for player outlines; model→mask to clip region.
        float p_mask_to_src_x = (player_masks.w > 0) ? (float)source_w_ / (float)player_masks.w : 1.0f;
        float p_mask_to_src_y = (player_masks.h > 0) ? (float)source_h_ / (float)player_masks.h : 1.0f;
        float p_model_to_mask_x = (player_masks.w > 0) ? model_w_ / (float)player_masks.w : 1.0f;
        float p_model_to_mask_y = (player_masks.h > 0) ? model_h_ / (float)player_masks.h : 1.0f;

        Parameters outlines_arr = Parameters::array();

        // Players, refs, hoop — all boxes scaled to source space
        if (players_md.contains("detections") && players_md["detections"].is_array()) {
            Parameters players_arr = Parameters::array();
            Parameters refs_arr = Parameters::array();
            Parameters hoop;

            for (const auto& det : players_md["detections"]) {
                if (!det.is_object()) continue;
                std::string label = getOr<std::string>(det, "label", std::string());
                auto box = scaleBox(det);
                if (box.is_null()) continue;
                float conf = getOr<float>(det, "conf", 0.0f);

                if (label == "Player") {
                    Parameters p;
                    int tid = getOr<int>(det, "track_id", -1);
                    if (tid >= 0) p["id"] = tid;
                    std::string tab = getOr<std::string>(det, "team_ab", std::string("?"));
                    if (tab != "?") p["team"] = tab;
                    if (det.contains("team_lock_age")) p["team_lock_age"] = det["team_lock_age"].get<uint32_t>();
                    p["box"] = box;
                    if (tid >= 0 && tid == handler_track_id) p["has_ball"] = true;
                    if (det.contains("track_state")) {
                        std::string ts = det["track_state"].get<std::string>();
                        if (ts != "tracked") p["track_state"] = ts;
                    }

                    if (tid >= 0 && have_player_masks && det.contains("xyxy") && det["xyxy"].size() >= 4) {
                        float bx1 = det["xyxy"][0].get<float>(), by1 = det["xyxy"][1].get<float>();
                        float bx2 = det["xyxy"][2].get<float>(), by2 = det["xyxy"][3].get<float>();
                        int mx0 = (int)(bx1 / p_model_to_mask_x) - 1, my0 = (int)(by1 / p_model_to_mask_y) - 1;
                        int mx1 = (int)(bx2 / p_model_to_mask_x) + 2, my1 = (int)(by2 / p_model_to_mask_y) + 2;

                        for (const auto& ts : track_to_seg) {
                            if (ts.first == tid) {
                                int si = ts.second.seg_det_idx;
                                if (si >= 0 && si < (int)seg_det_to_mask.size()) {
                                    int mi = seg_det_to_mask[(size_t)si];
                                    if (mi >= 0 && mi < player_masks.num_masks) {
                                        const float* mdata = player_masks.data +
                                            (size_t)mi * (size_t)player_masks.w * (size_t)player_masks.h;
                                        auto contour = traceContourRegion(mdata, player_masks.w, player_masks.h,
                                                                          mx0, my0, mx1, my1,
                                                                          p_mask_to_src_x, p_mask_to_src_y);
                                        if (!contour.empty()) {
                                            Parameters oe;
                                            oe["id"] = tid;
                                            oe["outline"] = contour;
                                            outlines_arr.push_back(std::move(oe));
                                        }
                                    }
                                }
                                break;
                            }
                        }
                    }

                    players_arr.push_back(std::move(p));
                } else if (label == "Ref") {
                    Parameters r;
                    int tid = getOr<int>(det, "track_id", -1);
                    if (tid >= 0) r["id"] = tid;
                    r["box"] = box;
                    r["conf"] = (int)(conf * 100.0f + 0.5f);
                    refs_arr.push_back(std::move(r));
                } else if (label == "Hoop" && hoop.is_null()) {
                    hoop = {{"box", box}, {"conf", (int)(conf * 100.0f + 0.5f)}};
                }
            }

            if (!players_arr.empty()) out["players"] = players_arr;
            if (!refs_arr.empty()) out["refs"] = refs_arr;
            if (!hoop.is_null()) out["hoop"] = hoop;
        }

        // Ball
        if (ball_md.contains("detections") && ball_md["detections"].is_array()) {
            for (const auto& det : ball_md["detections"]) {
                if (!det.is_object()) continue;
                auto box = scaleBox(det);
                if (box.is_null()) continue;
                Parameters b;
                b["box"] = box;
                b["conf"] = (int)(getOr<float>(det, "conf", 0.0f) * 100.0f + 0.5f);
                if (det.contains("source")) b["state"] = det["source"];
                if (det.contains("coast_streak")) {
                    int cs = det["coast_streak"].get<int>();
                    if (cs > 0) b["coast_streak"] = cs;
                }
                out["ball"] = b;
                break;
            }
        }

        // Viewport
        if (viewport_md.contains("detections") && viewport_md["detections"].is_array()) {
            for (const auto& det : viewport_md["detections"]) {
                auto box = scaleBox(det);
                if (!box.is_null()) { out["viewport"] = box; break; }
            }
        }

        // Ball trail sidecar — coords in source space
        if (ball_md.contains("trail") && ball_md["trail"].is_array() && !ball_md["trail"].empty()) {
            Parameters trail_entry;
            trail_entry["type"] = "trail";
            trail_entry["frame"] = frame_counter_;
            trail_entry["pts"] = pts;
            Parameters trail = Parameters::array();
            for (const auto& pt : ball_md["trail"]) {
                if (!pt.is_array() || pt.size() < 3) continue;
                trail.push_back(Parameters::array({
                    scaleX((float)pt[0].get<int>()),
                    scaleY((float)pt[1].get<int>()),
                    pt[2].get<int64_t>()
                }));
            }
            if (!trail.empty()) {
                trail_entry["trail"] = trail;
                result.trail = trail_entry;
            }
        }

        if (!outlines_arr.empty()) {
            Parameters outlines_entry;
            outlines_entry["type"] = "outlines";
            outlines_entry["frame"] = frame_counter_;
            outlines_entry["pts"] = pts;
            outlines_entry["players"] = outlines_arr;
            result.outlines = outlines_entry;
        }

        // Court contours — source space
        MaskInfo court_masks = {};
        if (readCpuMasks(raw, court_seg_slot_, court_masks)) {
            float court_scale_x = (float)source_w_ / (float)court_masks.w;
            float court_scale_y = (float)source_h_ / (float)court_masks.h;

            std::vector<std::string> court_labels;
            if (court_seg_md.contains("detections") && court_seg_md["detections"].is_array()) {
                for (const auto& det : court_seg_md["detections"]) {
                    if (!det.is_object()) continue;
                    court_labels.push_back(getOr<std::string>(det, "label", std::string("unknown")));
                }
            }

            Parameters court_entry;
            court_entry["type"] = "court";
            court_entry["frame"] = frame_counter_;
            court_entry["pts"] = pts;
            Parameters court_arr = Parameters::array();
            for (int i = 0; i < court_masks.num_masks; ++i) {
                const float* mdata = court_masks.data + (size_t)i * (size_t)court_masks.w * (size_t)court_masks.h;
                auto contour = traceContour(mdata, court_masks.w, court_masks.h, court_scale_x, court_scale_y);
                if (contour.empty()) continue;
                Parameters entry;
                entry["label"] = (i < (int)court_labels.size()) ? court_labels[(size_t)i] : "unknown";
                entry["outline"] = contour;
                court_arr.push_back(entry);
            }
            if (!court_arr.empty()) {
                court_entry["regions"] = court_arr;
                result.court = court_entry;
            }
        }

        return result;
    }

    void emitEvent(Parameters ev, int64_t pts) {
        if (!file_events_.opened) return;
        ev["frame"] = frame_counter_;
        ev["pts"] = pts;
        if (fps_ > 0) ev["t"] = std::round((float)frame_counter_ / (float)fps_ * 1000.0f) / 1000.0f;
        file_events_.writeLine(ev.dump());
    }

    void startScoreCandidate(const ScoreState& observed, int64_t pts) {
        score_candidate_ = observed;
        score_candidate_frames_ = 1;
        score_candidate_first_frame_ = frame_counter_;
        score_candidate_first_pts_ = pts;
    }

    bool buildScoreboardShotResult(const ScoreState& before,
                                   const ScoreState& after,
                                   const std::string& side,
                                   int delta,
                                   Parameters& score_ev,
                                   Parameters& shot_ev,
                                   ShotResultData& result) {
        if (!have_pending_shot_) {
            if (delta == 2 || delta == 3) score_ev["shot_match"] = "none";
            return false;
        }
        if (delta != 2 && delta != 3) {
            score_ev["shot_match"] = "not_field_goal_delta";
            return false;
        }
        if (pending_shot_.frames_waited > shot_result_wait_frames_) {
            score_ev["shot_match"] = "pending_shot_expired";
            return false;
        }

        const std::string visual_team = pending_shot_.shooting_team;
        observeTeamIdentityEvidence(visual_team, side);
        const std::string scoreboard_team_abbrev = scoreboardTeamAbbrevForSide(side);
        const std::string scoreboard_team_name = scoreboardTeamNameForSide(side);

        const std::string scoreboard_attempt_type = attemptTypeForPoints(delta);
        score_ev["shot_match"] = "pending_shot";
        score_ev["shot_release_frame"] = pending_shot_.release_frame;
        score_ev["shot_arrival_frame"] = pending_shot_.arrival_frame;
        if (!visual_team.empty()) score_ev["visual_team"] = visual_team;
        if (!scoreboard_team_abbrev.empty()) score_ev["scoreboard_team_abbrev"] = scoreboard_team_abbrev;
        if (!scoreboard_team_name.empty()) score_ev["scoreboard_team_name"] = scoreboard_team_name;
        if (team_identity_locked_) score_ev["team_identity"] = teamIdentityJson(false);

        shot_ev["type"] = "shot_result";
        shot_ev["result"] = "made";
        shot_ev["source"] = "scoreboard_delta";
        shot_ev["scoreboard_verified"] = true;
        shot_ev["points"] = delta;
        shot_ev["points_scoreboard"] = delta;
        shot_ev["attempt_type_scoreboard"] = scoreboard_attempt_type;
        shot_ev["score_delta_side"] = side;
        if (!scoreboard_team_abbrev.empty()) shot_ev["scoreboard_team_abbrev"] = scoreboard_team_abbrev;
        if (!scoreboard_team_name.empty()) shot_ev["scoreboard_team_name"] = scoreboard_team_name;
        shot_ev["score_before"] = scoreJson(before);
        shot_ev["score_after"] = scoreJson(after);
        shot_ev["release_frame"] = pending_shot_.release_frame;
        shot_ev["arrival_frame"] = pending_shot_.arrival_frame;
        if (pending_shot_.player_id >= 0) shot_ev["player_id"] = pending_shot_.player_id;
        if (!visual_team.empty()) {
            shot_ev["tracking_team"] = visual_team;
            shot_ev["visual_team"] = visual_team;
        }
        if (team_identity_locked_) shot_ev["team_identity"] = teamIdentityJson(false);
        if (!pending_shot_.attempt_type_detected.empty()) shot_ev["attempt_type_detected"] = pending_shot_.attempt_type_detected;
        if (pending_shot_.points_detected > 0) shot_ev["points_detected"] = pending_shot_.points_detected;
        if (pending_shot_.detector_result_seen) {
            shot_ev["detector_result"] = pending_shot_.detector_result;
            if (!pending_shot_.detector_source.empty()) shot_ev["detector_source"] = pending_shot_.detector_source;
            if (pending_shot_.detector_confidence >= 0.0f) {
                shot_ev["detector_confidence"] =
                    std::round(pending_shot_.detector_confidence * 1000.0f) / 1000.0f;
            }
        }

        const bool detector_disagrees = pending_shot_.detector_result_seen &&
                                        pending_shot_.detector_result == "missed";
        const bool corrected_attempt = pending_shot_.points_detected > 0 &&
                                       pending_shot_.points_detected != delta;
        if (detector_disagrees) shot_ev["detector_disagreement"] = true;
        if (corrected_attempt) shot_ev["attempt_type_corrected_by_scoreboard"] = true;

        result.result = "made";
        result.source = "scoreboard_delta";
        result.points = delta;
        result.attempt_type = scoreboard_attempt_type;
        result.scoreboard_verified = true;
        result.points_detected = pending_shot_.points_detected;
        result.attempt_type_detected = pending_shot_.attempt_type_detected;
        result.points_scoreboard = delta;
        result.attempt_type_scoreboard = scoreboard_attempt_type;
        result.score_delta_side = side;
        result.visual_team = visual_team;
        result.scoreboard_team_abbrev = scoreboard_team_abbrev;
        result.scoreboard_team_name = scoreboard_team_name;
        result.score_before = before;
        result.score_after = after;
        result.detector_result = pending_shot_.detector_result;
        result.detector_disagreement = detector_disagrees;
        result.attempt_type_corrected_by_scoreboard = corrected_attempt;
        return true;
    }

    void processScoreObservation(const Parameters& frame_json, int64_t pts) {
        ScoreState observed;
        if (!readScoreState(frame_json, observed)) return;

        if (!have_confirmed_score_) {
            if (score_candidate_frames_ <= 0 || !sameScore(score_candidate_, observed)) {
                startScoreCandidate(observed, pts);
            } else {
                ++score_candidate_frames_;
            }
            if (score_candidate_frames_ >= score_change_confirm_frames_) {
                confirmed_score_ = observed;
                have_confirmed_score_ = true;
                score_candidate_frames_ = 0;
                Parameters ev;
                ev["type"] = "score_initialized";
                ev["score"] = scoreJson(confirmed_score_);
                ev["confirm_frames"] = score_change_confirm_frames_;
                ev["first_seen_frame"] = score_candidate_first_frame_;
                ev["first_seen_pts"] = score_candidate_first_pts_;
                emitEvent(ev, pts);
            }
            return;
        }

        if (sameScore(observed, confirmed_score_)) {
            score_candidate_frames_ = 0;
            return;
        }

        if (observed.a < confirmed_score_.a || observed.b < confirmed_score_.b) {
            ++score_ocr_regressions_;
            score_candidate_frames_ = 0;
            return;
        }

        const int delta_a = observed.a - confirmed_score_.a;
        const int delta_b = observed.b - confirmed_score_.b;
        const bool one_side_changed = (delta_a > 0 && delta_b == 0) || (delta_b > 0 && delta_a == 0);
        const int delta = std::max(delta_a, delta_b);
        if (!one_side_changed || delta <= 0 || delta > score_change_max_delta_) {
            ++score_ocr_rejected_;
            score_candidate_frames_ = 0;
            return;
        }

        if (score_candidate_frames_ <= 0 || !sameScore(score_candidate_, observed)) {
            startScoreCandidate(observed, pts);
            return;
        }

        ++score_candidate_frames_;
        if (score_candidate_frames_ < score_change_confirm_frames_) return;

        const ScoreState before = confirmed_score_;
        const ScoreState after = observed;
        const std::string side = delta_a > 0 ? "team_a" : "team_b";

        Parameters ev;
        ev["type"] = "score_change";
        ev["side"] = side;
        ev["delta"] = delta;
        ev["score_before"] = scoreJson(before);
        ev["score_after"] = scoreJson(after);
        ev["confirm_frames"] = score_change_confirm_frames_;
        ev["first_seen_frame"] = score_candidate_first_frame_;
        ev["first_seen_pts"] = score_candidate_first_pts_;

        ++score_events_total_;
        if (delta == 2) ++score_events_2pt_;
        else if (delta == 3) ++score_events_3pt_;

        // Lead / run aggregates use the scoreboard side ("team_a"/"team_b").
        // Visual A/B is irrelevant here — we just track who scored next.
        if (current_run_side_ == side) {
            current_run_points_ += delta;
        } else {
            current_run_side_ = side;
            current_run_points_ = delta;
        }
        if (current_run_points_ > longest_run_points_) longest_run_points_ = current_run_points_;
        const int lead = std::abs(after.a - after.b);
        if (lead > largest_lead_) largest_lead_ = lead;
        std::string new_leader = after.a > after.b ? "A" : (after.b > after.a ? "B" : std::string());
        if (!new_leader.empty() && !current_leader_.empty() && new_leader != current_leader_) ++lead_changes_;
        if (!new_leader.empty()) current_leader_ = new_leader;

        Parameters shot_ev;
        ShotResultData shot_result;
        const bool have_scoreboard_shot_result =
            buildScoreboardShotResult(before, after, side, delta, ev, shot_ev, shot_result);

        emitEvent(ev, pts);
        if (have_scoreboard_shot_result) {
            emitEvent(shot_ev, pts);
            onShotResult(shot_result);
            have_pending_shot_ = false;
        }
        if (team_identity_just_locked_) {
            Parameters identity_ev;
            identity_ev["type"] = "team_identity_locked";
            identity_ev["team_identity"] = teamIdentityJson(true);
            emitEvent(identity_ev, pts);
            team_identity_just_locked_ = false;
        }

        confirmed_score_ = observed;
        score_candidate_frames_ = 0;
    }

    void detectAndEmitEvents(const Parameters& frame_json, int64_t pts) {
        // Update last-known handler before any events (current frame's handler, if any)
        if (frame_json.contains("possession")) {
            const auto& pos = frame_json["possession"];
            int hid = pos.value("handler_id", -1);
            std::string tm = pos.value("handler_team", std::string());
            if (hid >= 0) last_known_handler_id_ = hid;
            if (!tm.empty()) last_known_handler_team_ = tm;
        }

        // Camera shot change
        const std::string cam = frame_json.value("camera_shot", std::string());
        if (!cam.empty() && cam != prev_camera_shot_) {
            if (!prev_camera_shot_.empty()) {
                Parameters ev;
                ev["type"] = "camera_shot_change";
                ev["from"] = prev_camera_shot_;
                ev["to"] = cam;
                emitEvent(ev, pts);
            }
            prev_camera_shot_ = cam;
        }

        // Shot release / hoop arrival
        // Shot state uses cumulative counters from the concise `shot` object.
        const Parameters empty_shot = Parameters::object();
        const Parameters& se_dump = (frame_json.contains("shot") && frame_json["shot"].is_object())
            ? frame_json["shot"] : empty_shot;
        int total_rel = getOr<int>(se_dump, "total_releases", prev_total_releases_);
        int total_arr = getOr<int>(se_dump, "total_hoop_arrivals", prev_total_arrivals_);
        if (total_rel > prev_total_releases_) {
            Parameters ev;
            ev["type"] = "shot_release";
            // At the release frame, possession often already flipped to shot_in_air.
            // Attribute to the last-known handler from prior frames.
            if (last_known_handler_id_ >= 0) ev["player_id"] = last_known_handler_id_;
            if (!last_known_handler_team_.empty()) ev["team"] = last_known_handler_team_;
            if (se_dump.contains("attempt_type")) ev["attempt_type_detected"] = se_dump["attempt_type"];
            if (se_dump.contains("attempt_points")) ev["points_detected"] = se_dump["attempt_points"];
            emitEvent(ev, pts);
            have_last_release_ = true;
            last_release_.frame = frame_counter_;
            last_release_.pts = pts;
            last_release_.player_id = last_known_handler_id_;
            last_release_.shooting_team = last_known_handler_team_;
            last_release_.attempt_type = getOr<std::string>(se_dump, "attempt_type", std::string("unknown"));
            last_release_.attempt_points = getOr<int>(se_dump, "attempt_points", 0);
            last_release_.score_at_release = ScoreState();
            readScoreState(frame_json, last_release_.score_at_release);
            onShotRelease(pts, frame_json);
            prev_total_releases_ = total_rel;
        }
        if (total_arr > prev_total_arrivals_) {
            Parameters ev;
            ev["type"] = "shot_hoop_arrival";
            int hoop_dist_val = -1;
            if (se_dump.contains("ball_hoop_dist")) hoop_dist_val = se_dump["ball_hoop_dist"].get<int>();
            if (hoop_dist_val >= 0) ev["ball_hoop_dist"] = hoop_dist_val;
            if (!last_known_handler_team_.empty()) ev["team"] = last_known_handler_team_;
            emitEvent(ev, pts);
            onShotArrival(pts, hoop_dist_val);
            // Start pending shot if a recent release is associated
            if (have_last_release_ && (frame_counter_ - last_release_.frame) < 200) {
                have_pending_shot_ = true;
                pending_shot_.release_frame = last_release_.frame;
                pending_shot_.arrival_frame = frame_counter_;
                pending_shot_.arrival_pts = pts;
                pending_shot_.player_id = last_release_.player_id;
                pending_shot_.shooting_team = last_release_.shooting_team;
                pending_shot_.attempt_type_detected = last_release_.attempt_type;
                pending_shot_.points_detected = last_release_.attempt_points;
                pending_shot_.score_at_release = last_release_.score_at_release;
                pending_shot_.detector_result_seen = false;
                pending_shot_.detector_result.clear();
                pending_shot_.detector_source.clear();
                pending_shot_.detector_confidence = -1.0f;
                pending_shot_.frames_waited = 0;
                have_last_release_ = false;
            }
            prev_total_arrivals_ = total_arr;
        }
        if (getOr<bool>(se_dump, "result_event", false)) {
            Parameters ev;
            ev["type"] = "shot_detector_result";
            if (!last_known_handler_team_.empty()) ev["team"] = last_known_handler_team_;
            ev["result"] = getOr<std::string>(se_dump, "result", std::string("outcome_unknown"));
            if (se_dump.contains("result_source")) ev["source"] = se_dump["result_source"];
            if (se_dump.contains("confidence")) ev["confidence"] = se_dump["confidence"];
            if (se_dump.contains("result_vx")) ev["vx"] = se_dump["result_vx"];
            if (se_dump.contains("result_vy")) ev["vy"] = se_dump["result_vy"];
            if (se_dump.contains("attempt_type")) ev["attempt_type_detected"] = se_dump["attempt_type"];
            if (se_dump.contains("attempt_points")) ev["points_detected"] = se_dump["attempt_points"];
            if (se_dump.contains("points")) ev["points"] = se_dump["points"];
            emitEvent(ev, pts);
            if (have_pending_shot_) {
                pending_shot_.detector_result_seen = true;
                pending_shot_.detector_result = getOr<std::string>(se_dump, "result", std::string("outcome_unknown"));
                pending_shot_.detector_source = getOr<std::string>(se_dump, "result_source", std::string());
                pending_shot_.detector_confidence = getOr<float>(se_dump, "confidence", -1.0f);
            }
        }

        // Possession change — debounced: new handler must persist N frames before emitting.
        if (frame_json.contains("possession")) {
            const auto& pos = frame_json["possession"];
            int hid = pos.value("handler_id", -1);
            std::string pteam = pos.value("handler_team", std::string());
            if (pteam.empty()) pteam = pos.value("team", std::string());

            if (hid >= 0 && hid != prev_handler_id_) {
                if (hid == pending_new_handler_id_) {
                    ++pending_new_handler_frames_;
                    if (!pteam.empty()) pending_new_handler_team_ = pteam;
                } else {
                    pending_new_handler_id_ = hid;
                    pending_new_handler_team_ = pteam;
                    pending_new_handler_frames_ = 1;
                }
                if (pending_new_handler_frames_ >= possession_change_confirm_frames_) {
                    Parameters ev;
                    const bool team_changed = !prev_possessing_team_.empty() &&
                                              !pending_new_handler_team_.empty() &&
                                              pending_new_handler_team_ != prev_possessing_team_;
                    ev["type"] = team_changed ? "team_control_change" : "handler_change";
                    if (prev_handler_id_ >= 0) ev["from_id"] = prev_handler_id_;
                    ev["to_id"] = pending_new_handler_id_;
                    if (team_changed) {
                        ev["from_team"] = prev_possessing_team_;
                        ev["to_team"] = pending_new_handler_team_;
                    } else if (!pending_new_handler_team_.empty()) {
                        ev["team"] = pending_new_handler_team_;
                    }
                    // handler_change is high-volume noise for downstream LLM analysis;
                    // touches are aggregated into the possession record instead.
                    if (team_changed || emit_handler_change_events_) emitEvent(ev, pts);
                    prev_handler_id_ = pending_new_handler_id_;
                    prev_possessing_team_ = pending_new_handler_team_;
                    pending_new_handler_id_ = -1;
                    pending_new_handler_team_.clear();
                    pending_new_handler_frames_ = 0;
                }
            } else {
                // Reverted to the last confirmed handler (flap) — drop pending.
                pending_new_handler_id_ = -1;
                pending_new_handler_team_.clear();
                pending_new_handler_frames_ = 0;
                // Do NOT emit possession_change when only the team label changes for the
                // same handler_id — that's a classifier reclassification, not a real pass.
            }
        }

        // Period change & shot clock reset
        if (frame_json.contains("game")) {
            const auto& gs = frame_json["game"];
            int pn = gs.value("period", -1);
            if (pn > 0 && pn != prev_period_num_) {
                if (prev_period_num_ > 0) {
                    Parameters ev;
                    ev["type"] = "period_change";
                    ev["from"] = prev_period_num_;
                    ev["to"] = pn;
                    emitEvent(ev, pts);
                    if (poss_active_) poss_acc_.period_changed_during = true;
                }
                prev_period_num_ = pn;
            }
            if (emit_ocr_clock_events_) {
                int sc = gs.value("shot_clock_sec", -1);
                if (sc >= 0 && prev_shot_clock_sec_ >= 0 && sc > prev_shot_clock_sec_ + 1) {
                    Parameters ev;
                    ev["type"] = "shot_clock_reset";
                    ev["to"] = sc;
                    emitEvent(ev, pts);
                }
                if (sc >= 0) prev_shot_clock_sec_ = sc;
            }
        }

        processScoreObservation(frame_json, pts);
    }

    void tickPendingShot(const Parameters& frame_json, int64_t pts) {
        if (!have_pending_shot_) return;
        (void)frame_json;
        ++pending_shot_.frames_waited;
        if (pending_shot_.frames_waited >= shot_result_wait_frames_) {
            const bool detector_claimed_score = pending_shot_.detector_result_seen &&
                                                (pending_shot_.detector_result == "scored" ||
                                                 pending_shot_.detector_result == "made");
            const bool detector_claimed_miss = pending_shot_.detector_result_seen &&
                                               pending_shot_.detector_result == "missed";
            const std::string result = detector_claimed_miss ? "missed" :
                                       (detector_claimed_score ? "scoreboard_unconfirmed" : "outcome_unknown");
            Parameters ev;
            ev["type"] = "shot_result";
            if (!pending_shot_.shooting_team.empty()) ev["team"] = pending_shot_.shooting_team;
            if (!pending_shot_.shooting_team.empty()) ev["visual_team"] = pending_shot_.shooting_team;
            ev["result"] = result;
            ev["source"] = "scoreboard_no_delta";
            ev["scoreboard_verified"] = false;
            ev["release_frame"] = pending_shot_.release_frame;
            ev["arrival_frame"] = pending_shot_.arrival_frame;
            if (pending_shot_.player_id >= 0) ev["player_id"] = pending_shot_.player_id;
            if (!pending_shot_.attempt_type_detected.empty()) ev["attempt_type_detected"] = pending_shot_.attempt_type_detected;
            if (pending_shot_.points_detected > 0) ev["points_detected"] = pending_shot_.points_detected;
            if (haveScore(pending_shot_.score_at_release)) ev["score_at_release"] = scoreJson(pending_shot_.score_at_release);
            if (pending_shot_.detector_result_seen) {
                ev["detector_result"] = pending_shot_.detector_result;
                if (!pending_shot_.detector_source.empty()) ev["detector_source"] = pending_shot_.detector_source;
                if (pending_shot_.detector_confidence >= 0.0f) {
                    ev["detector_confidence"] =
                        std::round(pending_shot_.detector_confidence * 1000.0f) / 1000.0f;
                }
            }
            if (detector_claimed_score) ev["detector_disagreement"] = true;
            emitEvent(ev, pts);

            ShotResultData data;
            data.result = result;
            data.source = "scoreboard_no_delta";
            data.scoreboard_verified = false;
            data.points_detected = pending_shot_.points_detected;
            data.attempt_type_detected = pending_shot_.attempt_type_detected;
            data.visual_team = pending_shot_.shooting_team;
            data.detector_result = pending_shot_.detector_result;
            data.detector_disagreement = detector_claimed_score;
            onShotResult(data);
            have_pending_shot_ = false;
        }
    }

    // Picks the shot whose result is most informative for labelling the
    // possession outcome. Made shots win over misses, misses over unknowns,
    // and within a tier we prefer the latest shot.
    static const ShotRecord* representativeShot(const PossessionAcc& a) {
        const ShotRecord* best = nullptr;
        int best_tier = -1;
        for (const auto& sh : a.shots) {
            int tier;
            if (sh.scoreboard_verified && sh.points > 0) tier = 3;
            else if (sh.result == "missed") tier = 2;
            else if (!sh.result.empty()) tier = 1;
            else tier = 0;
            if (!best || tier >= best_tier) { best = &sh; best_tier = tier; }
        }
        return best;
    }

    // Classifies how the possession ended. Used by both the legacy
    // possessions.ndjson record and the LLM-facing pbp.ndjson stream.
    static std::string possessionOutcomeType(const PossessionAcc& a) {
        const ShotRecord* sh = representativeShot(a);
        if (sh) {
            if (sh->scoreboard_verified && sh->points > 0) {
                if (sh->points == 3) return "made_3pt";
                if (sh->points == 2) return "made_2pt";
                if (sh->points == 1) return "made_1pt";
                return "made_shot";
            }
            if (sh->result == "missed") return "missed_shot";
            return "shot_attempt_unverified";
        }
        if (a.period_changed_during) return "end_of_period";
        if (a.clock_at_end == 0) return "end_of_period";
        if (a.shot_clock_at_end == 0) return "shot_clock_violation";
        return "turnover";
    }

    void writePbpRecord(const PossessionAcc& a) {
        if (!file_pbp_.opened) return;
        if (a.team.empty() && a.shots.empty()) return;
        Parameters p;
        p["type"] = "possession";
        p["id"] = a.possession_id;
        if (!a.team.empty()) p["team"] = a.team;
        if (fps_ > 0) {
            p["start_t"] = std::round((float)a.start_frame / (float)fps_ * 1000.0f) / 1000.0f;
            p["end_t"] = std::round((float)a.end_frame / (float)fps_ * 1000.0f) / 1000.0f;
            p["duration_sec"] = std::round((float)a.total_frames / (float)fps_ * 100.0f) / 100.0f;
        }
        Parameters outcome;
        outcome["type"] = possessionOutcomeType(a);
        const ShotRecord* sh = representativeShot(a);
        if (sh) {
            if (sh->points > 0) outcome["points"] = sh->points;
            if (!sh->attempt_type.empty()) outcome["attempt_type"] = sh->attempt_type;
            if (sh->scoreboard_verified) outcome["scoreboard_verified"] = true;
        }
        p["outcome"] = outcome;
        if (a.score_a_at_start >= 0 && a.score_b_at_start >= 0) {
            Parameters sb; sb["a"] = a.score_a_at_start; sb["b"] = a.score_b_at_start;
            p["score_before"] = sb;
        }
        if (a.score_a_at_end >= 0 && a.score_b_at_end >= 0) {
            Parameters sa; sa["a"] = a.score_a_at_end; sa["b"] = a.score_b_at_end;
            p["score_after"] = sa;
        }
        if (a.period_at_start > 0) p["period"] = a.period_at_start;
        if (a.clock_at_start >= 0) p["clock_at_start_sec"] = a.clock_at_start;
        if (a.clock_at_end >= 0) p["clock_at_end_sec"] = a.clock_at_end;
        const std::string dz = dominantZone(a.zone_frames);
        if (!dz.empty()) {
            Parameters c; c["dominant"] = dz;
            Parameters visited = Parameters::array();
            for (const auto& z : a.zones_visited) visited.push_back(z);
            if (!visited.empty()) c["visited"] = visited;
            p["court"] = c;
        }
        if (!a.touches.empty()) {
            Parameters arr = Parameters::array();
            for (const auto& t : a.touches) {
                Parameters tj;
                tj["id"] = t.track_id;
                if (fps_ > 0) tj["sec"] = std::round((float)t.frames / (float)fps_ * 100.0f) / 100.0f;
                else tj["frames"] = t.frames;
                arr.push_back(tj);
            }
            p["touches"] = arr;
            // pass_count = number of distinct-handler hand-offs (touches.size()-1).
            // It overcounts the rare case where the ball returns to a teammate
            // mid-possession, but undercounts nothing — close enough for an LLM.
            p["pass_count"] = (int)a.touches.size() - 1;
        }
        if (!a.shots.empty()) {
            Parameters arr = Parameters::array();
            for (const auto& s : a.shots) {
                Parameters sj;
                if (s.release_frame >= 0 && fps_ > 0) sj["release_t"] = std::round((float)s.release_frame / (float)fps_ * 1000.0f) / 1000.0f;
                if (s.arrival_frame >= 0 && fps_ > 0) sj["arrival_t"] = std::round((float)s.arrival_frame / (float)fps_ * 1000.0f) / 1000.0f;
                if (!s.attempt_type.empty()) sj["attempt_type"] = s.attempt_type;
                if (!s.result.empty()) sj["result"] = s.result;
                if (s.points > 0) sj["points"] = s.points;
                if (s.scoreboard_verified) sj["scoreboard_verified"] = true;
                if (s.game_clock_sec >= 0) sj["clock_sec"] = s.game_clock_sec;
                arr.push_back(sj);
            }
            p["shots"] = arr;
        }
        file_pbp_.writeLine(p.dump());
    }

    void writePossessionRecord(const PossessionAcc& a) {
        writePbpRecord(a);
        if (!file_possessions_.opened) return;
        if (a.team.empty() && a.shots.empty()) return;
        Parameters p;
        p["type"] = "possession";
        p["possession_id"] = a.possession_id;
        if (!a.team.empty()) p["team"] = a.team;
        if (team_identity_locked_) {
            const int v = visualTeamIndex(a.team);
            if (v >= 0 && !visual_to_scoreboard_[v].empty()) {
                p["scoreboard_side"] = visual_to_scoreboard_[v];
                const std::string abbr = scoreboardTeamAbbrevForSide(visual_to_scoreboard_[v]);
                if (!abbr.empty()) {
                    p["team_abbrev"] = abbr;
                    const std::string full = nbaFullName(abbr);
                    if (!full.empty()) p["team_name"] = full;
                }
            }
        }
        Parameters span;
        span["start_frame"] = a.start_frame;
        span["end_frame"] = a.end_frame;
        span["frames"] = a.total_frames;
        if (fps_ > 0) {
            span["start_t"] = std::round((float)a.start_frame / (float)fps_ * 1000.0f) / 1000.0f;
            span["end_t"] = std::round((float)a.end_frame / (float)fps_ * 1000.0f) / 1000.0f;
            span["duration_sec"] = std::round((float)a.total_frames / (float)fps_ * 100.0f) / 100.0f;
        }
        p["span"] = span;
        if (!a.handler_ids.empty()) p["handler_track_ids"] = a.handler_ids;

        Parameters court;
        const std::string dz = dominantZone(a.zone_frames);
        if (!dz.empty()) court["dominant_zone"] = dz;
        Parameters zone_counts;
        Parameters visited = Parameters::array();
        for (const auto& kv : a.zone_frames) {
            if (kv.second < 5) continue;
            zone_counts[kv.first] = kv.second;
            visited.push_back(kv.first);
        }
        if (!zone_counts.empty()) court["zone_frames"] = zone_counts;
        if (!visited.empty()) court["zones_visited"] = visited;
        if (!court.empty()) p["court"] = court;

        Parameters bs;
        bs["controlled"] = a.frames_controlled;
        bs["loose"] = a.frames_loose;
        bs["in_flight"] = a.frames_in_flight;
        p["ball_state_frames"] = bs;
        const bool has_shots = !a.shots.empty();
        Parameters outcome;
        outcome["type"] = possessionOutcomeType(a);
        if (has_shots) {
            Parameters arr = Parameters::array();
            for (const auto& sh : a.shots) {
                Parameters sj;
                if (sh.release_frame >= 0) sj["release_frame"] = sh.release_frame;
                if (sh.arrival_frame >= 0) sj["arrival_frame"] = sh.arrival_frame;
                if (sh.release_hoop_dist >= 0) sj["release_hoop_dist"] = sh.release_hoop_dist;
                if (sh.hoop_dist >= 0) sj["hoop_dist"] = sh.hoop_dist;
                if (!sh.attempt_type.empty()) sj["attempt_type"] = sh.attempt_type;
                if (sh.game_clock_sec >= 0) sj["game_clock_sec"] = sh.game_clock_sec;
                if (sh.period_num >= 0) sj["period_num"] = sh.period_num;
                if (!sh.result.empty()) sj["result"] = sh.result;
                if (!sh.result_source.empty()) sj["result_source"] = sh.result_source;
                if (sh.points > 0) sj["points"] = sh.points;
                if (sh.scoreboard_verified) sj["scoreboard_verified"] = true;
                else if (!sh.result_source.empty()) sj["scoreboard_verified"] = false;
                if (sh.points_detected > 0) sj["points_detected"] = sh.points_detected;
                if (!sh.attempt_type_detected.empty()) sj["attempt_type_detected"] = sh.attempt_type_detected;
                if (sh.points_scoreboard > 0) sj["points_scoreboard"] = sh.points_scoreboard;
                if (!sh.attempt_type_scoreboard.empty()) sj["attempt_type_scoreboard"] = sh.attempt_type_scoreboard;
                if (!sh.score_delta_side.empty()) sj["score_delta_side"] = sh.score_delta_side;
                if (!sh.visual_team.empty()) sj["visual_team"] = sh.visual_team;
                if (!sh.scoreboard_team_abbrev.empty()) sj["scoreboard_team_abbrev"] = sh.scoreboard_team_abbrev;
                if (!sh.scoreboard_team_name.empty()) sj["scoreboard_team_name"] = sh.scoreboard_team_name;
                if (sh.score_before_a >= 0 && sh.score_before_b >= 0) {
                    Parameters score_before;
                    score_before["team_a"] = sh.score_before_a;
                    score_before["team_b"] = sh.score_before_b;
                    sj["score_before"] = score_before;
                }
                if (sh.score_after_a >= 0 && sh.score_after_b >= 0) {
                    Parameters score_after;
                    score_after["team_a"] = sh.score_after_a;
                    score_after["team_b"] = sh.score_after_b;
                    sj["score_after"] = score_after;
                }
                if (!sh.detector_result.empty()) sj["detector_result"] = sh.detector_result;
                if (sh.detector_disagreement) sj["detector_disagreement"] = true;
                if (sh.attempt_type_corrected_by_scoreboard) sj["attempt_type_corrected_by_scoreboard"] = true;
                arr.push_back(sj);
                if (!sh.result.empty()) outcome["result"] = sh.result;
                if (sh.points > 0) outcome["points"] = sh.points;
                if (!sh.attempt_type.empty()) outcome["attempt_type"] = sh.attempt_type;
                if (sh.scoreboard_verified) outcome["scoreboard_verified"] = true;
                if (!sh.score_delta_side.empty()) outcome["score_delta_side"] = sh.score_delta_side;
                if (!sh.visual_team.empty()) outcome["visual_team"] = sh.visual_team;
                if (!sh.scoreboard_team_abbrev.empty()) outcome["scoreboard_team_abbrev"] = sh.scoreboard_team_abbrev;
                if (!sh.scoreboard_team_name.empty()) outcome["scoreboard_team_name"] = sh.scoreboard_team_name;
            }
            p["shots"] = arr;
        }
        p["outcome"] = outcome;
        // Possession counters are written here; shot counters are updated when a
        // shot result event is observed so unknown-team shots still reach summary.
        ++possessions_by_team_[a.team.empty() ? std::string("?") : a.team];
        total_possession_frames_ += a.total_frames;
        file_possessions_.writeLine(p.dump());
    }

    Parameters buildClipSummary() const {
        Parameters s;
        s["type"] = "clip_summary";
        s["total_frames"] = frame_counter_;
        if (fps_ > 0) s["duration_sec"] = std::round((float)frame_counter_ / (float)fps_ * 100.0f) / 100.0f;
        if (!camera_shot_frames_.empty()) {
            Parameters cs;
            for (const auto& kv : camera_shot_frames_) cs[kv.first] = kv.second;
            s["camera_shot_frames"] = cs;
        }
        int poss_total = 0;
        if (!possessions_by_team_.empty()) {
            Parameters pb;
            for (const auto& kv : possessions_by_team_) { pb[kv.first] = kv.second; poss_total += kv.second; }
            s["possessions_by_team"] = pb;
            s["possessions_total"] = poss_total;
        }
        s["shot_attempts"] = shot_attempts_total_;
        if (shot_attempts_total_ > 0) {
            Parameters r;
            r["scored"] = shots_made_;
            r["missed"] = shots_missed_;
            r["outcome_unknown"] = shots_unknown_;
            r["scoreboard_verified"] = scoreboard_verified_shots_;
            r["scoreboard_unverified"] = scoreboard_unverified_shots_;
            r["detector_score_disagreements"] = detector_score_disagreements_;
            s["shot_results"] = r;
        }
        if (score_events_total_ > 0) {
            Parameters se;
            se["total"] = score_events_total_;
            se["two_point_deltas"] = score_events_2pt_;
            se["three_point_deltas"] = score_events_3pt_;
            s["score_changes"] = se;
        }
        // Game-shape aggregates derived from score-change history and possession durations.
        Parameters shape;
        if (score_events_total_ > 0) {
            shape["lead_changes"] = lead_changes_;
            shape["largest_lead"] = largest_lead_;
            shape["longest_run_points"] = longest_run_points_;
        }
        if (poss_total > 0) {
            if (fps_ > 0 && total_possession_frames_ > 0) {
                shape["avg_possession_sec"] =
                    std::round((float)total_possession_frames_ / (float)poss_total / (float)fps_ * 100.0f) / 100.0f;
            }
            if (fps_ > 0 && frame_counter_ > 0) {
                const float duration_min = (float)frame_counter_ / (float)fps_ / 60.0f;
                if (duration_min > 0.0f) {
                    shape["pace_per_minute"] =
                        std::round((float)poss_total / duration_min * 100.0f) / 100.0f;
                }
            }
        }
        if (!shape.empty()) s["game_shape"] = shape;
        if (score_ocr_regressions_ > 0 || score_ocr_rejected_ > 0) {
            Parameters oq;
            if (score_ocr_regressions_ > 0) oq["regressions_ignored"] = score_ocr_regressions_;
            if (score_ocr_rejected_ > 0) oq["invalid_changes_ignored"] = score_ocr_rejected_;
            s["score_ocr_quality"] = oq;
        }
        if (first_game_clock_sec_ >= 0) s["first_game_clock_sec"] = first_game_clock_sec_;
        if (latest_game_clock_sec_ >= 0) s["last_game_clock_sec"] = latest_game_clock_sec_;
        if (team_identity_locked_ || teamIdentityEvidenceTotal() > 0 ||
            !locked_scoreboard_team_[0].empty() || !locked_scoreboard_team_[1].empty()) {
            s["team_identity"] = teamIdentityJson(true);
        }
        s["ocr_game_state"] = include_ocr_game_state_ ? "included" : "disabled";
        return s;
    }

    void writeClipSummaryFile() {
        if (output_file_summary_.empty()) return;
        std::ofstream out(output_file_summary_, std::ios::out | std::ios::trunc);
        if (!out) return;
        out << buildClipSummary().dump(2) << '\n';
    }

    void writeClipSummary() {
        Parameters s = buildClipSummary();
        std::string json = s.dump();
        if (file_main_.opened) file_main_.writeLine(json);
        if (file_events_.opened) file_events_.writeLine(json);
        if (file_possessions_.opened) file_possessions_.writeLine(json);
        if (file_pbp_.opened) file_pbp_.writeLine(json);
        writeClipSummaryFile();
    }

    void flushDeferredPossession() {
        if (!have_deferred_poss_) return;
        have_deferred_poss_ = false;
        writePossessionRecord(deferred_poss_);
    }

    void flushPossession() {
        if (!poss_active_) return;
        poss_active_ = false;
        // If the last shot is unresolved, defer flush so the result lands here.
        const bool last_unresolved = !poss_acc_.shots.empty() && poss_acc_.shots.back().result.empty();
        if (last_unresolved && have_pending_shot_) {
            flushDeferredPossession();  // evict any older deferred
            deferred_poss_ = poss_acc_;
            have_deferred_poss_ = true;
            return;
        }
        writePossessionRecord(poss_acc_);
    }

    void updatePossessionAcc(const Parameters& frame_json, int64_t pts) {
        if (!frame_json.contains("possession")) return;
        const auto& pos = frame_json["possession"];
        int pid = pos.value("possession_id", -1);
        if (pid < 0) return;
        std::string team = pos.value("team", std::string());

        const bool poss_starting = !poss_active_ || pid != poss_acc_.possession_id;
        if (poss_starting) {
            flushPossession();
            poss_acc_ = PossessionAcc();
            poss_active_ = true;
            poss_acc_.possession_id = pid;
            poss_acc_.team = team;
            poss_acc_.start_frame = frame_counter_;
            poss_acc_.start_pts = pts;
            if (frame_json.contains("game")) {
                const auto& gs = frame_json["game"];
                poss_acc_.period_at_start = gs.value("period", -1);
                if (gs.contains("period_clock_remaining_sec")) poss_acc_.clock_at_start = gs["period_clock_remaining_sec"].get<int>();
                else if (gs.contains("clock_sec")) poss_acc_.clock_at_start = gs["clock_sec"].get<int>();
                poss_acc_.score_a_at_start = gs.value("score_a", -1);
                poss_acc_.score_b_at_start = gs.value("score_b", -1);
            }
        }
        if (poss_acc_.team.empty() && !team.empty()) poss_acc_.team = team;
        poss_acc_.end_frame = frame_counter_;
        poss_acc_.end_pts = pts;
        ++poss_acc_.total_frames;
        std::string bs = pos.value("state", std::string());
        if (bs == "controlled") ++poss_acc_.frames_controlled;
        else if (bs == "loose") ++poss_acc_.frames_loose;
        else if (bs == "shot_in_air") ++poss_acc_.frames_in_flight;
        int hid = pos.value("handler_id", -1);
        if (hid >= 0) {
            if (std::find(poss_acc_.handler_ids.begin(), poss_acc_.handler_ids.end(), hid) == poss_acc_.handler_ids.end()) {
                poss_acc_.handler_ids.push_back(hid);
            }
            const size_t n = poss_acc_.touches.size();
            if (n >= 1 && poss_acc_.touches.back().track_id == hid) {
                ++poss_acc_.touches.back().frames;
            } else if (n >= 2 &&
                       poss_acc_.touches[n - 2].track_id == hid &&
                       poss_acc_.touches.back().frames <= touch_merge_gap_frames_) {
                // Bridge a brief mis-classification of the ball-handler back
                // into the prior touch — same player resumes possession after a
                // tracking flicker, not a real two-pass sequence.
                Touch& prev = poss_acc_.touches[n - 2];
                prev.frames += poss_acc_.touches.back().frames + 1;
                poss_acc_.touches.pop_back();
            } else {
                Touch t;
                t.track_id = hid;
                t.start_frame = (int)frame_counter_;
                t.frames = 1;
                poss_acc_.touches.push_back(t);
            }
        }
        if (frame_json.contains("game")) {
            const auto& gs = frame_json["game"];
            poss_acc_.period_at_end = gs.value("period", poss_acc_.period_at_end);
            if (gs.contains("period_clock_remaining_sec")) poss_acc_.clock_at_end = gs["period_clock_remaining_sec"].get<int>();
            else if (gs.contains("clock_sec")) poss_acc_.clock_at_end = gs.value("clock_sec", poss_acc_.clock_at_end);
            if (gs.contains("shot_clock_sec")) poss_acc_.shot_clock_at_end = gs["shot_clock_sec"].get<int>();
            poss_acc_.score_a_at_end = gs.value("score_a", poss_acc_.score_a_at_end);
            poss_acc_.score_b_at_end = gs.value("score_b", poss_acc_.score_b_at_end);
        }
        if (frame_json.contains("court_zone")) {
            const auto& cz = frame_json["court_zone"];
            std::string z = cz.value("zone", std::string());
            if (!isKnownZone(z)) z = cz.value("handler_zone", std::string());
            if (!isKnownZone(z)) z = cz.value("ball_zone", std::string());
            if (isKnownZone(z)) {
                ++poss_acc_.zone_frames[z];
                if (std::find(poss_acc_.zones_visited.begin(), poss_acc_.zones_visited.end(), z) == poss_acc_.zones_visited.end())
                    poss_acc_.zones_visited.push_back(z);
            }
        }
    }

    void onShotRelease(int64_t /*pts*/, const Parameters& frame_json) {
        if (!poss_active_) return;
        ShotRecord s;
        s.release_frame = (int)frame_counter_;
        s.visual_team = poss_acc_.team;
        if (frame_json.contains("shot") && frame_json["shot"].contains("ball_hoop_dist")) {
            s.release_hoop_dist = frame_json["shot"]["ball_hoop_dist"].get<int>();
        }
        if (frame_json.contains("shot") && frame_json["shot"].contains("attempt_type")) {
            s.attempt_type = frame_json["shot"]["attempt_type"].get<std::string>();
            s.attempt_type_detected = s.attempt_type;
        } else if (frame_json.contains("court_zone") && frame_json["court_zone"].contains("inside_three_point_area")) {
            s.attempt_type = frame_json["court_zone"]["inside_three_point_area"].get<bool>() ? "2pt" : "3pt";
        } else {
            s.attempt_type = "unknown";
        }
        if (frame_json.contains("shot") && frame_json["shot"].contains("attempt_points")) {
            s.points_detected = frame_json["shot"]["attempt_points"].get<int>();
        }
        if (frame_json.contains("game")) {
            const auto& gs = frame_json["game"];
            if (gs.contains("period_clock_remaining_sec")) s.game_clock_sec = gs["period_clock_remaining_sec"].get<int>();
            else if (gs.contains("clock_sec")) s.game_clock_sec = gs["clock_sec"].get<int>();
            if (gs.contains("period")) s.period_num = gs["period"].get<int>();
        }
        poss_acc_.shots.push_back(s);
    }

    void onShotArrival(int64_t /*pts*/, int hoop_dist) {
        if (!poss_active_) return;
        // Update the last shot awaiting arrival; else start a new shot with arrival only.
        for (auto it = poss_acc_.shots.rbegin(); it != poss_acc_.shots.rend(); ++it) {
            if (it->arrival_frame < 0) {
                it->arrival_frame = (int)frame_counter_;
                if (hoop_dist >= 0) it->hoop_dist = hoop_dist;
                return;
            }
        }
        ShotRecord s;
        s.arrival_frame = (int)frame_counter_;
        if (hoop_dist >= 0) s.hoop_dist = hoop_dist;
        poss_acc_.shots.push_back(s);
    }

    static void applyResult(std::vector<ShotRecord>& shots, const ShotResultData& data) {
        for (auto it = shots.rbegin(); it != shots.rend(); ++it) {
            if (it->result.empty()) {
                it->result = data.result;
                it->result_source = data.source;
                if (!data.attempt_type.empty() && data.attempt_type != "unknown") it->attempt_type = data.attempt_type;
                if (data.points > 0) it->points = data.points;
                it->scoreboard_verified = data.scoreboard_verified;
                if (data.points_detected > 0) it->points_detected = data.points_detected;
                if (!data.attempt_type_detected.empty()) it->attempt_type_detected = data.attempt_type_detected;
                if (data.points_scoreboard > 0) it->points_scoreboard = data.points_scoreboard;
                if (!data.attempt_type_scoreboard.empty()) it->attempt_type_scoreboard = data.attempt_type_scoreboard;
                it->score_delta_side = data.score_delta_side;
                it->visual_team = data.visual_team;
                it->scoreboard_team_abbrev = data.scoreboard_team_abbrev;
                it->scoreboard_team_name = data.scoreboard_team_name;
                it->score_before_a = data.score_before.a;
                it->score_before_b = data.score_before.b;
                it->score_after_a = data.score_after.a;
                it->score_after_b = data.score_after.b;
                it->detector_result = data.detector_result;
                it->detector_disagreement = data.detector_disagreement;
                it->attempt_type_corrected_by_scoreboard = data.attempt_type_corrected_by_scoreboard;
                return;
            }
        }
        // No pending shot — record a standalone result (rare: result without release/arrival).
        ShotRecord s;
        s.result = data.result;
        s.result_source = data.source;
        if (!data.attempt_type.empty()) s.attempt_type = data.attempt_type;
        if (data.points > 0) s.points = data.points;
        s.scoreboard_verified = data.scoreboard_verified;
        s.points_detected = data.points_detected;
        s.attempt_type_detected = data.attempt_type_detected;
        s.points_scoreboard = data.points_scoreboard;
        s.attempt_type_scoreboard = data.attempt_type_scoreboard;
        s.score_delta_side = data.score_delta_side;
        s.visual_team = data.visual_team;
        s.scoreboard_team_abbrev = data.scoreboard_team_abbrev;
        s.scoreboard_team_name = data.scoreboard_team_name;
        s.score_before_a = data.score_before.a;
        s.score_before_b = data.score_before.b;
        s.score_after_a = data.score_after.a;
        s.score_after_b = data.score_after.b;
        s.detector_result = data.detector_result;
        s.detector_disagreement = data.detector_disagreement;
        s.attempt_type_corrected_by_scoreboard = data.attempt_type_corrected_by_scoreboard;
        shots.push_back(s);
    }

    void countShotResultForSummary(const std::string& result) {
        ++shot_attempts_total_;
        if (result == "made" || result == "scored") ++shots_made_;
        else if (result == "missed") ++shots_missed_;
        else ++shots_unknown_;
    }

    void onShotResult(const ShotResultData& data) {
        countShotResultForSummary(data.result);
        if (data.scoreboard_verified) ++scoreboard_verified_shots_;
        else if (data.source == "scoreboard_no_delta") ++scoreboard_unverified_shots_;
        if (data.detector_disagreement) ++detector_score_disagreements_;

        // A deferred possession owns this result (shot was released there).
        if (have_deferred_poss_) {
            applyResult(deferred_poss_.shots, data);
            flushDeferredPossession();
            return;
        }
        if (!poss_active_) return;
        applyResult(poss_acc_.shots, data);
    }

    void onShotResult(const std::string& result, int points, const std::string& attempt_type = std::string()) {
        ShotResultData data;
        data.result = result;
        data.points = points;
        data.attempt_type = attempt_type;
        onShotResult(data);
    }

    void resetRunState() {
        frame_counter_ = 0;
        cached_json_.clear();
        header_written_ = false;
        prev_total_releases_ = 0;
        prev_total_arrivals_ = 0;
        prev_handler_id_ = -1;
        prev_possessing_team_.clear();
        prev_period_num_ = -1;
        prev_shot_clock_sec_ = -1;
        prev_camera_shot_.clear();
        have_last_release_ = false;
        have_pending_shot_ = false;
        last_known_handler_id_ = -1;
        last_known_handler_team_.clear();
        pending_new_handler_id_ = -1;
        pending_new_handler_team_.clear();
        pending_new_handler_frames_ = 0;
        poss_active_ = false;
        poss_acc_ = PossessionAcc();
        have_deferred_poss_ = false;
        deferred_poss_ = PossessionAcc();
        last_game_clock_sec_ = -1;
        last_game_clock_change_t_ = -1.0f;
        clock_currently_stopped_ = false;
        camera_shot_frames_.clear();
        possessions_by_team_.clear();
        shots_made_ = shots_missed_ = shots_unknown_ = shot_attempts_total_ = 0;
        score_events_total_ = score_events_2pt_ = score_events_3pt_ = 0;
        scoreboard_verified_shots_ = scoreboard_unverified_shots_ = detector_score_disagreements_ = 0;
        score_ocr_regressions_ = score_ocr_rejected_ = 0;
        have_confirmed_score_ = false;
        confirmed_score_ = ScoreState();
        score_candidate_ = ScoreState();
        score_candidate_frames_ = 0;
        score_candidate_first_frame_ = 0;
        score_candidate_first_pts_ = 0;
        for (int v = 0; v < 2; ++v) {
            visual_to_scoreboard_[v].clear();
            for (int s = 0; s < 2; ++s) team_identity_evidence_[v][s] = 0;
        }
        for (int s = 0; s < 2; ++s) {
            scoreboard_to_visual_[s].clear();
            locked_scoreboard_team_[s].clear();
            scoreboard_team_candidate_[s].clear();
        }
        team_identity_locked_ = false;
        team_identity_just_locked_ = false;
        team_identity_lock_frame_ = -1;
        scoreboard_team_candidate_hits_ = 0;
        first_game_clock_sec_ = latest_game_clock_sec_ = -1;
    }

public:
    using NodeSISO<av::VideoFrame, av::VideoFrame>::NodeSISO;
    bool consumeEofIfPresent() override {
        return false;
    }

    void process() override {
        av::VideoFrame frm = this->source_->get();
        if (isEofMarker(frm)) {
            flushPossession();
            flushDeferredPossession();
            writeClipSummary();
            file_main_.close();
            file_court_.close();
            file_outlines_.close();
            file_trail_.close();
            file_events_.close();
            file_possessions_.close();
            file_pbp_.close();
            resetRunState();
            this->sink_->put(frm);
            this->finished_ = true;
            return;
        }
        if (!frm) return;

        ++frame_counter_;

        if (!header_written_) {
            source_w_ = frm.width();
            source_h_ = frm.height();
            auto players_md = tryParse(frm.raw(), player_metadata_key_);
            model_w_ = getOr<float>(players_md, "model_width", 960.0f);
            model_h_ = getOr<float>(players_md, "model_height", 544.0f);
            scale_x_ = (source_w_ > 0 && model_w_ > 0) ? (float)source_w_ / model_w_ : 1.0f;
            scale_y_ = (source_h_ > 0 && model_h_ > 0) ? (float)source_h_ / model_h_ : 1.0f;

            if (!output_file_.empty() && !file_main_.opened && !file_main_.open(output_file_))
                logstream << "metadata_dump: cannot open output file: " << output_file_;
            if (!output_file_court_.empty() && !file_court_.opened && !file_court_.open(output_file_court_))
                logstream << "metadata_dump: cannot open court file: " << output_file_court_;
            if (!output_file_outlines_.empty() && !file_outlines_.opened && !file_outlines_.open(output_file_outlines_))
                logstream << "metadata_dump: cannot open outlines file: " << output_file_outlines_;
            if (!output_file_trail_.empty() && !file_trail_.opened && !file_trail_.open(output_file_trail_))
                logstream << "metadata_dump: cannot open trail file: " << output_file_trail_;
            if (!output_file_events_.empty() && !file_events_.opened && !file_events_.open(output_file_events_))
                logstream << "metadata_dump: cannot open events file: " << output_file_events_;
            if (!output_file_possessions_.empty() && !file_possessions_.opened && !file_possessions_.open(output_file_possessions_))
                logstream << "metadata_dump: cannot open possessions file: " << output_file_possessions_;
            if (!output_file_pbp_.empty() && !file_pbp_.opened && !file_pbp_.open(output_file_pbp_))
                logstream << "metadata_dump: cannot open pbp file: " << output_file_pbp_;

            const std::string header = buildHeader().dump();
            if (file_main_.opened) file_main_.writeLine(header);
            if (file_events_.opened) file_events_.writeLine(header);
            if (file_possessions_.opened) file_possessions_.writeLine(header);
            if (file_pbp_.opened) file_pbp_.writeLine(header);
            if (file_court_.opened) file_court_.writeLine(header);
            if (file_outlines_.opened) file_outlines_.writeLine(header);
            if (file_trail_.opened) file_trail_.writeLine(header);

            header_written_ = true;
        }

        if (dump_every_n_ > 1 && (frame_counter_ % (uint64_t)dump_every_n_) != 1) {
            if (!cached_json_.empty()) {
                av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), cached_json_.c_str(), 0);
            }
            this->sink_->put(frm);
            return;
        }

        DumpResult dump = buildDump(frm);
        cached_json_ = dump.main.dump();
        av_dict_set(&frm.raw()->metadata, output_metadata_key_.c_str(), cached_json_.c_str(), 0);

        int64_t pts = frm.pts().timestamp();

        // Camera-shot counter (clip summary).
        if (dump.main.contains("camera_shot")) {
            ++camera_shot_frames_[dump.main["camera_shot"].get<std::string>()];
        }

        // Clock-movement events.
        if (dump.main.contains("game") && dump.main["game"].contains("clock_sec")) {
            int sec = dump.main["game"]["clock_sec"].get<int>();
            float now_t = (fps_ > 0) ? (float)frame_counter_ / (float)fps_ : 0.0f;
            if (first_game_clock_sec_ < 0) first_game_clock_sec_ = sec;
            latest_game_clock_sec_ = sec;
            if (sec != last_game_clock_sec_) {
                if (emit_ocr_clock_events_ && clock_currently_stopped_) {
                    Parameters ev; ev["type"] = "clock_started"; ev["to"] = sec;
                    emitEvent(ev, pts);
                }
                clock_currently_stopped_ = false;
                last_game_clock_sec_ = sec;
                last_game_clock_change_t_ = now_t;
            } else if (emit_ocr_clock_events_ && !clock_currently_stopped_ && last_game_clock_change_t_ >= 0.0f &&
                       (now_t - last_game_clock_change_t_) >= clock_stopped_threshold_s_) {
                Parameters ev; ev["type"] = "clock_stopped"; ev["at"] = sec;
                emitEvent(ev, pts);
                clock_currently_stopped_ = true;
            }
        }

        file_main_.writeLine(cached_json_);
        if (!dump.court.is_null()) file_court_.writeLine(dump.court.dump());
        if (!dump.outlines.is_null()) file_outlines_.writeLine(dump.outlines.dump());
        if (!dump.trail.is_null()) file_trail_.writeLine(dump.trail.dump());

        detectAndEmitEvents(dump.main, pts);
        tickPendingShot(dump.main, pts);
        updatePossessionAcc(dump.main, pts);

        if (!output_file_summary_.empty() && summary_update_every_n_ > 0 &&
            (frame_counter_ % (uint64_t)summary_update_every_n_) == 0) {
            writeClipSummaryFile();
        }

        if (debug_log_every_n_ > 0 && (frame_counter_ % (uint64_t)debug_log_every_n_) == 0) {
            logstream << "metadata_dump: frame=" << frame_counter_ << " " << cached_json_;
        }

        this->sink_->put(frm);
    }

    static std::shared_ptr<MetadataDump> create(NodeCreationInfo& nci) {
        EdgeManager& edges = nci.edges;
        const Parameters& params = nci.params;
        auto r = NodeSISO<av::VideoFrame, av::VideoFrame>::template createCommon<MetadataDump>(edges, params);

        if (params.count("player_metadata_key")) r->player_metadata_key_ = params["player_metadata_key"].get<std::string>();
        if (params.count("ball_metadata_key")) r->ball_metadata_key_ = params["ball_metadata_key"].get<std::string>();
        if (params.count("ball_handler_metadata_key")) r->ball_handler_metadata_key_ = params["ball_handler_metadata_key"].get<std::string>();
        if (params.count("possession_metadata_key")) r->possession_metadata_key_ = params["possession_metadata_key"].get<std::string>();
        if (params.count("court_zone_metadata_key")) r->court_zone_metadata_key_ = params["court_zone_metadata_key"].get<std::string>();
        if (params.count("camera_shot_metadata_key")) r->camera_shot_metadata_key_ = params["camera_shot_metadata_key"].get<std::string>();
        if (params.count("shot_events_metadata_key")) r->shot_events_metadata_key_ = params["shot_events_metadata_key"].get<std::string>();
        if (params.count("scoreboard_metadata_key")) r->scoreboard_metadata_key_ = params["scoreboard_metadata_key"].get<std::string>();
        if (params.count("game_state_metadata_key")) r->game_state_metadata_key_ = params["game_state_metadata_key"].get<std::string>();
        if (params.count("viewport_metadata_key")) r->viewport_metadata_key_ = params["viewport_metadata_key"].get<std::string>();
        if (params.count("player_seg_metadata_key")) r->player_seg_metadata_key_ = params["player_seg_metadata_key"].get<std::string>();
        if (params.count("court_seg_metadata_key")) r->court_seg_metadata_key_ = params["court_seg_metadata_key"].get<std::string>();
        if (params.count("output_metadata_key")) r->output_metadata_key_ = params["output_metadata_key"].get<std::string>();
        if (params.count("output_file")) r->output_file_ = params["output_file"].get<std::string>();
        if (params.count("output_file_court")) r->output_file_court_ = params["output_file_court"].get<std::string>();
        if (params.count("output_file_outlines")) r->output_file_outlines_ = params["output_file_outlines"].get<std::string>();
        if (params.count("output_file_trail")) r->output_file_trail_ = params["output_file_trail"].get<std::string>();
        if (params.count("output_file_events")) r->output_file_events_ = params["output_file_events"].get<std::string>();
        if (params.count("output_file_possessions")) r->output_file_possessions_ = params["output_file_possessions"].get<std::string>();
        if (params.count("output_file_pbp")) r->output_file_pbp_ = params["output_file_pbp"].get<std::string>();
        if (params.count("emit_handler_change_events")) r->emit_handler_change_events_ = params["emit_handler_change_events"].get<bool>();
        if (params.count("touch_merge_gap_frames")) r->touch_merge_gap_frames_ = std::max(0, params["touch_merge_gap_frames"].get<int>());
        if (params.count("video_label")) r->video_label_ = params["video_label"].get<std::string>();
        if (params.count("fps")) r->fps_ = params["fps"].get<int>();
        if (params.count("dump_every_n")) r->dump_every_n_ = std::max(1, params["dump_every_n"].get<int>());
        if (params.count("debug_log_every_n")) r->debug_log_every_n_ = params["debug_log_every_n"].get<int>();
        if (params.count("shot_result_wait_frames")) r->shot_result_wait_frames_ = std::max(1, params["shot_result_wait_frames"].get<int>());
        if (params.count("score_change_confirm_frames")) r->score_change_confirm_frames_ = std::max(1, params["score_change_confirm_frames"].get<int>());
        if (params.count("score_change_max_delta")) r->score_change_max_delta_ = std::max(1, params["score_change_max_delta"].get<int>());
        if (params.count("team_identity_lock_min_evidence")) r->team_identity_lock_min_evidence_ = std::max(1, params["team_identity_lock_min_evidence"].get<int>());
        if (params.count("team_identity_lock_min_margin")) r->team_identity_lock_min_margin_ = std::max(0, params["team_identity_lock_min_margin"].get<int>());
        if (params.count("scoreboard_team_name_lock_hits")) r->scoreboard_team_name_lock_hits_ = std::max(1, params["scoreboard_team_name_lock_hits"].get<int>());
        if (params.count("include_ocr_game_state")) r->include_ocr_game_state_ = params["include_ocr_game_state"].get<bool>();
        if (params.count("emit_ocr_clock_events")) r->emit_ocr_clock_events_ = params["emit_ocr_clock_events"].get<bool>();
        if (params.count("possession_change_confirm_frames")) r->possession_change_confirm_frames_ = std::max(1, params["possession_change_confirm_frames"].get<int>());
        if (params.count("clock_stopped_threshold_s")) r->clock_stopped_threshold_s_ = params["clock_stopped_threshold_s"].get<float>();
        if (params.count("output_file_summary")) r->output_file_summary_ = params["output_file_summary"].get<std::string>();
        if (params.count("summary_update_every_n")) r->summary_update_every_n_ = std::max(1, params["summary_update_every_n"].get<int>());
        if (params.count("court_seg_slot")) r->court_seg_slot_ = params["court_seg_slot"].get<int>();
        if (params.count("player_seg_slot")) r->player_seg_slot_ = params["player_seg_slot"].get<int>();
        if (params.count("mask_threshold")) r->mask_threshold_ = params["mask_threshold"].get<float>();
        if (params.count("contour_simplify_step")) r->contour_simplify_step_ = std::max(1, params["contour_simplify_step"].get<int>());
        return r;
    }
};

DECLNODE(metadata_dump, MetadataDump)
